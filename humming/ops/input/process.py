import copy
import dataclasses
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .enums import ActivationType, GroupScaleLayout, LayoutType, QuantizationMode
from .plan import select_process_input_plan
from .spec import (
    E8M0_TORCH_DTYPE,
    HIDDEN_BASE,
    QUANT_MODE_SPECS,
    QUANT_STORAGE,
    SCALE_TORCH_DTYPE,
    STATIC_GROUP,
    STATIC_TENSOR,
)


@dataclasses.dataclass(kw_only=True)
class _ProcessInput:
    _family_cache: ClassVar[dict[tuple[object, ...], torch.Tensor]] = {}

    inputs: torch.Tensor
    family_key: tuple[object, ...] | None = None
    outputs: torch.Tensor | None = None
    inplace: bool = False
    quant_mode: QuantizationMode | str = "none"
    quant_dtype: str | None = None
    quant_group_size: int | None = None
    group_scales: torch.Tensor | None = None
    group_scale_dtype: str | None = None
    token_scales: torch.Tensor | None = None
    activation_type: ActivationType | str = "none"
    activation_impl: str | None = None
    hadamard_block_size: int | None = None
    layout: LayoutType | str = "normal"
    expert_layout: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    zero_invalid: bool = False
    group_scale_layout: GroupScaleLayout | str = "row_major"
    use_pdl: bool = False

    def validate_tensor(self, tensor: torch.Tensor) -> None:
        assert tensor.device == self.inputs.device and tensor.is_contiguous()

    @staticmethod
    def scale_dtype_name(tensor: torch.Tensor) -> str:
        if tensor.dtype == torch.float32:
            return "float32"
        if tensor.dtype == torch.float8_e4m3fn:
            return "float8e4m3"
        if tensor.dtype == torch.uint8 or tensor.dtype == E8M0_TORCH_DTYPE:
            return "float8e8m0"
        raise AssertionError(f"unsupported scale dtype: {tensor.dtype}")

    def normalize(self) -> None:
        assert self.inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
        assert self.inputs.is_contiguous() and self.inputs.size(-1) > 0
        self.layout = LayoutType(self.layout)
        self.group_scale_layout = GroupScaleLayout(self.group_scale_layout)
        self.quant_mode = QuantizationMode(self.quant_mode)
        self.dynamic_scale_mode, self.static_mode = QUANT_MODE_SPECS[self.quant_mode]
        self.quantized = self.quant_mode != QuantizationMode.Disabled
        assert self.quantized == (self.quant_dtype is not None), "quant_dtype must match quant_mode"
        valid_quant_dtype = not self.quantized or self.quant_dtype in QUANT_STORAGE
        assert valid_quant_dtype, f"unsupported quant_dtype: {self.quant_dtype}"

        self.input_row_size = self.inputs.size(-1)
        self.activation_type = ActivationType(self.activation_type)
        if self.activation_type == ActivationType.None_:
            assert self.activation_impl in (None, ""), "activation_impl requires activation_type"
            self.activation_impl = ""
        else:
            assert self.activation_impl and "\n" not in self.activation_impl
            assert all(c not in self.activation_impl for c in ";{}"), "activation_impl must be one expression"

        binary_types = (ActivationType.BinarySplit, ActivationType.BinaryInterleaved)
        binary = self.activation_type in binary_types
        if binary:
            assert self.input_row_size % 2 == 0
        self.hidden_size = self.input_row_size // 2 if binary else self.input_row_size

        if self.inplace:
            valid_output = self.outputs is None or self.outputs is self.inputs
            assert valid_output, "inplace does not accept a separate output tensor"
            assert not self.quantized, "inplace does not support quantization"
            unary_activation = self.activation_type in (ActivationType.None_, ActivationType.Unary)
            assert unary_activation, "inplace does not support binary activation"
            allowed_layouts = (LayoutType.Normal, LayoutType.Grouped, LayoutType.GroupedPadded)
            assert self.layout in allowed_layouts, f"inplace does not support {self.layout.value} layout"
            self.outputs = self.inputs

        has_static_group = bool(self.static_mode & STATIC_GROUP)
        uses_group_scale = self.dynamic_scale_mode in ("group", "group_token")
        uses_group_scale |= has_static_group
        self.hadamard_block_size = self.hadamard_block_size or 1
        has_hadamard = self.hadamard_block_size > 1
        if has_hadamard:
            err_msg = f"hadamard_block_size must be <= 512, got {self.hadamard_block_size}"
            assert self.hadamard_block_size <= 512, err_msg
            is_power_of_two = not self.hadamard_block_size & (self.hadamard_block_size - 1)
            assert is_power_of_two, "hadamard_block_size must be a power of two"
            err_msg = "hadamard_block_size must divide hidden_size"
            assert self.hidden_size % self.hadamard_block_size == 0, err_msg

        if uses_group_scale:
            if self.quant_group_size is None or self.quant_group_size == 0:
                self.quant_group_size = min(self.hidden_size & -self.hidden_size, 512)
            is_power_of_two = not self.quant_group_size & (self.quant_group_size - 1)
            err_msg = f"quant_group_size {self.quant_group_size} must be a power of 2 >= 2"
            assert self.quant_group_size >= 2 and is_power_of_two, err_msg
            assert self.quant_group_size <= 512, "per-group quant_group_size must be <= 512"
            assert self.hidden_size % self.quant_group_size == 0, "quant_group_size must divide hidden_size"
            self.tile_size = self.quant_group_size
        else:
            self.quant_group_size = self.hidden_size
            default_tile_size = min(self.hidden_size & -self.hidden_size, 256)
            if not self.quantized and has_hadamard:
                self.tile_size = self.hadamard_block_size
            else:
                self.tile_size = default_tile_size

        self.num_quant_groups = self.hidden_size // self.quant_group_size
        self.num_tiles = self.hidden_size // self.tile_size

    def prepare_layout(self) -> None:
        self.num_experts = 1
        self.max_tokens_per_expert = 1
        self.output_width = 1
        if self.layout == LayoutType.Normal:
            assert self.expert_layout is None and self.indices is None
            self.output_leading_shape = tuple(self.inputs.shape[:-1])
            self.num_work_rows = math.prod(self.output_leading_shape)
        elif self.layout in (LayoutType.Grouped, LayoutType.Permute):
            assert self.inputs.ndim == 2 and self.expert_layout is not None
            assert self.expert_layout.ndim == 1 and self.expert_layout.numel() >= 2
            assert self.expert_layout.dtype in (torch.int32, torch.int64)
            self.validate_tensor(self.expert_layout)
            if self.layout == LayoutType.Grouped:
                assert self.indices is None
                rows = self.inputs.size(0)
            else:
                assert self.indices is not None and self.indices.ndim == 1
                assert self.indices.dtype in (torch.int32, torch.int64)
                self.validate_tensor(self.indices)
                rows = self.indices.numel()
            self.output_leading_shape = (rows,)
            self.num_work_rows = rows
            self.num_experts = self.expert_layout.numel() - 1
        elif self.layout == LayoutType.GroupedPadded:
            assert self.inputs.ndim == 3 and self.expert_layout is not None
            assert self.indices is None
            self.num_experts, self.max_tokens_per_expert, _ = self.inputs.shape
            assert self.expert_layout.shape == (self.num_experts,)
            assert self.expert_layout.dtype in (torch.int32, torch.int64)
            self.validate_tensor(self.expert_layout)
            self.output_leading_shape = (self.num_experts, self.max_tokens_per_expert)
            self.num_work_rows = math.prod(self.output_leading_shape)
        else:
            assert self.layout == LayoutType.Scatter
            assert self.inputs.ndim == 2 and self.expert_layout is None and self.indices is not None
            assert self.indices.ndim == 2 and self.indices.size(0) == self.inputs.size(0)
            assert self.indices.size(1) > 0 and self.indices.dtype == torch.int64
            self.validate_tensor(self.indices)
            output_rows = self.outputs.size(0) if self.outputs is not None else self.indices.numel()
            assert (self.outputs is None or self.outputs.ndim == 2) and output_rows > 0
            self.output_leading_shape = (output_rows,)
            self.num_work_rows = self.inputs.size(0)
            self.output_width = self.indices.size(1)
        self.num_output_rows = math.prod(self.output_leading_shape)

    def allocate_outputs(self) -> None:
        output_storage = QUANT_STORAGE[self.quant_dtype] if self.quant_dtype else None
        output_dtype, packing = output_storage or (self.inputs.dtype, 1)
        assert self.hidden_size % packing == 0
        output_shape = self.output_leading_shape + (self.hidden_size // packing,)
        if self.outputs is None:
            self.outputs = torch.empty(output_shape, dtype=output_dtype, device=self.inputs.device)
            return
        allowed_dtypes = (output_dtype,)
        if self.quant_dtype in HIDDEN_BASE:
            allowed_dtypes += (torch.uint8,)
        assert self.outputs.shape == output_shape and self.outputs.dtype in allowed_dtypes
        self.validate_tensor(self.outputs)

    def allocate_scales(self) -> None:
        static_tensor = bool(self.static_mode & STATIC_TENSOR)
        static_group = bool(self.static_mode & STATIC_GROUP)
        if self.group_scales is not None:
            tensor_scale_dtype = self.scale_dtype_name(self.group_scales)
            if self.group_scale_dtype is not None:
                assert self.group_scale_dtype == tensor_scale_dtype
            self.group_scale_dtype = tensor_scale_dtype
        elif self.group_scale_dtype is None and self.dynamic_scale_mode == "group_token":
            self.group_scale_dtype = "float8e4m3"
        elif self.group_scale_dtype is None:
            self.group_scale_dtype = "float32"
        assert self.group_scale_dtype in SCALE_TORCH_DTYPE
        if static_tensor:
            assert self.token_scales is not None and self.token_scales.dtype == torch.float32
            self.validate_tensor(self.token_scales)
            assert self.token_scales.numel() == self.num_experts
        if static_group:
            assert self.group_scales is not None
            self.validate_tensor(self.group_scales)
            assert self.group_scales.numel() == self.num_experts * self.num_quant_groups

        uses_group = self.dynamic_scale_mode in ("group", "group_token")
        uses_token = self.dynamic_scale_mode in ("token", "group_token")
        row_major_scales = self.group_scale_layout == GroupScaleLayout.RowMajor
        assert uses_group or row_major_scales, "non-row-major scale layout requires dynamic group scales"
        padded_rows = (self.num_output_rows + 3) // 4 * 4
        if self.group_scale_layout == GroupScaleLayout.RowMajor:
            self.group_scale_stride = self.num_output_rows
        else:
            self.group_scale_stride = padded_rows

        if uses_group:
            if self.dynamic_scale_mode == "group_token":
                assert self.group_scale_dtype == "float8e4m3"
            if self.group_scale_layout == GroupScaleLayout.MxPacked:
                assert self.group_scale_dtype in ("float8e4m3", "float8e8m0")
                scale_shape = ((self.num_quant_groups + 3) // 4, self.group_scale_stride, 4)
            elif self.group_scale_layout == GroupScaleLayout.MMajor:
                scale_shape = (self.num_quant_groups, self.group_scale_stride)
            else:
                scale_shape = self.output_leading_shape + (self.num_quant_groups,)
            if self.group_scales is None:
                self.group_scales = torch.empty(
                    scale_shape,
                    dtype=SCALE_TORCH_DTYPE[self.group_scale_dtype],
                    device=self.inputs.device,
                )
            else:
                assert self.group_scales.shape == scale_shape
                self.validate_tensor(self.group_scales)
        elif not static_group:
            assert self.group_scales is None, "group_scales is not used by quant_mode"

        if uses_token:
            if self.token_scales is None:
                self.token_scales = torch.empty(
                    self.output_leading_shape,
                    dtype=torch.float32,
                    device=self.inputs.device,
                )
            else:
                assert self.token_scales.shape == self.output_leading_shape
                assert self.token_scales.dtype == torch.float32
                self.validate_tensor(self.token_scales)
        elif not static_tensor:
            assert self.token_scales is None, "token_scales is not used by quant_mode"

    def result(self) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        returned_outputs = self.outputs
        if self.quant_dtype in HIDDEN_BASE and returned_outputs.dtype != torch.uint8:
            returned_outputs = returned_outputs.view(torch.uint8)
        return returned_outputs, self.group_scales, self.token_scales

    def select_schedule(self) -> None:
        self.capability = torch.cuda.get_device_capability(self.inputs.device)
        if self.quant_dtype in ("float8e4m3", "float8e5m2") and self.capability < (8, 9):
            actual = f"SM{self.capability[0]}{self.capability[1]}"
            raise RuntimeError(f"{self.quant_dtype} output requires SM89 or newer, got {actual}")
        if self.quant_dtype in ("float8e3m4", "float4e0m3", "float4e2m1"):
            if self.capability < (10, 0):
                actual = f"SM{self.capability[0]}{self.capability[1]}"
                raise RuntimeError(f"{self.quant_dtype} output requires SM100 or newer, got {actual}")
        tensors = (self.inputs, self.outputs, self.group_scales, self.token_scales)
        tensors += (self.expert_layout, self.indices)
        self.working_set_bytes = sum(t.numel() * t.element_size() for t in tensors if t is not None)
        self.source_bits = self.inputs.element_size() * 8
        if self.quantized:
            self.target_bits = 8 // QUANT_STORAGE[self.quant_dtype][1]
        else:
            self.target_bits = self.source_bits
        self.properties = torch.cuda.get_device_properties(self.inputs.device)
        separate_outputs = self.separate_scatter_outputs(self.num_work_rows)
        expanded_rows = self.num_work_rows * self.output_width
        self.schedule_rows = expanded_rows if separate_outputs else self.num_work_rows
        self.schedule_width = 1 if separate_outputs else self.output_width
        self.plan = select_process_input_plan(self, self.properties)
        self.plan = dataclasses.replace(self.plan, separate_outputs=separate_outputs)

    def separate_scatter_outputs(self, rows: int) -> bool:
        can_repeat_input = self.activation_type == ActivationType.None_
        can_repeat_input &= self.hadamard_block_size <= 1
        row_limit = self.properties.multi_processor_count
        if self.dynamic_scale_mode not in ("token", "group_token"):
            row_limit = (row_limit + 1) // 2
        separate_outputs = self.output_width > 1 and can_repeat_input
        return separate_outputs and rows * self.output_width <= row_limit

    def plan_for_rows(self, rows: int):
        operation = copy.copy(self)
        operation.num_work_rows = rows
        separate_outputs = self.separate_scatter_outputs(rows)
        operation.schedule_rows = rows * self.output_width if separate_outputs else rows
        operation.schedule_width = 1 if separate_outputs else self.output_width
        bytes_per_row = (self.working_set_bytes + self.num_work_rows - 1) // self.num_work_rows
        operation.working_set_bytes = bytes_per_row * rows
        plan = select_process_input_plan(operation, self.properties)
        return dataclasses.replace(plan, separate_outputs=separate_outputs)

    def plan_intervals(self):
        plans = {self.num_work_rows: self.plan}

        def get_plan(rows: int):
            if rows not in plans:
                plans[rows] = self.plan_for_rows(rows)
            return plans[rows]

        def partition(first: int, last: int):
            if first > last:
                return []
            distance = last - first
            probes = {first, last, first + distance // 4, first + distance // 2, first + 3 * distance // 4}
            probe_plans = {get_plan(row) for row in probes}
            if len(probe_plans) == 1:
                return [(first, last, probe_plans.pop())]
            if distance <= 16:
                return [(row, row, get_plan(row)) for row in range(first, last + 1)]
            middle = (first + last) // 2
            return partition(first, middle) + partition(middle + 1, last)

        small_limit = max(1, 4 * self.properties.multi_processor_count)
        intervals = [(row, row, get_plan(row)) for row in range(1, small_limit + 1)]
        bytes_per_row = max(1, (self.working_set_bytes + self.num_work_rows - 1) // self.num_work_rows)
        l2_rows = max(1, self.properties.L2_cache_size // bytes_per_row)
        maximum = 1 << 30
        cuts = {small_limit, maximum}
        cuts.update(row for row in range(l2_rows - 2, l2_rows + 3) if small_limit < row < maximum)
        first = small_limit + 1
        for last in sorted(cuts):
            if last < first:
                continue
            intervals.extend(partition(first, last))
            first = last + 1

        merged = []
        for first, last, plan in intervals:
            if merged and merged[-1][1] + 1 == first and merged[-1][2] == plan:
                merged[-1] = merged[-1][0], last, plan
            else:
                merged.append((first, last, plan))
        return merged

    def kernel_specs(self, plan):
        from humming import dtypes
        from humming.kernel.process_input import ProcessInputKernel, ProcessInputScaleKernel

        target_dtype = (
            dtypes.DataType.from_str(self.quant_dtype) if self.quant_dtype is not None else dtypes.float32
        )
        single_expert_grouped = self.layout == LayoutType.Grouped and self.num_experts == 1
        kernel_layout = LayoutType.Normal if single_expert_grouped else self.layout
        kernel_args = dict(
            source_dtype=dtypes.DataType.from_torch_dtype(self.inputs.dtype),
            target_dtype=target_dtype,
            hidden_size=self.hidden_size,
            quant_group_size=self.quant_group_size,
            hadamard_block_size=self.hadamard_block_size,
            threads_per_task=plan.threads_per_task,
            values_per_thread=plan.values_per_thread,
            tokens_per_block=plan.tokens_per_block,
            tile_size=self.tile_size,
            layout=kernel_layout,
            semantic_layout=self.layout,
            layout_width=self.output_width,
            scatter_single_output=plan.separate_outputs,
            expert_layout_int64=self.expert_layout is not None and self.expert_layout.dtype == torch.int64,
            index_int64=self.indices is not None and self.indices.dtype == torch.int64,
            zero_invalid=self.zero_invalid and self.layout == LayoutType.GroupedPadded,
            activation_type=self.activation_type,
            activation_impl=self.activation_impl,
            quant_mode=self.quant_mode,
            group_scale_dtype=self.group_scale_dtype,
            scale_layout=self.group_scale_layout,
            work_partition=plan.work_partition,
            tiles_per_block=plan.tiles_per_block,
            use_pdl=self.use_pdl and self.capability[0] >= 9,
        )
        if self.dynamic_scale_mode == "token" and plan.two_stage:
            phases = (1, 2)
        else:
            phases = (0,)
        phase_specs = {}
        for phase in phases:
            phase_specs[phase] = (ProcessInputKernel, kernel_args | {"quantization_phase": phase})
        scale_spec = None
        if self.dynamic_scale_mode == "group_token" and plan.work_partition == 1:
            scale_args = kernel_args | {"quantization_phase": 0}
            scale_args["finalize_tokens_per_block"] = plan.finalize_tokens_per_block
            scale_spec = ProcessInputScaleKernel, scale_args
        if self.dynamic_scale_mode == "token" and plan.two_stage:
            return phase_specs[1], phase_specs[2]
        else:
            return phase_specs[0], scale_spec

    def prepare_launch_configs(self) -> torch.Tensor:
        self.prepare()
        self.select_schedule()
        intervals = self.plan_intervals()
        plan_specs = {plan: self.kernel_specs(plan) for _, _, plan in intervals}

        def spec_key(spec):
            kernel_type, kernel_args = spec
            return kernel_type, tuple(sorted(kernel_args.items()))

        unique_specs = {}
        for primary, secondary in plan_specs.values():
            for spec in (primary, secondary):
                if spec is not None:
                    unique_specs.setdefault(spec_key(spec), spec)

        def compile_kernel(spec):
            kernel_type, kernel_args = spec
            return kernel_type(**kernel_args)

        parallel = len(unique_specs) > 1
        parallel &= os.environ.get("HUMMING_DISABLE_PARALLEL_BUILD", "0") != "1"
        if parallel:
            workers = min(16, len(unique_specs))
            with ThreadPoolExecutor(
                max_workers=workers,
                initializer=torch.cuda.set_device,
                initargs=(self.inputs.device.index,),
            ) as executor:
                compiled = executor.map(compile_kernel, unique_specs.values())
                kernels = dict(zip(unique_specs, compiled, strict=True))
        else:
            kernels = {key: compile_kernel(spec) for key, spec in unique_specs.items()}
        for kernel in kernels.values():
            kernel.load_cubin()

        compiled_plans = {}
        for plan, (primary, secondary) in plan_specs.items():
            primary_id = kernels[spec_key(primary)].process_input_kernel_id
            secondary_kernel = None if secondary is None else kernels[spec_key(secondary)]
            secondary_id = -1 if secondary_kernel is None else secondary_kernel.process_input_kernel_id
            compiled_plans[plan] = primary_id, secondary_id

        compiled_intervals = []
        for first, last, plan in intervals:
            kernel_ids = compiled_plans[plan]
            if compiled_intervals and compiled_intervals[-1][2] == kernel_ids:
                compiled_intervals[-1] = compiled_intervals[-1][0], last, kernel_ids
            else:
                compiled_intervals.append((first, last, kernel_ids))
        configs = []
        for first, last, (primary_id, secondary_id) in compiled_intervals:
            configs.extend((first - 1, last, primary_id, secondary_id))
        self.launch_configs = torch.tensor(configs, dtype=torch.int64, device="cpu")
        if self.family_key is not None:
            self._family_cache[self.family_key] = self.launch_configs
        return self.launch_configs

    def prepare(self) -> None:
        self.normalize()
        self.prepare_layout()
        self.allocate_outputs()
        self.allocate_scales()


def process_input(
    inputs: torch.Tensor,
    *,
    outputs: torch.Tensor | None = None,
    inplace: bool = False,
    quant_mode: str = "none",
    quant_dtype: str | None = None,
    quant_group_size: int | None = None,
    group_scales: torch.Tensor | None = None,
    group_scale_dtype: str | None = None,
    token_scales: torch.Tensor | None = None,
    activation_type: str = "none",
    activation_impl: str | None = None,
    hadamard_block_size: int | None = None,
    layout: str = "normal",
    expert_layout: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    zero_invalid: bool = False,
    group_scale_layout: str = "row_major",
    use_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    layout_type = LayoutType(layout)
    output_width = 1
    if layout_type == LayoutType.Scatter and indices is not None:
        output_width = indices.size(1)
    single_expert = layout_type == LayoutType.Grouped and expert_layout is not None
    single_expert = single_expert and expert_layout.numel() == 2
    # M and ordinary leading dimensions remain runtime values. K, scatter width,
    # and the single-expert specialization change the generated cubin.
    family_key = (
        inputs.device.index,
        inputs.dtype,
        inputs.size(-1),
        outputs.dtype if outputs is not None else None,
        group_scales.dtype if group_scales is not None else None,
        group_scale_dtype,
        token_scales.dtype if token_scales is not None else None,
        expert_layout.dtype if expert_layout is not None else None,
        indices.dtype if indices is not None else None,
        inplace,
        quant_mode,
        quant_dtype,
        quant_group_size,
        activation_type,
        activation_impl,
        hadamard_block_size,
        layout_type,
        output_width,
        single_expert,
        zero_invalid,
        group_scale_layout,
        use_pdl,
    )
    fake = isinstance(inputs, FakeTensor)
    launch_configs = None if fake else _ProcessInput._family_cache.get(family_key)
    operation = None
    if launch_configs is None:
        operation = _ProcessInput(
            inputs=inputs,
            family_key=family_key,
            outputs=outputs,
            inplace=inplace,
            quant_mode=quant_mode,
            quant_dtype=quant_dtype,
            quant_group_size=quant_group_size,
            group_scales=group_scales,
            group_scale_dtype=group_scale_dtype,
            token_scales=token_scales,
            activation_type=activation_type,
            activation_impl=activation_impl,
            hadamard_block_size=hadamard_block_size,
            layout=layout,
            expert_layout=expert_layout,
            indices=indices,
            zero_invalid=zero_invalid,
            group_scale_layout=group_scale_layout,
            use_pdl=use_pdl,
        )
    if fake:
        operation.prepare()
        return operation.result()

    assert inputs.is_cuda
    if operation is not None:
        with torch.cuda.device(inputs.device):
            launch_configs = operation.prepare_launch_configs()
        outputs = operation.outputs
        group_scales = operation.group_scales
        token_scales = operation.token_scales
    result = torch.ops.humming.launch_process_input(
        launch_configs,
        inputs,
        outputs,
        group_scales,
        token_scales,
        expert_layout,
        indices,
        inplace,
    )
    result_output, result_group_scales, result_token_scales = result
    if quant_dtype in HIDDEN_BASE and result_output.dtype != torch.uint8:
        result_output = result_output.view(torch.uint8)
    return result_output, result_group_scales, result_token_scales


def quant_input(
    inputs: torch.Tensor,
    dtype: str,
    scales: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    group_size: int | None = None,
    use_pdl: bool = False,
    m_major_scale: bool = False,
    scale_dtype: str = "float32",
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    is_dynamic = scales is None
    if group_size is None or group_size == 0:
        group_size = inputs.size(-1)
    if not is_dynamic:
        assert global_scale is None and not m_major_scale and scale_dtype == "float32"
        quantized, _, _ = process_input(
            inputs,
            outputs=outputs,
            quant_mode=QuantizationMode.StaticGroup,
            quant_dtype=dtype,
            quant_group_size=group_size,
            group_scales=scales,
            use_pdl=use_pdl,
        )
        return quantized, scales

    group_scale_layout = GroupScaleLayout.RowMajor
    if m_major_scale:
        group_scale_layout = GroupScaleLayout.MMajor
        if scale_dtype == "float8e8m0":
            group_scale_layout = GroupScaleLayout.MxPacked
    quant_mode = (
        QuantizationMode.StaticTensorDynamicGroup
        if global_scale is not None
        else QuantizationMode.DynamicGroup
    )
    quantized, result_group_scales, _ = process_input(
        inputs,
        outputs=outputs,
        quant_mode=quant_mode,
        quant_dtype=dtype,
        quant_group_size=group_size,
        group_scale_dtype=scale_dtype,
        token_scales=global_scale,
        group_scale_layout=group_scale_layout,
        use_pdl=use_pdl,
    )
    assert result_group_scales is not None
    return quantized, result_group_scales
