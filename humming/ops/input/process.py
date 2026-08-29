import copy
import dataclasses
import math

import torch
from torch._subclasses.fake_tensor import FakeTensor

from humming import dtypes

from .enums import ActivationType, GroupScaleLayout, LayoutType, QuantizationMode
from .plan import select_process_input_plan

QUANT_DTYPE_MIN_CAPABILITY = {
    dtypes.int4: (7, 5),
    dtypes.int8: (7, 5),
    dtypes.float8e4m3: (8, 9),
    dtypes.float8e5m2: (8, 9),
    dtypes.float4e0m3: (10, 0),
    dtypes.float4e2m1: (10, 0),
    dtypes.float8e3m4: (10, 0),
}


@dataclasses.dataclass(kw_only=True)
class _ProcessInput:
    inputs: torch.Tensor
    outputs: torch.Tensor | None = None
    inplace: bool = False
    quant_mode: QuantizationMode | str = "none"
    quant_dtype: dtypes.DataType | None = None
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
        scale_dtype = dtypes.DataType.from_torch_dtype(tensor.dtype)
        supported = (dtypes.float32, dtypes.float8e4m3, dtypes.float8e8m0)
        assert scale_dtype in supported, f"unsupported scale dtype: {tensor.dtype}"
        return scale_dtype.to_str()

    def normalize(self) -> None:
        assert self.inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)
        assert self.inputs.is_contiguous() and self.inputs.size(-1) > 0
        self.layout = LayoutType(self.layout)
        self.group_scale_layout = GroupScaleLayout(self.group_scale_layout)
        self.quant_mode = QuantizationMode(self.quant_mode)
        self.quantized = self.quant_mode.quantized
        assert self.quantized == (self.quant_dtype is not None), "quant_dtype must match quant_mode"
        valid_quant_dtype = not self.quantized or self.quant_dtype in QUANT_DTYPE_MIN_CAPABILITY
        assert valid_quant_dtype, f"unsupported quant_dtype: {self.quant_dtype}"

        self.input_row_size = self.inputs.size(-1)
        self.activation_type = ActivationType(self.activation_type)
        if self.activation_type == ActivationType.None_:
            assert self.activation_impl in (None, ""), "activation_impl requires activation_type"
            self.activation_impl = ""
        else:
            assert self.activation_impl, "activation_impl is required"

        binary_types = (ActivationType.BinarySplit, ActivationType.BinaryInterleaved)
        binary = self.activation_type in binary_types
        if binary:
            assert self.input_row_size % 2 == 0
        self.hidden_size = self.input_row_size // (2 if binary else 1)

        if self.inplace:
            valid_output = self.outputs is None or self.outputs is self.inputs
            assert valid_output, "inplace does not accept a separate output tensor"
            assert not self.quantized, "inplace does not support quantization"
            unary_activation = self.activation_type in (ActivationType.None_, ActivationType.Unary)
            assert unary_activation, "inplace does not support binary activation"
            allowed_layouts = (LayoutType.Normal, LayoutType.Grouped, LayoutType.GroupedPadded)
            assert self.layout in allowed_layouts, f"inplace does not support {self.layout.value} layout"
            self.outputs = self.inputs

        uses_group_scale = self.quant_mode.uses_group_scale
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
        output_dtype = self.inputs.dtype
        if self.quant_dtype is not None:
            output_dtype = dtypes.torch_dtype_map.get(self.quant_dtype, torch.uint8)
        packing = 8 // self.quant_dtype.num_bits if self.quant_dtype is not None else 1
        assert self.hidden_size % packing == 0
        output_shape = self.output_leading_shape + (self.hidden_size // packing,)
        if self.outputs is None:
            self.outputs = torch.empty(output_shape, dtype=output_dtype, device=self.inputs.device)
            return
        assert self.outputs.shape == output_shape and self.outputs.dtype == output_dtype
        self.validate_tensor(self.outputs)

    def allocate_scales(self) -> None:
        static_tensor = self.quant_mode.has_static_tensor_scale
        if self.group_scales is not None:
            tensor_scale_dtype = self.scale_dtype_name(self.group_scales)
            if self.group_scale_dtype is not None:
                assert self.group_scale_dtype == tensor_scale_dtype
            self.group_scale_dtype = tensor_scale_dtype
        elif self.group_scale_dtype is None and self.quant_mode.dynamic_scale_mode == "group_token":
            self.group_scale_dtype = "float8e4m3"
        elif self.group_scale_dtype is None:
            self.group_scale_dtype = "float32"
        group_scale_dtype = dtypes.DataType.from_str(self.group_scale_dtype)
        supported = (dtypes.float32, dtypes.float8e4m3, dtypes.float8e8m0)
        assert group_scale_dtype in supported
        if static_tensor:
            assert self.token_scales is not None and self.token_scales.dtype == torch.float32
            self.validate_tensor(self.token_scales)
            assert self.token_scales.numel() == self.num_experts
        uses_group = self.quant_mode.uses_group_scale
        uses_token = self.quant_mode.has_dynamic_token_scale
        row_major_scales = self.group_scale_layout == GroupScaleLayout.RowMajor
        assert uses_group or row_major_scales, "non-row-major scale layout requires dynamic group scales"
        padded_rows = (self.num_output_rows + 3) // 4 * 4
        if self.group_scale_layout == GroupScaleLayout.RowMajor:
            self.group_scale_stride = self.num_output_rows
        else:
            self.group_scale_stride = padded_rows

        if uses_group:
            if self.quant_mode.dynamic_scale_mode == "group_token":
                assert group_scale_dtype == dtypes.float8e4m3
            if self.group_scale_layout == GroupScaleLayout.MxPacked:
                assert group_scale_dtype in (dtypes.float8e4m3, dtypes.float8e8m0)
                scale_shape = ((self.num_quant_groups + 3) // 4, self.group_scale_stride, 4)
            elif self.group_scale_layout == GroupScaleLayout.MMajor:
                scale_shape = (self.num_quant_groups, self.group_scale_stride)
            else:
                scale_shape = self.output_leading_shape + (self.num_quant_groups,)
            if self.group_scales is None:
                dtype = dtypes.torch_dtype_map[group_scale_dtype]
                self.group_scales = torch.empty(scale_shape, dtype=dtype, device=self.inputs.device)
            else:
                assert self.group_scales.shape == scale_shape
                self.validate_tensor(self.group_scales)
        else:
            assert self.group_scales is None, "group_scales is not used by quant_mode"

        if uses_token:
            if self.token_scales is None:
                shape = self.output_leading_shape
                self.token_scales = torch.empty(shape, dtype=torch.float32, device=self.inputs.device)
            else:
                assert self.token_scales.shape == self.output_leading_shape
                assert self.token_scales.dtype == torch.float32
                self.validate_tensor(self.token_scales)
        elif not self.quant_mode.uses_token_scale:
            assert self.token_scales is None, "token_scales is not used by quant_mode"

    def result(self) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return self.outputs, self.group_scales, self.token_scales

    def select_schedule(self) -> None:
        self.properties = torch.cuda.get_device_properties(self.inputs.device)
        self.capability = (self.properties.major, self.properties.minor)
        minimum_capability = QUANT_DTYPE_MIN_CAPABILITY.get(self.quant_dtype)
        if minimum_capability is not None and self.capability < minimum_capability:
            actual = f"SM{self.capability[0]}{self.capability[1]}"
            minimum = f"SM{minimum_capability[0]}{minimum_capability[1]}"
            raise RuntimeError(f"{self.quant_dtype} output requires {minimum} or newer, got {actual}")
        tensors = (self.inputs, self.outputs, self.group_scales, self.token_scales)
        tensors += (self.expert_layout, self.indices)
        self.working_set_bytes = sum(t.numel() * t.element_size() for t in tensors if t is not None)
        self.source_bits = self.inputs.element_size() * 8
        if self.quantized:
            self.target_bits = self.quant_dtype.num_bits
        else:
            self.target_bits = self.source_bits
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
        if not self.quant_mode.has_dynamic_token_scale:
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
                merged[-1] = (merged[-1][0], last, plan)
            else:
                merged.append((first, last, plan))
        return merged

    def kernel_config(self):
        target_dtype = self.quant_dtype or dtypes.float32
        kernel_args = dict(
            source_dtype=dtypes.DataType.from_torch_dtype(self.inputs.dtype),
            target_dtype=target_dtype,
            hidden_size=self.hidden_size,
            quant_group_size=self.quant_group_size,
            hadamard_block_size=self.hadamard_block_size,
            tile_size=self.tile_size,
            layout=self.layout,
            layout_width=self.output_width,
            expert_layout_int64=self.expert_layout is not None and self.expert_layout.dtype == torch.int64,
            index_int64=self.indices is not None and self.indices.dtype == torch.int64,
            zero_invalid=self.zero_invalid and self.layout == LayoutType.GroupedPadded,
            activation_type=self.activation_type,
            activation_impl=self.activation_impl,
            quant_mode=self.quant_mode,
            group_scale_dtype=self.group_scale_dtype,
            scale_layout=self.group_scale_layout,
            use_pdl=self.use_pdl and self.capability[0] >= 9,
        )
        return kernel_args

    def prepare_launch_configs(self, cache_key=None) -> torch.Tensor:
        from humming.kernel.process_input import ProcessInputKernel

        self.prepare()
        self.select_schedule()
        intervals = self.plan_intervals()
        return ProcessInputKernel.prepare_kernels(
            self.kernel_config(),
            intervals,
            self.quant_mode,
            self.inputs.device,
            cache_key,
        )

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
    quant_dtype: str | dtypes.DataType | None = None,
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
    from humming.kernel.process_input import ProcessInputKernel

    quant_dtype = None if quant_dtype is None else dtypes.DataType.from_str(quant_dtype)
    layout_type = LayoutType(layout)
    output_width = 1
    if layout_type == LayoutType.Scatter and indices is not None:
        output_width = indices.size(1)
    # M and ordinary leading dimensions remain runtime values. K and scatter
    # width change the generated cubin.
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
        zero_invalid,
        group_scale_layout,
        use_pdl,
    )
    fake = isinstance(inputs, FakeTensor)
    launch_configs = None if fake else ProcessInputKernel._str2kernel_cache.get(family_key)
    operation = None
    if launch_configs is None:
        operation = _ProcessInput(
            inputs=inputs,
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
            launch_configs = operation.prepare_launch_configs(family_key)
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
    return result


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
    if group_size is None or group_size == 0:
        group_size = inputs.size(-1)

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
        group_scales=scales,
        group_scale_dtype=scale_dtype,
        token_scales=global_scale,
        group_scale_layout=group_scale_layout,
        use_pdl=use_pdl,
    )
    assert result_group_scales is not None
    return quantized, result_group_scales
