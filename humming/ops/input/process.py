import dataclasses
import math
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
    _executor_cache: ClassVar[dict[tuple[object, ...], tuple[object, ...]]] = {}

    inputs: torch.Tensor
    outputs: torch.Tensor | None = None
    inplace: bool = False
    quant_mode: QuantizationMode | str = "none"
    quant_dtype: str | None = None
    quant_group_size: int | None = None
    group_scales: torch.Tensor | None = None
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
            assert self.hidden_size % self.hadamard_block_size == 0, "hadamard_block_size must divide hidden_size"

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
            self.group_scale_dtype = self.scale_dtype_name(self.group_scales)
        elif self.dynamic_scale_mode == "group_token":
            self.group_scale_dtype = "float8e4m3"
        else:
            self.group_scale_dtype = "float32"
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

    def make_executor_key(self) -> tuple[object, ...]:
        from humming.kernel.process_input import ProcessInputKernel

        context = ProcessInputKernel.current_context()
        return (
            context,
            self.inputs.dtype,
            self.quant_dtype,
            self.hidden_size,
            self.quant_group_size,
            self.hadamard_block_size,
            self.tile_size,
            self.layout,
            self.output_width,
            self.num_experts == 1,
            self.expert_layout.dtype if self.expert_layout is not None else None,
            self.indices.dtype if self.indices is not None else None,
            self.zero_invalid,
            self.activation_type,
            self.activation_impl,
            self.quant_mode,
            self.group_scale_dtype,
            self.group_scale_layout,
            self.use_pdl,
            self.schedule_rows,
            self.schedule_width,
            self.working_set_bytes,
        )

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
        properties = torch.cuda.get_device_properties(self.inputs.device)
        expanded_rows = self.num_work_rows * self.output_width
        can_repeat_input = self.activation_type == ActivationType.None_
        can_repeat_input &= self.hadamard_block_size <= 1
        row_limit = properties.multi_processor_count
        if self.dynamic_scale_mode not in ("token", "group_token"):
            row_limit = (row_limit + 1) // 2
        separate_outputs = self.output_width > 1 and can_repeat_input
        separate_outputs &= expanded_rows <= row_limit
        self.schedule_rows = expanded_rows if separate_outputs else self.num_work_rows
        self.schedule_width = 1 if separate_outputs else self.output_width
        self.executor_key = self.make_executor_key()
        self.cached_executor = self._executor_cache.get(self.executor_key)
        if self.cached_executor is not None:
            self.plan, self.phase_kernels, self.scale_kernel = self.cached_executor
            return
        self.plan = select_process_input_plan(self, properties)
        self.plan = dataclasses.replace(self.plan, separate_outputs=separate_outputs)

    def prepare_executor(self) -> None:
        from humming import dtypes
        from humming.kernel.process_input import ProcessInputKernel, ProcessInputScaleKernel

        self.num_input_rows = self.inputs.numel() // self.input_row_size
        uses_token_scale = self.dynamic_scale_mode in ("token", "group_token")
        self.static_tensor_scales = self.token_scales if self.static_mode & STATIC_TENSOR else None
        self.static_group_scales = self.group_scales if self.static_mode & STATIC_GROUP else None
        self.output_token_scales = self.token_scales if uses_token_scale else None

        self.kernel_outputs = self.outputs
        if self.quant_dtype in HIDDEN_BASE and self.outputs.dtype == torch.uint8:
            self.kernel_outputs = self.outputs.view(QUANT_STORAGE[self.quant_dtype][0])
        self.scales_kernel_view = self.group_scales
        if self.group_scales is not None and self.group_scale_dtype == "float8e8m0":
            self.scales_kernel_view = self.group_scales.view(torch.uint8)

        if self.cached_executor is not None:
            return

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
            threads_per_task=self.plan.threads_per_task,
            values_per_thread=self.plan.values_per_thread,
            tokens_per_block=self.plan.tokens_per_block,
            tile_size=self.tile_size,
            layout=kernel_layout,
            layout_width=self.output_width,
            scatter_single_output=self.plan.separate_outputs,
            expert_layout_int64=self.expert_layout is not None and self.expert_layout.dtype == torch.int64,
            index_int64=self.indices is not None and self.indices.dtype == torch.int64,
            zero_invalid=self.zero_invalid and self.layout == LayoutType.GroupedPadded,
            activation_type=self.activation_type,
            activation_impl=self.activation_impl,
            quant_mode=self.quant_mode,
            group_scale_dtype=self.group_scale_dtype,
            scale_layout=self.group_scale_layout,
            work_partition=self.plan.work_partition,
            tiles_per_block=self.plan.tiles_per_block,
            use_pdl=self.use_pdl and self.capability[0] >= 9,
        )
        if self.dynamic_scale_mode == "token" and self.plan.two_stage:
            phases = (1, 2)
        else:
            phases = (0,)
        self.phase_kernels = {
            phase: ProcessInputKernel(**kernel_args, quantization_phase=phase) for phase in phases
        }
        self.scale_kernel = None
        if self.dynamic_scale_mode == "group_token" and self.plan.work_partition == 1:
            self.scale_kernel = ProcessInputScaleKernel(
                **kernel_args,
                quantization_phase=0,
                finalize_tokens_per_block=self.plan.finalize_tokens_per_block,
            )
        self._executor_cache[self.executor_key] = self.plan, self.phase_kernels, self.scale_kernel

    def launch_phase(self, quantization_phase: int, output_scales: torch.Tensor | None) -> None:
        kernel = self.phase_kernels[quantization_phase]
        kernel(
            inputs=self.inputs,
            outputs=self.kernel_outputs,
            static_tensor_scales=self.static_tensor_scales,
            static_group_scales=self.static_group_scales,
            output_scales=output_scales,
            token_scales=self.output_token_scales,
            expert_layout=self.expert_layout,
            indices=self.indices,
            num_input_rows=self.num_input_rows,
            num_output_rows=self.num_output_rows,
            num_work_rows=self.num_work_rows,
            num_experts=self.num_experts,
            max_tokens_per_expert=self.max_tokens_per_expert,
            group_scale_stride=self.group_scale_stride,
        )

    def launch_group_token_finalizer(self, intermediate_scales: torch.Tensor) -> None:
        self.scale_kernel(
            intermediate_scales=intermediate_scales,
            group_scales=self.scales_kernel_view,
            token_scales=self.output_token_scales,
            expert_layout=self.expert_layout,
            indices=self.indices,
            num_input_rows=self.num_input_rows,
            num_output_rows=self.num_output_rows,
            num_work_rows=self.num_work_rows,
            num_experts=self.num_experts,
            max_tokens_per_expert=self.max_tokens_per_expert,
            group_scale_stride=self.group_scale_stride,
        )

    def launch(self) -> None:
        if self.dynamic_scale_mode == "group_token":
            if self.plan.work_partition == 0:
                self.launch_phase(0, self.scales_kernel_view)
                return
            intermediate_scales = torch.empty(
                self.num_output_rows * self.num_quant_groups,
                dtype=torch.uint16,
                device=self.inputs.device,
            )
            self.launch_phase(0, intermediate_scales)
            self.launch_group_token_finalizer(intermediate_scales)
        elif self.dynamic_scale_mode == "token" and self.plan.two_stage:
            self.launch_phase(1, self.output_token_scales)
            self.launch_phase(2, self.output_token_scales)
        else:
            uses_token_scale = self.dynamic_scale_mode in ("token", "group_token")
            output_scales = self.output_token_scales if uses_token_scale else self.scales_kernel_view
            self.launch_phase(0, output_scales)

    def run(self) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        self.normalize()
        self.prepare_layout()
        self.allocate_outputs()
        self.allocate_scales()
        result = self.result()
        if isinstance(self.inputs, FakeTensor):
            return result
        assert self.inputs.is_cuda
        self.select_schedule()
        self.prepare_executor()
        self.launch()
        return result


def process_input(
    inputs: torch.Tensor,
    *,
    outputs: torch.Tensor | None = None,
    inplace: bool = False,
    quant_mode: str = "none",
    quant_dtype: str | None = None,
    quant_group_size: int | None = None,
    group_scales: torch.Tensor | None = None,
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
    operation = _ProcessInput(
        inputs=inputs,
        outputs=outputs,
        inplace=inplace,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        quant_group_size=quant_group_size,
        group_scales=group_scales,
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
    return operation.run()


def _allocate_legacy_group_scales(
    inputs: torch.Tensor,
    group_size: int,
    scale_dtype: str,
    group_scale_layout: GroupScaleLayout,
) -> torch.Tensor:
    rows = inputs.numel() // inputs.size(-1)
    groups = inputs.size(-1) // group_size
    stride = (rows + 3) // 4 * 4 if group_scale_layout != GroupScaleLayout.RowMajor else rows
    if group_scale_layout == GroupScaleLayout.RowMajor:
        shape = tuple(inputs.shape[:-1]) + (groups,)
    elif group_scale_layout == GroupScaleLayout.MMajor:
        shape = (groups, stride)
    else:
        shape = ((groups + 3) // 4, stride, 4)
    return torch.empty(shape, device=inputs.device, dtype=SCALE_TORCH_DTYPE[scale_dtype])


def _legacy_group_scale_view(
    scales: torch.Tensor,
    group_scale_layout: GroupScaleLayout,
) -> torch.Tensor:
    if group_scale_layout != GroupScaleLayout.MxPacked:
        return scales
    return scales.view(torch.int32).reshape(scales.size(0), scales.size(1))


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
    group_scales = _allocate_legacy_group_scales(inputs, group_size, scale_dtype, group_scale_layout)
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
        group_scales=group_scales,
        token_scales=global_scale,
        group_scale_layout=group_scale_layout,
        use_pdl=use_pdl,
    )
    assert result_group_scales is not None
    return quantized, _legacy_group_scale_view(result_group_scales, group_scale_layout)
