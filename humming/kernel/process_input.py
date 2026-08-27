import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import jinja2
import torch

from humming import dtypes
from humming.jit.runtime import KernelRuntime
from humming.ops.input.enums import ActivationType, GroupScaleLayout, LayoutType, QuantizationMode

_SOURCE_TYPE_CPP = {
    dtypes.float16: "__half",
    dtypes.bfloat16: "__nv_bfloat16",
    dtypes.float32: "float",
}

_SCALE_TYPE_CPP = {
    "float32": "Float32",
    "float8e4m3": "Float8E4M3",
    "float8e8m0": "Float8E8M0",
    "m3bfloat16": "M3BFloat16",
}


CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/process_input.cuh>

{% if activation_type == "None" %}
using ProcessInputActivation = NoActivation;
{% else %}
struct ProcessInputActivation {
  static constexpr ActivationType kType = ActivationType::{{ activation_type }};
  __device__ __forceinline__ static float apply({{ activation_arguments }}) {
    return {{ activation_impl }};
  }
};
{% endif %}

using RuntimeConfig = ProcessInputConfig<
    {{ source_type }}, {{ target_type }},
    InputShape<{{ hidden_size }}, {{ quant_group_size }}, {{ hadamard_block_size }}>,
    InputSchedule<{{ threads_per_task }}, {{ values_per_thread }}, {{ tokens_per_block }},
        static_cast<WorkPartition>({{ work_partition }}), {{ tile_size }},
        {{ tiles_per_block }}, {{ use_pdl }}>,
    InputLayoutConfig<LayoutType::{{ layout }}, {{ layout_width }},
        {{ scatter_single_output }}, {{ expert_layout_int64 }}, {{ index_int64 }}, {{ zero_invalid }}>,
    ProcessInputActivation,
    InputQuantization<QuantizationMode::{{ quant_mode }},
        static_cast<ProcessPhase>({{ quantization_phase }}),
        {{ group_scale_type }}, GroupScaleLayout::{{ scale_layout }}>>;
""")


def _cvt_patch_mode(target_dtype):
    if target_dtype == dtypes.float8e3m4:
        return "cvt_e3m4"
    if target_dtype == dtypes.float4e0m3:
        return "cvt_e0m3"
    return None


def _render_code(kernel) -> str:
    activation_arguments = "float a" if kernel.activation_type == ActivationType.Unary else "float a, float b"
    return CODE_TEMPLATE.render(
        source_type=_SOURCE_TYPE_CPP[kernel.source_dtype],
        target_type=kernel.target_dtype.to_cpp_str(),
        hidden_size=kernel.hidden_size,
        quant_group_size=kernel.quant_group_size,
        hadamard_block_size=kernel.hadamard_block_size,
        threads_per_task=kernel.threads_per_task,
        values_per_thread=kernel.values_per_thread,
        tokens_per_block=kernel.tokens_per_block,
        work_partition=kernel.work_partition,
        tile_size=kernel.tile_size,
        tiles_per_block=kernel.tiles_per_block,
        use_pdl=int(kernel.use_pdl),
        layout=kernel.layout.name,
        layout_width=kernel.layout_width,
        scatter_single_output=int(kernel.scatter_single_output),
        expert_layout_int64=int(kernel.expert_layout_int64),
        index_int64=int(kernel.index_int64),
        zero_invalid=int(kernel.zero_invalid),
        activation_type=kernel.activation_type.cpp_name,
        activation_arguments=activation_arguments,
        activation_impl=kernel.activation_impl,
        quant_mode=kernel.quant_mode.name,
        quantization_phase=kernel.quantization_phase,
        group_scale_type=_SCALE_TYPE_CPP[kernel.group_scale_dtype],
        scale_layout=kernel.scale_layout.name,
    )


@dataclasses.dataclass(kw_only=True)
class ProcessInputKernel(KernelRuntime):
    # Kernel metadata
    name: ClassVar[str] = "process_input_kernel"

    # Input/output types
    source_dtype: dtypes.DataType
    target_dtype: dtypes.DataType

    # Shape
    hidden_size: int
    quant_group_size: int
    hadamard_block_size: int

    # Layout
    layout: LayoutType | str = LayoutType.Normal
    layout_width: int = 1
    scatter_single_output: bool = False
    expert_layout_int64: bool = False
    index_int64: bool = False
    zero_invalid: bool = False

    # Activation
    activation_type: ActivationType | str = ActivationType.None_
    activation_impl: str = ""

    # Quantization
    quant_mode: QuantizationMode | str = QuantizationMode.DynamicGroup
    group_scale_dtype: str = "float32"
    scale_layout: GroupScaleLayout | str = GroupScaleLayout.RowMajor
    quantization_phase: int = 0

    # Schedule
    threads_per_task: int
    values_per_thread: int
    tokens_per_block: int = 1
    work_partition: int = 0
    tile_size: int
    tiles_per_block: int = 1
    use_pdl: bool = False

    def __post_init__(self):
        self.activation_type = ActivationType(self.activation_type)
        self.layout = LayoutType(self.layout)
        self.quant_mode = QuantizationMode(self.quant_mode)
        self.scale_layout = GroupScaleLayout(self.scale_layout)
        super().__post_init__()

    def init_kernel(self):
        assert self.hidden_size % self.quant_group_size == 0
        assert self.hidden_size % self.tile_size == 0
        assert self.values_per_thread > 0
        assert self.threads_per_task > 0
        assert not self.scatter_single_output or self.layout == LayoutType.Scatter
        threads = self.threads_per_task * self.tokens_per_block
        assert threads % 32 == 0 and 32 <= threads <= 1024
        self.code = _render_code(self)
        self.kernel_expr = "process_input_kernel<RuntimeConfig>"
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
        )
        self.prepare()

    def postprocess_cubin(self, cubin_path: str):
        from humming.utils.cubin import patch_cubin

        mode = _cvt_patch_mode(self.target_dtype)
        if mode:
            patch_cubin(cubin_path=cubin_path, mode=mode)

    @property
    def threads(self) -> int:
        return self.threads_per_task * self.tokens_per_block

    def __call__(
        self,
        *,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        static_tensor_scales: torch.Tensor | None,
        static_group_scales: torch.Tensor | None,
        output_scales: torch.Tensor | None,
        token_scales: torch.Tensor | None,
        expert_layout: torch.Tensor | None,
        indices: torch.Tensor | None,
        num_input_rows: int,
        num_output_rows: int,
        num_work_rows: int,
        num_experts: int,
        max_tokens_per_expert: int,
        group_scale_stride: int,
    ) -> None:
        self.check_context()
        assert inputs.is_contiguous() and outputs.is_contiguous()
        assert inputs.dtype == dtypes.torch_dtype_map[self.source_dtype]
        device = inputs.device
        work_rows = num_work_rows * self.layout_width if self.scatter_single_output else num_work_rows
        if self.work_partition == 0:
            grid_x = (work_rows + self.tokens_per_block - 1) // self.tokens_per_block
        else:
            num_tiles = self.hidden_size // self.tile_size
            blocks_per_row = (num_tiles + self.tiles_per_block - 1) // self.tiles_per_block
            grid_x = work_rows * blocks_per_row
        assert 0 < grid_x <= 0x7FFFFFFF

        config = cbd.CUlaunchConfig()
        config.gridDimX = grid_x
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = self.threads
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream
        self.set_pdl_launch_attribute(config, self.use_pdl)
        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            static_tensor_scales.data_ptr() if static_tensor_scales is not None else 0,
            static_group_scales.data_ptr() if static_group_scales is not None else 0,
            output_scales.data_ptr() if output_scales is not None else 0,
            token_scales.data_ptr() if token_scales is not None else 0,
            expert_layout.data_ptr() if expert_layout is not None else 0,
            indices.data_ptr() if indices is not None else 0,
            num_input_rows,
            num_output_rows,
            num_experts,
            max_tokens_per_expert,
            group_scale_stride,
        )
        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)


@dataclasses.dataclass(kw_only=True)
class ProcessInputScaleKernel(ProcessInputKernel):
    name: ClassVar[str] = "finalize_group_token_scales_kernel"
    finalize_tokens_per_block: int = 4

    def init_kernel(self):
        assert self.quant_mode == QuantizationMode.DynamicGroupToken
        assert self.work_partition == 1
        assert self.quantization_phase == 0
        assert 1 <= self.finalize_tokens_per_block <= 32
        self.code = _render_code(self)
        self.kernel_expr = (
            f"finalize_group_token_scales_kernel<RuntimeConfig, {self.finalize_tokens_per_block}>"
        )
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
        )
        self.prepare()

    def postprocess_cubin(self, cubin_path: str):
        pass

    def __call__(
        self,
        *,
        intermediate_scales: torch.Tensor,
        group_scales: torch.Tensor,
        token_scales: torch.Tensor,
        expert_layout: torch.Tensor | None,
        indices: torch.Tensor | None,
        num_input_rows: int,
        num_output_rows: int,
        num_work_rows: int,
        num_experts: int,
        max_tokens_per_expert: int,
        group_scale_stride: int,
    ) -> None:
        self.check_context()
        device = intermediate_scales.device
        work_rows = num_work_rows * self.layout_width if self.scatter_single_output else num_work_rows
        grid_x = (work_rows + self.finalize_tokens_per_block - 1) // self.finalize_tokens_per_block
        assert 0 < grid_x <= 0x7FFFFFFF
        config = cbd.CUlaunchConfig()
        config.gridDimX = grid_x
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = self.finalize_tokens_per_block * 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream
        self.set_pdl_launch_attribute(config, self.use_pdl)
        arg_values = (
            intermediate_scales.data_ptr(),
            group_scales.data_ptr(),
            token_scales.data_ptr(),
            expert_layout.data_ptr() if expert_layout is not None else 0,
            indices.data_ptr() if indices is not None else 0,
            num_input_rows,
            num_output_rows,
            num_experts,
            max_tokens_per_expert,
            group_scale_stride,
        )
        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)
