import dataclasses
from typing import ClassVar

import jinja2

from humming import dtypes
from humming.jit.runtime import KernelRuntime
from humming.ops.input.enums import ActivationType, GroupScaleLayout, LayoutType, QuantizationMode
from humming.ops.input.spec import QUANT_STORAGE

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

extern "C" __constant__ uint32_t PI_NUM_THREADS = RuntimeConfig::kThreads;
extern "C" __constant__ uint32_t PI_SOURCE_DTYPE_ID = {{ source_dtype_id }};
extern "C" __constant__ uint32_t PI_OUTPUT_DTYPE_ID = {{ output_dtype_id }};
extern "C" __constant__ uint32_t PI_GROUP_SCALE_DTYPE_ID = {{ group_scale_dtype_id }};
extern "C" __constant__ uint32_t PI_INPUT_ROW_SIZE = {{ input_row_size }};
extern "C" __constant__ uint32_t PI_HIDDEN_SIZE = RuntimeConfig::kHiddenSize;
extern "C" __constant__ uint32_t PI_QUANT_GROUP_SIZE = RuntimeConfig::kQuantGroupSize;
extern "C" __constant__ uint32_t PI_TILE_SIZE = RuntimeConfig::kTileSize;
extern "C" __constant__ uint32_t PI_TILES_PER_BLOCK = RuntimeConfig::kTilesPerBlock;
extern "C" __constant__ uint32_t PI_TOKENS_PER_BLOCK = RuntimeConfig::kTokensPerBlock;
extern "C" __constant__ uint32_t PI_LAYOUT = static_cast<uint32_t>(LayoutType::{{ semantic_layout }});
extern "C" __constant__ uint32_t PI_LAYOUT_WIDTH = {{ layout_width }};
extern "C" __constant__ uint32_t PI_WORK_PARTITION = static_cast<uint32_t>(RuntimeConfig::kPartition);
extern "C" __constant__ uint32_t PI_QUANT_MODE = static_cast<uint32_t>(RuntimeConfig::kQuantization);
extern "C" __constant__ uint32_t PI_QUANT_PHASE = static_cast<uint32_t>(RuntimeConfig::kPhase);
extern "C" __constant__ uint32_t PI_SCALE_LAYOUT = static_cast<uint32_t>(RuntimeConfig::kGroupScaleLayout);
extern "C" __constant__ uint32_t PI_OUTPUT_PACKING = {{ output_packing }};
extern "C" __constant__ uint32_t PI_FINALIZE_TOKENS = {{ finalize_tokens }};
extern "C" __constant__ uint32_t PI_SCATTER_SINGLE_OUTPUT = {{ scatter_single_output }};
extern "C" __constant__ uint32_t PI_EXPERT_LAYOUT_INT64 = {{ expert_layout_int64 }};
extern "C" __constant__ uint32_t PI_INDEX_INT64 = {{ index_int64 }};
extern "C" __constant__ uint32_t PI_USE_PDL = {{ use_pdl }};
extern "C" __constant__ uint32_t PI_ALLOW_BYTE_OUTPUT = {{ allow_byte_output }};
extern "C" __constant__ uint32_t PI_IS_FINALIZER = {{ is_finalizer }};
""")


def _cvt_patch_mode(target_dtype):
    if target_dtype == dtypes.float8e3m4:
        return "cvt_e3m4"
    if target_dtype == dtypes.float4e0m3:
        return "cvt_e0m3"
    return None


def _render_code(kernel) -> str:
    activation_arguments = "float a" if kernel.activation_type == ActivationType.Unary else "float a, float b"
    quantized = kernel.quant_mode != QuantizationMode.Disabled
    if quantized:
        output_torch_dtype = QUANT_STORAGE[kernel.target_dtype.to_str()][0]
    else:
        output_torch_dtype = dtypes.torch_dtype_map[kernel.source_dtype]
    output_dtype = dtypes.DataType.from_torch_dtype(output_torch_dtype)
    group_scale_dtype = dtypes.DataType.from_str(kernel.group_scale_dtype)
    output_packing = QUANT_STORAGE[kernel.target_dtype.to_str()][1] if quantized else 1
    binary_types = (ActivationType.BinarySplit, ActivationType.BinaryInterleaved)
    binary_activation = kernel.activation_type in binary_types
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
        source_dtype_id=kernel.source_dtype.id(),
        output_dtype_id=output_dtype.id(),
        group_scale_dtype_id=group_scale_dtype.id(),
        input_row_size=kernel.hidden_size * (2 if binary_activation else 1),
        semantic_layout=kernel.semantic_layout.name,
        output_packing=output_packing,
        finalize_tokens=getattr(kernel, "finalize_tokens_per_block", 0),
        allow_byte_output=int(kernel.target_dtype.to_str() in ("float8e3m4",)),
        is_finalizer=int(isinstance(kernel, ProcessInputScaleKernel)),
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
    semantic_layout: LayoutType | str | None = None
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
        if self.semantic_layout is None:
            self.semantic_layout = self.layout
        else:
            self.semantic_layout = LayoutType(self.semantic_layout)
        self.quant_mode = QuantizationMode(self.quant_mode)
        self.scale_layout = GroupScaleLayout(self.scale_layout)
        super().__post_init__()

    def load_cubin(self):
        if self.cubin_loaded:
            return None
        from humming.ops import register_process_input_kernel

        kernel_id, kernel_name = register_process_input_kernel(self.kernel_filename)
        assert self.name in kernel_name
        self.kernel_name = kernel_name
        self.process_input_kernel_id = kernel_id
        self.cubin_loaded = True

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
        self.prepare()

    def postprocess_cubin(self, cubin_path: str):
        from humming.utils.cubin import patch_cubin

        mode = _cvt_patch_mode(self.target_dtype)
        if mode:
            patch_cubin(cubin_path=cubin_path, mode=mode)


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
        self.prepare()

    def postprocess_cubin(self, cubin_path: str):
        pass
