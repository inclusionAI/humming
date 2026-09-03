import dataclasses
from typing import ClassVar

import jinja2
import torch

from humming import dtypes
from humming.config.base import BaseHummingConfig
from humming.jit.runtime import KernelRuntime
from humming.ops.input.enums import (
    ActivationType,
    GroupScaleLayout,
    LayoutType,
    QuantizationMode,
    QuantizationPhase,
)

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

struct ProcessInputActivation {
  static constexpr ActivationType kType = ActivationType::{{ activation_type }};
{% if activation_type == "Unary" %}
  CUDA_INLINE static float apply(float a) {
    return {{ activation_impl }};
  }
{% elif activation_type != "None" %}
  CUDA_INLINE static float apply(float a, float b) {
    return {{ activation_impl }};
  }
{% endif %}
};

class KernelConfig {
public:
  using SourceType = {{ source_dtype }};
  using TargetType = {{ target_dtype }};
  using GroupScaleType = {{ group_scale_dtype }};
  using Activation = ProcessInputActivation;
{{ process_input_config }}
};

using RuntimeConfig = ProcessInputConfig<KernelConfig>;

{{ process_input_extern }}
extern "C" __constant__ uint32_t NUM_THREADS = RuntimeConfig::kThreads;
extern "C" __constant__ uint32_t INPUT_ROW_SIZE = RuntimeConfig::kInputRowSize;
extern "C" __constant__ uint32_t OUTPUT_PACKING = RuntimeConfig::kOutputPacking;
extern "C" __constant__ uint32_t SOURCE_DTYPE_ID = {{ source_dtype_config }}::kId;
extern "C" __constant__ uint32_t TARGET_DTYPE_ID = KernelConfig::TargetType::kId;
extern "C" __constant__ uint32_t GROUP_SCALE_DTYPE_ID = {{ group_scale_data_type }}::kId;
extern "C" __constant__ uint32_t LAYOUT = static_cast<uint32_t>(KernelConfig::kLayout);
extern "C" __constant__ uint32_t QUANT_MODE = static_cast<uint32_t>(RuntimeConfig::kQuantization);
extern "C" __constant__ uint32_t QUANTIZATION_PHASE = static_cast<uint32_t>(RuntimeConfig::kPhase);
extern "C" __constant__ uint32_t SCALE_LAYOUT = static_cast<uint32_t>(RuntimeConfig::kGroupScaleLayout);
""")


@dataclasses.dataclass(kw_only=True)
class ProcessInputKernel(KernelRuntime, BaseHummingConfig):
    # Kernel metadata
    name: ClassVar[str] = "process_input_kernel"
    _str2kernel_cache: ClassVar[dict[tuple[object, ...], torch.Tensor]] = {}

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
    quantization_phase: QuantizationPhase | str = QuantizationPhase.Fused

    # Schedule
    threads_per_task: int
    values_per_thread: int
    tokens_per_block: int = 1
    use_tile_partition: bool = False
    tile_size: int
    tiles_per_block: int = 1
    use_pdl: bool = False
    finalize_tokens_per_block: int = 0

    def __post_init__(self):
        self.activation_type = ActivationType(self.activation_type)
        self.layout = LayoutType(self.layout)
        self.quant_mode = QuantizationMode(self.quant_mode)
        self.quantization_phase = QuantizationPhase(self.quantization_phase)
        self.scale_layout = GroupScaleLayout(self.scale_layout)
        super().__post_init__()

    def register_kernel(self):
        from humming.ops import register_process_input_kernel

        kernel_id, kernel_name = register_process_input_kernel(self.kernel_filename)
        assert self.name in kernel_name
        self.kernel_id = kernel_id
        self.kernel_name = kernel_name

    def init_kernel(self):
        is_finalizer = isinstance(self, ProcessInputScaleKernel)
        if is_finalizer:
            assert self.quant_mode == QuantizationMode.DynamicGroupToken
            assert self.use_tile_partition
            assert self.quantization_phase == QuantizationPhase.Fused
            assert 1 <= self.finalize_tokens_per_block <= 32
            finalize_tokens = self.finalize_tokens_per_block
            kernel_expr = f"finalize_group_token_scales_kernel<RuntimeConfig, {finalize_tokens}>"
        else:
            assert self.hidden_size % self.quant_group_size == 0
            assert self.hidden_size % self.tile_size == 0
            assert self.values_per_thread > 0
            assert self.threads_per_task > 0
            assert not self.scatter_single_output or self.layout == LayoutType.Scatter
            threads = self.threads_per_task * self.tokens_per_block
            assert threads % 32 == 0 and 32 <= threads <= 1024
            kernel_expr = "process_input_kernel<RuntimeConfig>"

        group_scale_data_type = dtypes.DataType.from_str(self.group_scale_dtype)
        template_args = self.to_template_args()
        template_args.update(
            process_input_config=self.to_cpp_str(ProcessInputKernel),
            process_input_extern=self.to_extern_cpp_str(ProcessInputKernel),
            source_dtype=_SOURCE_TYPE_CPP[self.source_dtype],
            source_dtype_config=self.source_dtype.to_cpp_str(),
            group_scale_dtype=_SCALE_TYPE_CPP[self.group_scale_dtype],
            group_scale_data_type=group_scale_data_type.to_cpp_str(),
            activation_type=self.activation_type.cpp_name,
        )
        self.code = CODE_TEMPLATE.render(**template_args)
        self.kernel_expr = kernel_expr
        self.prepare()
        self.register_kernel()

    def postprocess_cubin(self, cubin_path: str):
        from humming.utils.cubin import patch_cubin

        mode = None
        if self.target_dtype == dtypes.float8e3m4:
            mode = "cvt_e3m4"
        elif self.target_dtype == dtypes.float4e0m3:
            mode = "cvt_e0m3"
        if mode:
            patch_cubin(cubin_path=cubin_path, mode=mode)

    @classmethod
    def prepare_kernels(cls, kernel_args, intervals, quant_mode, device, cache_key=None):
        if cache_key is not None and cache_key in cls._str2kernel_cache:
            return cls._str2kernel_cache[cache_key]

        plan_specs = {}
        for _, _, plan in intervals:
            if plan in plan_specs:
                continue
            schedule_args = {
                "threads_per_task": plan.threads_per_task,
                "values_per_thread": plan.values_per_thread,
                "tokens_per_block": plan.tokens_per_block,
                "use_tile_partition": plan.use_tile_partition,
                "tiles_per_block": plan.tiles_per_block,
                "scatter_single_output": plan.separate_outputs,
            }
            plan_args = kernel_args | schedule_args
            primary = (cls, plan_args | {"quantization_phase": QuantizationPhase.Fused})
            secondary = None
            if quant_mode.dynamic_scale_mode == "token" and plan.two_stage:
                primary = (cls, plan_args | {"quantization_phase": QuantizationPhase.CollectAbsmax})
                secondary = (cls, plan_args | {"quantization_phase": QuantizationPhase.Quantize})
            elif quant_mode.dynamic_scale_mode == "group_token" and plan.use_tile_partition:
                finalizer_args = plan_args | {"quantization_phase": QuantizationPhase.Fused}
                finalizer_args["finalize_tokens_per_block"] = plan.finalize_tokens_per_block
                secondary = (ProcessInputScaleKernel, finalizer_args)
            plan_specs[plan] = primary, secondary

        def spec_key(spec):
            kernel_type, config = spec
            return kernel_type, tuple(sorted(config.items()))

        unique_specs = {}
        for primary, secondary in plan_specs.values():
            unique_specs.setdefault(spec_key(primary), primary)
            if secondary is not None:
                unique_specs.setdefault(spec_key(secondary), secondary)

        specs = list(unique_specs.values())
        compiled = cls.compile_many(specs, device)
        kernels = dict(zip(unique_specs, compiled, strict=True))
        plan_kernel_ids = {}
        for plan, (primary, secondary) in plan_specs.items():
            primary_id = kernels[spec_key(primary)].kernel_id
            secondary_id = -1
            if secondary is not None:
                secondary_id = kernels[spec_key(secondary)].kernel_id
            plan_kernel_ids[plan] = primary_id, secondary_id

        launch_configs = []
        for first, last, plan in intervals:
            primary_id, secondary_id = plan_kernel_ids[plan]
            launch_configs.extend((first - 1, last, primary_id, secondary_id))
        result = torch.tensor(launch_configs, dtype=torch.int64, device="cpu")
        if cache_key is not None:
            cls._str2kernel_cache[cache_key] = result
        return result


@dataclasses.dataclass(kw_only=True)
class ProcessInputScaleKernel(ProcessInputKernel):
    name: ClassVar[str] = "finalize_group_token_scales_kernel"
    finalize_tokens_per_block: int = 4

    def postprocess_cubin(self, cubin_path: str):
        pass
