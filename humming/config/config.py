import dataclasses
import functools
import math
from typing import ClassVar

import torch

from humming import dtypes
from humming.config.base import BaseHummingConfig
from humming.config.enum import (
    ActivationType,
    GemmType,
    InputQuantizationMode,
    MmaType,
    ProcessInputLayoutType,
    SmemReuseMode,
    WeightScale2Type,
    WeightScaleType,
)
from humming.device import DeviceInfo, current_device
from humming.utils.math import round_up


@functools.cache
def _cuda_compiler_version(compiler_cls):
    version = compiler_cls.signature().split("+", 1)[1]
    return tuple(int(x) for x in version.split(".")[:2])


@dataclasses.dataclass(kw_only=True, unsafe_hash=True)
class LayerConfig(BaseHummingConfig):
    sm_version: int | None = None

    # shape config
    shape_n: int
    shape_k: int
    pad_shape_n: int = 0
    pad_shape_k: int = 0
    num_experts: int = 0

    # datatype config
    b_dtype: dtypes.DataType
    a_dtype: dtypes.DataType
    c_dtype: dtypes.DataType
    bs_dtype: dtypes.DataType | None = None
    as_dtype: dtypes.DataType | None = None
    input_quant_mode: InputQuantizationMode | str | None = None

    # quant param config
    input_scale_group_size: int = 0
    weight_scale_group_size: int = 0
    weight_scale_group_size_n: int = 0
    weight_scale_type: WeightScaleType | None = None
    weight_scale_2_type: WeightScale2Type | None = None
    use_int_weight_scale: bool | None = None
    use_fused_e8m0_scale: bool | None = None
    has_zero_point: bool = False
    is_fp_zero_point: bool = False

    # bias config
    has_bias: bool = False

    # mma config
    mma_type: MmaType | None = None

    # packed-K layout (wgmma + 8-bit activation + even-bit weight only)
    use_packed_k_layout: bool | None = None

    _cpp_extra_names: ClassVar[tuple[str, ...]] = (
        "mma_type_id",
        "is_channel_weight_scale",
        "is_block_weight_scale",
        "is_group_weight_scale",
        "is_tensor_weight_scale",
        "is_channel_weight_scale_2",
        "is_tensor_weight_scale_2",
        "has_channel_weight_scale",
        "has_tensor_weight_scale",
        "has_input_scale",
        "has_input_scale_2",
        "is_group_input_scale",
        "is_token_input_scale",
        "is_tensor_input_scale",
        "is_token_input_scale_2",
        "is_tensor_input_scale_2",
        "use_native_dequant",
    )

    @property
    def use_native_dequant(self):
        from humming.jit.runtime import KernelRuntime

        cuda_version = _cuda_compiler_version(KernelRuntime._get_compiler())
        assert self.sm_version is not None
        native_low_bit_sm = self.sm_version >= 100
        if self.mma_type not in (MmaType.MMA, MmaType.UMMA):
            return False

        accepted_b_dtype = tuple()
        if self.a_dtype == dtypes.float16:
            if self.sm_version >= 89 and cuda_version >= (11, 8):
                accepted_b_dtype += (dtypes.float8e4m3, dtypes.float8e5m2)
            if native_low_bit_sm and cuda_version >= (12, 7):
                accepted_b_dtype += (dtypes.float4e2m1, dtypes.float6e3m2, dtypes.float6e2m3)

        if self.a_dtype == dtypes.bfloat16 and native_low_bit_sm and cuda_version >= (13, 2):
            accepted_b_dtype += (
                dtypes.float4e2m1,
                dtypes.float6e3m2,
                dtypes.float6e2m3,
                dtypes.float8e4m3,
                dtypes.float8e5m2,
            )

        return self.b_dtype in accepted_b_dtype

    @property
    def mxmma_supported(self):
        assert self.sm_version is not None
        if self.sm_version // 10 != 12:
            return False
        if self.input_quant_mode == InputQuantizationMode.DynamicGroupToken:
            if self.a_dtype not in (dtypes.float4e2m1, dtypes.float4e0m3):
                return False
            if self.input_scale_group_size != 16:
                return False
        is_channel_or_tensor_weight_scale = self.is_channel_weight_scale or self.is_tensor_weight_scale
        if not (self.is_group_weight_scale or is_channel_or_tensor_weight_scale):
            return False
        if (
            self.is_group_weight_scale
            and self.input_scale_group_size > 0
            and self.input_scale_group_size != self.weight_scale_group_size
        ):
            return False
        if self.a_dtype in (dtypes.float8e4m3, dtypes.float8e5m2, dtypes.float8e3m4):
            return self.input_scale_group_size in (0, 32) and (
                is_channel_or_tensor_weight_scale
                or self.weight_scale_group_size == 32
                and self.bs_dtype == dtypes.float8e8m0
            )
        if self.a_dtype in (dtypes.float4e2m1, dtypes.float4e0m3):
            if self.a_dtype == dtypes.float4e0m3 and self.weight_scale_group_size == 32:
                return False

            return self.input_scale_group_size in (0, 16, 32) and (
                is_channel_or_tensor_weight_scale
                or self.weight_scale_group_size == 16
                and self.bs_dtype in (dtypes.float8e8m0, dtypes.float8e4m3)
                or self.weight_scale_group_size == 32
                and self.bs_dtype == dtypes.float8e8m0
            )

        return False

    def __post_init__(self):
        if self.sm_version is None:
            self.sm_version = current_device.sm_version

        self.problem_shape = (0, self.shape_n, self.shape_k)
        self.pad_shape = (0, self.pad_shape_n, self.pad_shape_k)

        if isinstance(self.weight_scale_type, str):
            self.weight_scale_type = WeightScaleType(self.weight_scale_type)
        if self.weight_scale_type is None:
            if self.weight_scale_group_size_n > 1:
                self.weight_scale_type = WeightScaleType.BLOCK
            elif self.weight_scale_group_size == 0:
                self.weight_scale_type = WeightScaleType.CHANNEL
            else:
                self.weight_scale_type = WeightScaleType.GROUP

        if isinstance(self.weight_scale_2_type, str):
            self.weight_scale_2_type = WeightScale2Type(self.weight_scale_2_type)
        if self.weight_scale_2_type is None:
            self.weight_scale_2_type = WeightScale2Type.NONE
        if self.weight_scale_2_type != WeightScale2Type.NONE:
            assert self.weight_scale_type == WeightScaleType.GROUP, (
                "weight_scale_2_type requires weight_scale_type='group'"
            )

        if self.weight_scale_type in [WeightScaleType.CHANNEL, WeightScaleType.TENSOR]:
            assert self.weight_scale_group_size == 0, (
                f"{self.weight_scale_type} requires weight_scale_group_size=0"
            )
        elif self.weight_scale_type in [WeightScaleType.GROUP, WeightScaleType.BLOCK]:
            assert self.weight_scale_group_size > 0, (
                f"{self.weight_scale_type} requires weight_scale_group_size>0"
            )

        for name in ["a", "b", "c", "bs", "as"]:
            value = getattr(self, f"{name}_dtype")
            if isinstance(value, str):
                value = dtypes.DataType.from_str(value)
            setattr(self, f"{name}_dtype", value)

        if isinstance(self.input_quant_mode, str):
            self.input_quant_mode = InputQuantizationMode(self.input_quant_mode)
        elif self.input_quant_mode is None:
            if self.a_dtype.num_bits == 16:
                self.input_quant_mode = InputQuantizationMode.Disabled
            elif self.input_scale_group_size > 0:
                self.input_quant_mode = InputQuantizationMode.DynamicGroup
            else:
                self.input_quant_mode = InputQuantizationMode.DynamicToken

        self.has_input_scale = self.input_quant_mode.should_quantize
        self.has_input_scale_2 = self.input_quant_mode.has_secondary_scale
        self.is_group_input_scale = self.input_quant_mode.has_group_scale
        self.is_token_input_scale = self.input_quant_mode == InputQuantizationMode.DynamicToken
        self.is_tensor_input_scale = self.input_quant_mode == InputQuantizationMode.StaticTensor
        self.is_token_input_scale_2 = self.input_quant_mode == InputQuantizationMode.DynamicGroupToken
        self.is_tensor_input_scale_2 = self.input_quant_mode == InputQuantizationMode.StaticTensorDynamicGroup
        assert self.has_input_scale == (self.a_dtype.num_bits != 16)
        assert self.is_group_input_scale == (self.input_scale_group_size > 0)
        self.bs_dtype = self.bs_dtype or self.c_dtype

        if isinstance(self.b_dtype, dtypes.IntegerType):
            if isinstance(self.a_dtype, dtypes.FloatingPointType):
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=False)
            elif self.a_dtype.num_bits == self.b_dtype.num_bits:
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=True)
            else:
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=False)

        self._update_weight_scale_flags()

        if isinstance(self.mma_type, str):
            self.mma_type = MmaType(self.mma_type)
        elif self.mma_type is None:
            assert self.sm_version is not None
            if self.sm_version // 10 == 9:
                self.mma_type = MmaType.WGMMA
            elif self.mxmma_supported:
                self.mma_type = MmaType.MXMMA
            elif self.sm_version // 10 == 10 and self.a_dtype == self.c_dtype == dtypes.bfloat16:
                from humming.jit.runtime import KernelRuntime

                version = _cuda_compiler_version(KernelRuntime._get_compiler())
                self.mma_type = MmaType.UMMA if version >= (12, 9) else MmaType.MMA
            else:
                self.mma_type = MmaType.MMA
        if self.has_input_scale_2:
            assert self.mma_type == MmaType.MXMMA, f"{self.input_quant_mode.value} requires mma_type='mxmma'"
        if self.mma_type == MmaType.MXMMA and self.is_group_weight_scale and self.input_scale_group_size > 0:
            assert self.input_scale_group_size == self.weight_scale_group_size
        if self.input_quant_mode == InputQuantizationMode.DynamicGroupToken:
            assert self.a_dtype in (dtypes.float4e2m1, dtypes.float4e0m3)
            assert self.input_scale_group_size == 16

        if not self.has_input_scale:
            self.as_dtype = None
        elif self.as_dtype is None:
            if self.mma_type == MmaType.MXMMA and self.input_scale_group_size > 0:
                if self.is_group_weight_scale:
                    self.as_dtype = self.bs_dtype
                elif self.input_scale_group_size == 16:
                    self.as_dtype = dtypes.float8e4m3
                else:
                    self.as_dtype = dtypes.float8e8m0
            else:
                self.as_dtype = dtypes.float32

        if self.input_quant_mode == InputQuantizationMode.DynamicGroupToken:
            assert self.as_dtype == dtypes.float8e4m3
        if self.mma_type == MmaType.MXMMA and self.is_group_input_scale and self.is_group_weight_scale:
            assert self.as_dtype == self.bs_dtype

        is_channel_scale_2 = self.weight_scale_2_type == WeightScale2Type.CHANNEL

        if self.use_fused_e8m0_scale is None:
            has_native_mxf8f6f4 = self.mma_type == MmaType.MXMMA and self.a_dtype == dtypes.float8e4m3
            self.use_fused_e8m0_scale = (
                not has_native_mxf8f6f4
                and self.a_dtype in [dtypes.float8e4m3, dtypes.int8]
                and self.b_dtype in [dtypes.float4e2m1]
                and self.bs_dtype in [dtypes.float8e8m0]
                and self.weight_scale_group_size > 0
            )

        if self.use_int_weight_scale is None:
            self.use_int_weight_scale = (
                not self.use_fused_e8m0_scale
                and self.a_dtype in [dtypes.int8, dtypes.int4]
                and not is_channel_scale_2
                and self.input_scale_group_size == 0
                and self.weight_scale_group_size > 0
                and self.weight_scale_group_size_n == 1
            )

        if self.use_int_weight_scale:
            assert not is_channel_scale_2, "use_int_weight_scale is incompatible with channel weight_scale_2"
            assert self.input_scale_group_size == 0, "use_int_weight_scale requires input_scale_group_size=0"
            self.bs_dtype = self.c_dtype

        if self.use_int_weight_scale or self.use_fused_e8m0_scale:
            self.weight_scale_type = WeightScaleType.GROUP
            # the extracted min-exponent factor becomes the secondary scale:
            # per-channel when channel2 is requested, per-tensor otherwise
            if not is_channel_scale_2:
                self.weight_scale_2_type = WeightScale2Type.TENSOR
            self._update_weight_scale_flags()

        if self.use_packed_k_layout is None:
            self.use_packed_k_layout = (
                self.mma_type == MmaType.WGMMA
                and self.a_dtype.num_bits == 8
                and self.b_dtype.num_bits % 2 == 0
                and not self.use_fused_e8m0_scale
                and self.weight_scale_group_size == 128
            )
        elif self.use_packed_k_layout:
            assert self.mma_type == MmaType.WGMMA, "use_packed_k_layout requires wgmma"
            assert self.a_dtype.num_bits == 8, "use_packed_k_layout requires 8-bit activation"
            assert self.b_dtype.num_bits % 2 == 0, "use_packed_k_layout requires even-bit weight"
            assert not self.use_fused_e8m0_scale, "packed_k_layout is incompatible with fused-e8m0"

        if type(self) is LayerConfig:
            self._config_str = self.to_str()

    def _update_weight_scale_flags(self):
        self.is_group_weight_scale = self.weight_scale_type == WeightScaleType.GROUP
        self.is_block_weight_scale = self.weight_scale_type == WeightScaleType.BLOCK
        self.is_channel_weight_scale = self.weight_scale_type == WeightScaleType.CHANNEL
        self.is_tensor_weight_scale = self.weight_scale_type == WeightScaleType.TENSOR
        self.is_channel_weight_scale_2 = self.weight_scale_2_type == WeightScale2Type.CHANNEL
        self.is_tensor_weight_scale_2 = self.weight_scale_2_type == WeightScale2Type.TENSOR
        self.has_channel_weight_scale = self.is_channel_weight_scale or self.is_channel_weight_scale_2  # noqa
        self.has_tensor_weight_scale = self.is_tensor_weight_scale or self.is_tensor_weight_scale_2

    def to_str(self) -> str:
        if hasattr(self, "_config_str"):
            return self._config_str
        return super().to_str()

    def __setattr__(self, name, value):
        if hasattr(self, "_config_str"):
            raise AttributeError(f"Instance is frozen, cannot set {name}")
        super().__setattr__(name, value)

    def check_device(self, device: int | torch.device) -> None:
        actual_sm = DeviceInfo(device).sm_version
        if actual_sm != self.sm_version:
            raise RuntimeError(
                f"LayerConfig targets sm{self.sm_version}, but the input is on sm{actual_sm}; "
                "transform the layer separately for each GPU architecture"
            )

    @property
    def mma_type_id(self):
        assert self.mma_type is not None
        value = self.mma_type.value.lower()
        return ["mma", "wgmma", "umma", "mxmma"].index(value)

    @property
    def mxmma_native_mixed(self) -> bool:
        if self.mma_type != MmaType.MXMMA:
            return False
        if self.a_dtype not in (dtypes.float8e4m3, dtypes.float8e5m2):
            return False
        return self.b_dtype in (dtypes.float4e2m1, dtypes.float6e3m2, dtypes.float6e2m3)

    @property
    def weight_nbytes(self):
        nbytes1 = self.shape_n * self.shape_k * self.b_dtype.num_bits // 8
        num_groups = self.shape_k / (self.weight_scale_group_size or self.shape_k)
        assert self.bs_dtype is not None
        nbytes2 = self.shape_n * num_groups * self.bs_dtype.num_bits // 8
        nbytes3 = self.shape_n * num_groups * round_up(self.b_dtype.num_bits, 4) // 8
        nbytes = nbytes1 + nbytes2
        if self.has_zero_point and self.is_fp_zero_point:
            nbytes = nbytes + nbytes2
        elif self.has_zero_point:
            nbytes = nbytes + nbytes3
        return nbytes * (self.num_experts or 1)

    @property
    def param_dtype(self):
        if self.c_dtype == dtypes.float16:
            return torch.float16
        elif self.c_dtype == dtypes.bfloat16:
            return torch.bfloat16
        else:
            raise ValueError(f"unsupported c_dtype: {self.c_dtype}")

    @property
    def should_apply_bs_on_c(self):
        if self.use_fused_e8m0_scale:
            return False
        elif self.mma_type in (MmaType.MMA, MmaType.UMMA):
            return self.weight_scale_group_size == 0 or self.a_dtype.num_bits != 16
        elif self.mma_type == MmaType.WGMMA:
            return self.weight_scale_group_size == 0
        elif self.mma_type == MmaType.MXMMA:
            return self.is_channel_weight_scale
        else:
            raise ValueError(f"unsupported mma_type: {self.mma_type}")


@dataclasses.dataclass(kw_only=True)
class ComputeConfig(BaseHummingConfig):
    use_f16_accum: bool = False
    use_batch_invariant: bool = False
    use_m_major_input_scale: bool = False
    gemm_type: GemmType | None = None

    _cpp_extra_names: ClassVar[tuple[str, ...]] = (
        "gemm_type_id",
        "is_indexed_gemm",
        "is_grouped_gemm",
        "is_grouped_contiguous_gemm",
        "is_grouped_masked_gemm",
    )

    def __post_init__(self):
        if isinstance(self.gemm_type, str):
            self.gemm_type = GemmType(self.gemm_type)
        self.is_indexed_gemm = self.gemm_type == GemmType.INDEXED
        self.is_grouped_contiguous_gemm = self.gemm_type == GemmType.GROUPED_CONTIGUOUS
        self.is_grouped_masked_gemm = self.gemm_type == GemmType.GROUPED_MASKED
        self.is_grouped_gemm = self.is_grouped_contiguous_gemm or self.is_grouped_masked_gemm
        if self.is_indexed_gemm:
            assert not self.use_m_major_input_scale, "indexed GEMM does not support m-major input scales"

    @property
    def gemm_type_id(self):
        assert self.gemm_type is not None
        value = self.gemm_type.value.lower()
        return ["dense", "indexed", "grouped_contiguous", "grouped_masked"].index(value)


@dataclasses.dataclass(kw_only=True)
class TuningConfig(BaseHummingConfig):
    block_shape: tuple[int, int, int]
    warp_shape: tuple[int, int, int]

    use_stream_k: bool = True

    num_stages: int = 2
    num_ctas_per_sm: int = 1
    umma_num_dequant_warpgroups: int = 1

    use_warp_spec: bool | None = None
    use_mbarrier: bool | None = None
    use_cp_async: bool | None = None

    use_tma: bool | None = None
    use_tma_a: bool | None = None
    use_tma_as: bool | None = None
    use_tma_as2: bool | None = None
    use_tma_b: bool | None = None
    use_tma_c: bool | None = None
    use_tma_bs: bool | None = None
    use_tma_bs2: bool | None = None
    use_tma_bzp: bool | None = None
    use_tma_bias: bool | None = None

    smem_reuse_mode: SmemReuseMode | str | None = None

    num_write_splits: int = 1
    multi_cast_size_a: int = 1
    multi_cast_size_b: int = 1

    use_pdl: bool = False
    raster_group_m: int = 1

    _cpp_extra_names: ClassVar[tuple[str, ...]] = (
        "num_threads",
        "num_math_threads",
        "num_load_threads",
        "smem_reuse_mode_id",
    )

    _name_map = {
        "use_mbarrier": "kUseMBarrier",
        "use_tma_as": "kUseTmaAS",
        "use_tma_as2": "kUseTmaAS2",
        "use_tma_bs": "kUseTmaBS",
        "use_tma_bs2": "kUseTmaBS2",
        "use_tma_bzp": "kUseTmaBZP",
    }

    @property
    def smem_reuse_mode_id(self):
        return list(SmemReuseMode).index(self.smem_reuse_mode)

    def __post_init__(self):
        if self.smem_reuse_mode is None:
            mma_type = getattr(self, "mma_type", None)
            self.smem_reuse_mode = SmemReuseMode.ALL_STAGES
            if mma_type == MmaType.UMMA:
                self.smem_reuse_mode = SmemReuseMode.NONE
        self.smem_reuse_mode = SmemReuseMode(self.smem_reuse_mode)
        assert self.block_shape[0] <= 256
        if self.use_warp_spec is None:
            self.use_warp_spec = False

        if self.use_tma is None:
            self.use_tma = False

        if self.use_mbarrier is None:
            self.use_mbarrier = self.use_tma or self.use_warp_spec

        if self.use_cp_async is None:
            self.use_cp_async = current_device.sm_major >= 8

        self.num_math_threads = math.prod(self.block_shape) // math.prod(self.warp_shape) * 32
        if self.use_warp_spec:
            self.num_load_threads = 128
            self.num_threads = self.num_math_threads + 128
        else:
            self.num_load_threads = self.num_math_threads
            self.num_threads = self.num_math_threads

        if self.use_tma_as is None:
            self.use_tma_as = False

        for name in dir(self):
            if not name.startswith("use_tma_"):
                continue
            if not self.use_tma:
                assert getattr(self, name) is not True
            if getattr(self, name) is None:
                setattr(self, name, self.use_tma)


@dataclasses.dataclass(kw_only=True, unsafe_hash=True)
class ProcessInputProblemConfig(BaseHummingConfig):
    input_dtype: dtypes.DataType
    hidden_size: int
    quant_mode: InputQuantizationMode = InputQuantizationMode.Disabled
    quant_dtype: dtypes.DataType | None = None
    quant_group_size: int | None = None
    group_scale_dtype: dtypes.DataType | None = None
    use_m_major_input_scale: bool = False
    activation_type: ActivationType = ActivationType.None_
    activation_impl: str | None = None
    hadamard_block_size: int | None = None
    layout: ProcessInputLayoutType = ProcessInputLayoutType.Normal
    scatter_width: int = 1
    zero_invalid: bool = False

    def __post_init__(self):
        self.quant_mode = InputQuantizationMode(self.quant_mode)
        self.activation_type = ActivationType(self.activation_type)
        self.layout = ProcessInputLayoutType(self.layout)
        self.input_dtype = self.input_dtype and dtypes.DataType.from_any(self.input_dtype)
        self.quant_dtype = self.quant_dtype and dtypes.DataType.from_any(self.quant_dtype)
        self.group_scale_dtype = self.group_scale_dtype and dtypes.DataType.from_any(self.group_scale_dtype)
        assert self.hidden_size > 0
        assert self.quant_mode.should_quantize == (self.quant_dtype is not None)

        if self.activation_type != ActivationType.None_:
            assert self.activation_impl, "activation_impl is required"

        self.hadamard_block_size = self.hadamard_block_size or 1
        if self.quant_mode.has_group_scale:
            self.quant_group_size = self.quant_group_size or min(self.hidden_size & -self.hidden_size, 512)
        else:
            self.quant_group_size = self.hidden_size

        if self.group_scale_dtype is None:
            self.group_scale_dtype = dtypes.float32
            if self.quant_mode == InputQuantizationMode.DynamicGroupToken:
                self.group_scale_dtype = dtypes.float8e4m3

    @property
    def input_row_size(self) -> int:
        return self.hidden_size * (2 if self.activation_type.is_binary else 1)

    @property
    def input_torch_dtype(self) -> torch.dtype:
        return dtypes.torch_dtype_map[self.input_dtype]

    @property
    def output_packing(self) -> int:
        return 8 // self.quant_dtype.num_bits if self.quant_dtype is not None else 1

    @property
    def output_row_size(self) -> int:
        return self.hidden_size // self.output_packing

    @property
    def output_torch_dtype(self) -> torch.dtype:
        dtype = self.quant_dtype or self.input_dtype
        return dtypes.torch_dtype_map.get(dtype, torch.uint8)

    @property
    def group_scale_torch_dtype(self) -> torch.dtype:
        if self.group_scale_dtype.num_bits == 8:
            return torch.int32
        return dtypes.torch_dtype_map[self.group_scale_dtype]

    def get_group_scale_shape(self, rows: int) -> tuple[int, int]:
        groups = self.hidden_size // self.quant_group_size
        if self.group_scale_dtype.num_bits == 8:
            groups = (groups + 3) // 4
        if self.use_m_major_input_scale:
            return groups, round_up(rows, 4)
        return rows, groups


@dataclasses.dataclass(kw_only=True, unsafe_hash=True)
class ProcessInputTuningConfig(BaseHummingConfig):
    threads_per_task: int
    values_per_thread: int
    tokens_per_block: int = 1
    use_tile_partition: bool = False
    separate_outputs: bool = False
    two_stage: bool = False
    finalize_tokens_per_block: int = 4
    use_pdl: bool = False

    @property
    def threads(self) -> int:
        return self.threads_per_task * self.tokens_per_block

    @property
    def columns_per_task(self) -> int:
        return self.threads_per_task * self.values_per_thread
