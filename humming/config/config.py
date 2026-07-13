import dataclasses
import math
from typing import ClassVar

import torch

from humming import dtypes
from humming.config.base import BaseHummingConfig
from humming.config.enum import GemmType, MmaType, WeightScaleType


@dataclasses.dataclass(kw_only=True)
class LayerConfig(BaseHummingConfig):
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

    # quant param config
    input_scale_group_size: int = 0
    weight_scale_group_size: int = 0
    weight_scale_group_size_n: int = 0
    weight_scale_type: WeightScaleType | None = None
    use_int_weight_scale: bool = False
    use_fused_e8m0_scale: bool = False
    has_zero_point: bool = False
    is_fp_zero_point: bool = False

    # bias config
    has_bias: bool = False

    # mma config
    mma_type: MmaType | None = None

    # packed-K layout (wgmma + 8-bit activation only)
    use_packed_k_layout: bool = False

    _cpp_extra_names: ClassVar[tuple[str, ...]] = (
        "mma_type_id",
        "is_channel_weight_scale",
        "is_block_weight_scale",
        "is_group_weight_scale",
        "is_tensor_weight_scale",
        "has_input_scale",
    )

    def _should_use_mxmma(self, sm_version: int) -> bool:
        if sm_version != 12:
            return False
        if not isinstance(self.a_dtype, dtypes.FloatingPointType):
            return False
        if self.a_dtype.num_bits > 8:
            return False
        if self.bs_dtype not in (dtypes.float8e8m0, dtypes.float8e4m3):
            return False
        group = self.weight_scale_group_size
        if group <= 0:
            return False
        mma_k = 256 // self.a_dtype.num_bits
        return mma_k % group == 0 and mma_k // group in (1, 2, 4)

    def __post_init__(self):
        self.problem_shape = (0, self.shape_n, self.shape_k)
        self.pad_shape = (0, self.pad_shape_n, self.pad_shape_k)

        if self.bs_dtype is None:
            self.bs_dtype = self.c_dtype

        if self.weight_scale_type is None:
            if self.weight_scale_group_size_n > 1:
                self.weight_scale_type = WeightScaleType.BLOCK
            elif self.weight_scale_group_size == 0:
                self.weight_scale_type = WeightScaleType.CHANNEL
            elif self.weight_scale_group_size > 0:
                self.weight_scale_type = WeightScaleType.GROUP

        if isinstance(self.weight_scale_type, str):
            self.weight_scale_type = WeightScaleType(self.weight_scale_type)
        if self.weight_scale_type is None:
            if self.weight_scale_group_size == 0:
                self.weight_scale_type = WeightScaleType.CHANNEL
            elif self.weight_scale_group_size > 0 and self.weight_scale_group_size_n > 1:
                self.weight_scale_type = WeightScaleType.BLOCK
            elif self.weight_scale_group_size > 0:
                self.weight_scale_type = WeightScaleType.GROUP

        for name in ["a", "b", "c", "bs", "as"]:
            value = getattr(self, f"{name}_dtype")
            if isinstance(value, str):
                value = dtypes.DataType.from_str(value)
            setattr(self, f"{name}_dtype", value)

        if self.mma_type is None:
            sm_version = torch.cuda.get_device_capability()[0]
            if sm_version == 9:
                self.mma_type = MmaType.WGMMA
            elif self._should_use_mxmma(sm_version):
                self.mma_type = MmaType.MXMMA
            else:
                self.mma_type = MmaType.MMA
        if isinstance(self.mma_type, str):
            self.mma_type = MmaType(self.mma_type)

        self.has_input_scale = self.a_dtype.num_bits != 16
        if not self.has_input_scale:
            self.as_dtype = None
        elif self.as_dtype is None:
            self.as_dtype = self.bs_dtype if self.mma_type == MmaType.MXMMA else dtypes.float32
        self.is_channel_weight_scale = self.weight_scale_type == WeightScaleType.CHANNEL
        self.is_tensor_weight_scale = self.weight_scale_type in [
            WeightScaleType.TENSOR,
            WeightScaleType.GROUP_TENSOR,
        ]
        self.is_block_weight_scale = self.weight_scale_type == WeightScaleType.BLOCK
        self.is_group_weight_scale = self.weight_scale_type in [
            WeightScaleType.GROUP,
            WeightScaleType.GROUP_TENSOR,
        ]

        if self.use_packed_k_layout:
            assert self.mma_type == MmaType.WGMMA, "use_packed_k_layout requires wgmma"
            assert self.a_dtype.num_bits == 8, "use_packed_k_layout requires 8-bit (fp8/int8) activation"
            assert not self.use_fused_e8m0_scale, "use_packed_k_layout is incompatible with fused-e8m0 scale"

    @property
    def mma_type_id(self):
        assert self.mma_type is not None
        value = self.mma_type.value.lower()
        return ["mma", "wgmma", "umma_placeholder", "mxmma"].index(value)

    @property
    def mxmma_native_mixed(self) -> bool:
        if self.mma_type != MmaType.MXMMA:
            return False
        if self.a_dtype not in (dtypes.float8e4m3, dtypes.float8e5m2):
            return False
        return self.b_dtype in (dtypes.float4e2m1, dtypes.float6e3m2, dtypes.float6e2m3)


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

    use_warp_spec: bool | None = None
    use_mbarrier: bool | None = None
    use_cp_async: bool | None = None

    use_tma: bool | None = None
    use_tma_a: bool | None = None
    use_tma_as: bool | None = None
    use_tma_b: bool | None = None
    use_tma_c: bool | None = None
    use_tma_bs: bool | None = None
    use_tma_bzp: bool | None = None
    use_tma_bias: bool | None = None

    reduce_overlap_last_stage_only: bool = False

    num_write_splits: int = 1
    multi_cast_size_a: int = 1
    multi_cast_size_b: int = 1

    raster_group_m: int = 1

    _cpp_extra_names: ClassVar[tuple[str, ...]] = (
        "num_threads",
        "num_math_threads",
        "num_load_threads",
    )

    _name_map = {
        "use_mbarrier": "kUseMBarrier",
        "use_tma_as": "kUseTmaAS",
        "use_tma_bs": "kUseTmaBS",
        "use_tma_bzp": "kUseTmaBZP",
    }

    def __post_init__(self):
        if self.use_warp_spec is None:
            self.use_warp_spec = False

        if self.use_tma is None:
            self.use_tma = False

        if self.use_mbarrier is None:
            self.use_mbarrier = self.use_tma or self.use_warp_spec

        if self.use_cp_async is None:
            sm_version = torch.cuda.get_device_capability()
            self.use_cp_async = sm_version[0] >= 8

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
