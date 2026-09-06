import dataclasses
import math

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.device import current_device
from humming.tune.sm8x import Sm80Heuristics
from humming.utils.smem import estimate_smem_size_layer


class Sm100Heuristics(Sm80Heuristics):
    max_smem_size: int = 227 * 1024
    sm_version: int = 100
    b4_allowed_dtypes: list[dtypes.DataType] = [dtypes.int4, dtypes.float4e2m1]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if layer_config.use_mxumma:
            indexed = gemm_type == GemmType.INDEXED
            block_m = 64 if shape_m < 128 * (layer_config.num_experts or 1) else 128
            block_k = 256 if layer_config.shape_k % 256 == 0 else 128
            num_ctas = 1
            tiles = math.ceil(shape_m / block_m) * (layer_config.shape_n // 128)
            if tiles >= 2 * current_device.sm_count:
                smem_size = estimate_smem_size_layer(
                    layer_config, (block_m, 128, 128), gemm_type, 4,
                    warp_shape=(block_m, 32, 128),
                    use_mbarrier=True, use_warp_spec=True,
                )
                if 2 * smem_size <= cls.max_smem_size:
                    block_k, num_ctas = 128, 2
            return {
                "mma_type": MmaType.MXMMA.value,
                "block_shape": (block_m, 128, block_k),
                "warp_shape": (block_m, 32, block_k),
                "num_stages": 4,
                "num_ctas_per_sm": num_ctas,
                "use_warp_spec": True,
                "use_tma": True,
                "use_tma_a": not indexed,
                "use_tma_c": not indexed,
                "use_stream_k": False,
                "use_pdl": False,
                "raster_group_m": 1,
            }
        if layer_config.mma_type != MmaType.UMMA:
            return super().get_config(
                layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type
            )
        common_weight = layer_config.b_dtype in (
            dtypes.uint4, dtypes.uint8, dtypes.float4e2m1,
            dtypes.float8e4m3, dtypes.float8e5m2,
        )
        compatible = (
            layer_config.sm_version // 10 == 10
            and layer_config.a_dtype == layer_config.c_dtype == dtypes.bfloat16
            and layer_config.shape_n % 128 == 0
            and layer_config.shape_k >= 1024
            and layer_config.shape_k % 64 == 0
            and not use_f16_accum
            and not use_batch_invariant
        )
        # Decode and thin expert batches retain the existing MMA schedule.
        if layer_config.num_experts:
            tiles = math.ceil(shape_m / 128) * (layer_config.shape_n // 128)
            # Masked M may describe buffer capacity rather than live routed tokens.
            profitable = (
                gemm_type != GemmType.GROUPED_MASKED
                and layer_config.shape_n >= 1024
                and layer_config.shape_k >= 2048
                and shape_m >= 128 * layer_config.num_experts
                and tiles * 2 >= current_device.sm_count
            )
        else:
            min_m = (
                1024 if layer_config.shape_k <= 1024
                else 512 if layer_config.shape_k > 4096
                else 256
            )
            tiles = math.ceil(shape_m / 64) * (layer_config.shape_n // 128)
            profitable = shape_m >= min_m and tiles * 2 >= current_device.sm_count
        if common_weight and compatible and profitable:
            return cls.get_umma_config(layer_config, shape_m, gemm_type)
        mma_layer = dataclasses.replace(layer_config, mma_type=MmaType.MMA)
        return super().get_config(
            mma_layer, shape_m, use_f16_accum, use_batch_invariant, gemm_type
        ) | {"mma_type": MmaType.MMA.value}

    @classmethod
    def get_umma_config(
        cls, layer_config: LayerConfig, shape_m: int, gemm_type: GemmType
    ):
        if layer_config.num_experts:
            block_m = 128 if shape_m >= 128 * layer_config.num_experts else 64
        else:
            tiles = math.ceil(shape_m / 64) * (layer_config.shape_n // 128)
            block_m = 64 if tiles <= current_device.sm_count else 128
        num_stages = 4 if layer_config.b_dtype == dtypes.uint8 and block_m == 128 else 5
        num_ctas_per_sm = 1
        tiles = math.ceil(shape_m / block_m) * (layer_config.shape_n // 128)
        if block_m == 128 and layer_config.shape_k >= 1024 and tiles >= 2 * current_device.sm_count:
            resident_stages = 5
            if layer_config.b_dtype.num_bits >= 8 or (
                gemm_type == GemmType.INDEXED and layer_config.b_dtype == dtypes.float4e2m1
            ):
                resident_stages = 4
            smem_size = estimate_smem_size_layer(
                layer_config, (block_m, 128, 64), gemm_type, resident_stages,
                warp_shape=(block_m, 32, 64), use_mbarrier=True, use_warp_spec=True,
            )
            if 2 * smem_size <= cls.max_smem_size:
                num_ctas_per_sm = 2
                num_stages = resident_stages
        indexed = gemm_type == GemmType.INDEXED
        return {
            "mma_type": MmaType.UMMA.value,
            "block_shape": (block_m, 128, 64),
            "warp_shape": (block_m, 32, 64),
            "num_stages": num_stages,
            "num_ctas_per_sm": num_ctas_per_sm,
            "use_warp_spec": True,
            "use_tma": True,
            "use_tma_a": not indexed,
            "use_tma_c": not indexed,
            "use_stream_k": False,
            "use_pdl": False,
            "raster_group_m": 1,
        }
