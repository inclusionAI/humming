from humming import dtypes
from humming.config import GemmType
from humming.tune.base import DeviceHeuristics


class Sm75Heuristics(DeviceHeuristics):
    max_smem_size: int = 64 * 1024
    sm_version: int = 75
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8]

    @classmethod
    def get_base_config(
        cls,
        a_dtype: dtypes.DataType,
        b_dtype: dtypes.DataType,
        group_size: int,
        use_f16_accum: bool,
        use_fused_e8m0_scale: bool,
        gemm_type: GemmType,
        shape_k: int,
    ):
        if a_dtype.num_bits == 16:
            warp_shape_k = 64 if b_dtype.num_bits <= 3 else 32
            return {
                "block_shape": (64, 256, warp_shape_k),
                "warp_shape": (64, 64, warp_shape_k),
                "num_ctas_per_sm": 2,
                "output_chunk_rows": 32,
            }
        elif group_size == 0 and b_dtype.num_bits <= 5:
            return {
                "block_shape": (64, 256, 64),
                "warp_shape": (64, 64, 64),
                "num_ctas_per_sm": 2,
                "output_chunk_rows": 32,
            }
        elif group_size == 0 and gemm_type != GemmType.DENSE:
            return {
                "block_shape": (128, 256, 64),
                "warp_shape": (64, 64, 64),
            }
        elif group_size == 0:
            return {
                "block_shape": (128, 256, 64),
                "warp_shape": (128, 32, 64),
                "output_chunk_rows": 64,
            }
        else:
            warp_shape_k = 128 if b_dtype.num_bits <= 5 else 64
            return {
                "block_shape": (64, 256, warp_shape_k),
                "warp_shape": (64, 32, warp_shape_k),
            }
