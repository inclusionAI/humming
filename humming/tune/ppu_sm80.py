import math

import numpy as np

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import current_device
from humming.tune.base import DeviceHeuristics
from humming.utils.smem import estimate_smem_size_layer


class PPUSm80Heuristics(DeviceHeuristics):
    max_smem_size: int = 256 * 1024
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8]
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 80

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_m_major_input_scale: bool = False,
    ):
        max_block_m = 128
        if layer_config.a_dtype.num_bits != 16 and not layer_config.use_fused_e8m0_scale:
            if layer_config.input_scale_group_size > 0 or layer_config.weight_scale_group_size > 0:
                max_block_m = 64
        num_blocks_list = cls.calc_num_block_list(layer_config, shape_m, max_block_m)
        block_shape_m = np.argmin(num_blocks_list).item() * 16 + 16
        num_blocks_m = min(num_blocks_list)

        warp_shape_n = 32
        warp_shape_k = 512 // layer_config.a_dtype.num_bits
        block_shape_n = 256
        num_sms = current_device.sm_count
        num_blocks_n = layer_config.shape_n // block_shape_n

        num_ctas_per_sm = 2
        use_stream_k = False
        num_stages = 3

        while num_blocks_n * num_blocks_m <= num_sms * num_ctas_per_sm:
            if block_shape_n == 64:
                break
            block_shape_n = block_shape_n // 2
            num_blocks_n = num_blocks_n * 2

        num_warps_n = block_shape_n // warp_shape_n
        num_warps_k = 8 // num_warps_n
        block_shape_k = num_warps_k * warp_shape_k
        while layer_config.shape_k % block_shape_k != 0:
            block_shape_k = block_shape_k // 2
            num_warps_k = num_warps_k // 2

        # for small block_shape_m, allow 16 warps per cta
        if block_shape_m <= 64 and layer_config.shape_k % (block_shape_k * 2) == 0:
            block_shape_k = block_shape_k * 2
            num_warps_k = num_warps_k * 2

        # if smem is large enough, try double warp_shape_k and block_shape_k
        if layer_config.shape_k % (block_shape_k * 2) == 0:
            block_shape_new = (block_shape_m, block_shape_n, block_shape_k * 2)
            smem_size = estimate_smem_size_layer(layer_config, block_shape_new, gemm_type, num_stages)
            if smem_size * num_ctas_per_sm < cls.max_smem_size:
                block_shape_k = block_shape_k * 2
                warp_shape_k = warp_shape_k * 2

        # if shape_k is too small, use small block_shape_k and warp_shape_k
        if layer_config.shape_k < block_shape_k * num_stages:
            block_shape_k = 512 // layer_config.a_dtype.num_bits
            warp_shape_k = block_shape_k
            num_warps_k = block_shape_k // warp_shape_k

        # enable stream k only for some speical cases
        use_stream_k = False
        num_blocks_per_cta = num_blocks_n * num_blocks_m / (num_sms * num_ctas_per_sm)
        if num_blocks_per_cta < 3 and num_blocks_per_cta % 1 <= 0.2:
            use_stream_k = True

        # batch invariant: force disable stream k, force num_warps_k = 1
        if use_batch_invariant:
            use_stream_k = False
            if block_shape_k > warp_shape_k:
                warp_shape_k = 1024 // layer_config.a_dtype.num_bits
                block_shape_k = warp_shape_k

        config = {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (block_shape_m, warp_shape_n, warp_shape_k),
            "use_stream_k": use_stream_k,
            "use_f16_accum": use_f16_accum,
            "num_ctas_per_sm": num_ctas_per_sm,
            "num_stages": num_stages,
        }

        return config

    @classmethod
    def calc_num_block_list(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        max_block_m: int,
    ):
        num_blocks_list = []
        if not layer_config.num_experts:
            for i in range(max_block_m // 16):
                block_m = i * 16 + 16
                num_blocks_list.append(math.ceil(shape_m / block_m))
        else:
            random_state = np.random.RandomState(seed=0)
            samples = random_state.randint(0, layer_config.num_experts, size=shape_m)
            counts = np.bincount(samples)
            for i in range(max_block_m // 16):
                block_m = i * 16 + 16
                num_blocks = int(np.ceil(counts * 1.1 / block_m).sum().item())
                num_blocks_list.append(num_blocks)

        return num_blocks_list
