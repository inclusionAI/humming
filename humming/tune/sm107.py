import math

from humming.config import GemmType
from humming.tune.sm100 import Sm100Heuristics, Sm100UmmaHeuristics
from humming.utils.math import round_up


class Sm107UmmaHeuristics(Sm100UmmaHeuristics):
    sm_version = 107

    @classmethod
    def _select_cooperative_ctas(cls, layer_config, shape_m, config):
        num_bits = layer_config.a_dtype.num_bits
        if num_bits == 16 or not layer_config.use_raw_weight:
            return super()._select_cooperative_ctas(layer_config, shape_m, config)

        block_m, block_n, _ = config["block_shape"]
        uses_tma = config["use_tma"] and config["use_tma_a"] and config["use_tma_c"]
        if block_m < 128 or not uses_tma or config["num_ctas_per_sm"] != 1:
            return config

        # Raw FP8/FP4 kernels are limited by operand delivery on SM107, so a deep
        # pipeline of short K tiles beats fewer, longer stages.
        block_k, num_stages = 128, 6
        if layer_config.shape_k % block_k or layer_config.shape_k // block_k < 4 * num_stages:
            return config
        if num_bits == 4 and layer_config.shape_n % 256 == 0:
            # FP4 stages hold half the bytes. Spend them on a second weight group
            # that shares each activation tile instead of a taller M tile.
            m_blocks = math.ceil(shape_m / 128)
            block_m = round_up(math.ceil(shape_m / m_blocks), 8)
            block_n = 256

        block_shape = (block_m, block_n, block_k)
        can_cooperate = block_m % 32 == 0 and layer_config.shape_n % (2 * block_n) == 0
        for cta_group_size in (2, 1) if can_cooperate else (1,):
            if cls._fits_resources(
                layer_config, block_shape, num_stages, 1, GemmType.DENSE, cta_group_size, 32
            ):
                return config | {
                    "block_shape": block_shape,
                    "warp_shape": (block_m, 32, block_k),
                    "num_stages": num_stages,
                    "umma_cta_group_size": cta_group_size,
                    "output_chunk_rows": 32,
                }
        return config


class Sm107Heuristics(Sm100Heuristics):
    sm_version: int = 107
    umma_heuristics = Sm107UmmaHeuristics
