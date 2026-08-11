from humming.tune.sm120 import Sm120Heuristics


class Sm121Heuristics(Sm120Heuristics):
    sm_version: int = 121

    @classmethod
    def should_use_pdl_for_input(cls, layer_config, shape_m: int) -> bool:
        return True

    @classmethod
    def _tune_mxmma_dense(cls, layer_config, config, shape_m, is_mxmma, num_blocks_nk) -> None:
        return

    @classmethod
    def _tune_small_m_dense(cls, layer_config, config, is_wna16: bool) -> None:
        block_m, block_n, block_k = config["block_shape"]
        warp_m, warp_n, warp_k = config["warp_shape"]
        if block_m > 32:
            return

        num_a_bits = layer_config.a_dtype.num_bits
        num_b_bits = layer_config.b_dtype.num_bits
        use_tma = config.get("use_tma", False)
        use_stream_k = config.get("use_stream_k", False)
        shape_k = layer_config.shape_k
        if num_a_bits >= 8 and use_tma and block_k >= warp_k * 2:
            config["block_shape"] = (block_m, block_n, block_k // 2)
            if is_wna16:
                cls._use_tma_b_only(config)
        elif num_b_bits <= 4 and not use_stream_k and block_k == 128 and shape_k % (block_k * 2) == 0:
            if is_wna16:
                config["block_shape"] = (block_m, block_n, block_k * 2)
                config["warp_shape"] = (warp_m, warp_n, 64)
                config["num_stages"] = 3
