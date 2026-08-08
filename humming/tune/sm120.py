import math

from humming import dtypes
from humming.config import GemmType, MmaType
from humming.tune.sm8x import Sm89Heuristics
from humming.utils.smem import estimate_smem_size_layer


class Sm120Heuristics(Sm89Heuristics):
    sm_version: int = 120
    max_smem_size: int = 99 * 1024
    b8_allowed_dtypes: list[dtypes.DataType] = [
        dtypes.int8,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
        dtypes.float8e3m4,
    ]
    b4_allowed_dtypes: list[dtypes.DataType] = [dtypes.float4e2m1, dtypes.float4e0m3]

    @classmethod
    def should_use_pdl_for_input(cls, layer_config, shape_m: int) -> bool:
        return layer_config.shape_n >= 4096 and shape_m <= 32

    @classmethod
    def _fit_dense_block_m_to_grid(
        cls,
        block_shape: tuple[int, int],
        layer_config,
        shape_m: int,
        gemm_type: GemmType,
        num_ctas_per_sm: int,
    ) -> int:
        block_shape_m, block_shape_n = block_shape
        unsupported = gemm_type != GemmType.DENSE or layer_config.mma_type not in (MmaType.MMA, MmaType.MXMMA)
        unsuitable_m = block_shape_m <= 16 or shape_m <= block_shape_m <= 32
        if unsupported or unsuitable_m:
            return block_shape_m

        min_grid_blocks = math.ceil(cls.get_num_sms() * num_ctas_per_sm / 2)
        num_blocks_n = layer_config.shape_n // block_shape_n
        num_blocks_m = math.ceil(shape_m / block_shape_m)
        num_blocks = num_blocks_n * num_blocks_m
        if num_blocks >= min_grid_blocks:
            return block_shape_m

        max_num_blocks_m = num_blocks_m * 4
        target_grid_blocks = min_grid_blocks if num_blocks * 4 >= min_grid_blocks else num_blocks * 2
        candidates = [
            candidate
            for candidate in range(16, block_shape_m, 16)
            if num_blocks_n * math.ceil(shape_m / candidate) >= target_grid_blocks
            and math.ceil(shape_m / candidate) <= max_num_blocks_m
        ]
        return min(
            candidates,
            key=lambda candidate: (math.ceil(shape_m / candidate) * candidate, -candidate),
            default=block_shape_m,
        )

    @classmethod
    def _is_mxmma(cls, a_dtype, group_size, use_fused_e8m0_scale) -> bool:
        return (
            a_dtype.is_floating_point_type
            and a_dtype.num_bits <= 8
            and group_size > 0
            and not use_fused_e8m0_scale
        )

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
        if cls._is_mxmma(a_dtype, group_size, use_fused_e8m0_scale):
            block_k = 512 // a_dtype.num_bits
            if a_dtype.num_bits == 8 and b_dtype.num_bits < a_dtype.num_bits:
                return {
                    "block_shape": (112, 256, block_k),
                    "warp_shape": (112, 32, block_k),
                    "num_stages": 2,
                }
            return {
                "block_shape": (256, 128, block_k),
                "warp_shape": (128, 32, block_k),
                "num_stages": 2,
            }
        if a_dtype.is_floating_point_type and a_dtype.num_bits == 16 and not use_f16_accum:
            return {
                "block_shape": (128, 256, 64),
                "warp_shape": (128, 32, 64),
                "num_stages": 2,
            }
        return super().get_base_config(
            a_dtype, b_dtype, group_size, use_f16_accum, use_fused_e8m0_scale, gemm_type, shape_k
        )

    @classmethod
    def get_config(
        cls,
        layer_config,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        config = super().get_config(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        if use_batch_invariant:
            return config

        a = layer_config.a_dtype
        is_wna16 = a.is_floating_point_type and a.num_bits == 16 and not use_f16_accum
        if a.is_floating_point_type and a.num_bits <= 8 and not layer_config.use_fused_e8m0_scale:
            config["use_tma"] = True
            config["use_warp_spec"] = True
        elif is_wna16:
            config["use_tma"] = True
            block_shape = config["block_shape"]
            warp_shape = config["warp_shape"]
            m_warps = block_shape[0] // warp_shape[0]
            n_warps = block_shape[1] // warp_shape[1]
            k_warps = block_shape[2] // warp_shape[2]
            num_math_threads = m_warps * n_warps * k_warps * 32
            config["use_warp_spec"] = num_math_threads % 128 == 0
            config["num_stages"] = cls._fit_num_stages(layer_config, config, gemm_type, reduce_overlap=False)

        if gemm_type == GemmType.INDEXED:
            config["use_tma_a"] = False
            config["use_tma_c"] = False
            config["use_warp_spec"] = False

        if gemm_type != GemmType.DENSE and is_wna16 and layer_config.b_dtype.num_bits <= 8:
            tokens_per_expert = shape_m / layer_config.num_experts
            if tokens_per_expert >= 96:
                cls._use_m_tile(layer_config, config, gemm_type, 128)
            elif tokens_per_expert >= 48:
                cls._use_m_tile(layer_config, config, gemm_type, 64)

        num_b_bits = layer_config.b_dtype.num_bits
        block_m = config["block_shape"][0]
        if gemm_type == GemmType.GROUPED_MASKED and is_wna16 and num_b_bits <= 8 and block_m <= 64:
            cls._use_tma_b_only(config)

        group_size = layer_config.input_scale_group_size or layer_config.weight_scale_group_size
        is_mxmma = cls._is_mxmma(layer_config.a_dtype, group_size, layer_config.use_fused_e8m0_scale)
        if gemm_type != GemmType.INDEXED and is_mxmma:
            num_stages = cls._fit_num_stages(layer_config, config, gemm_type, reduce_overlap=True)
            config["num_stages"] = num_stages
            config["reduce_overlap_last_stage_only"] = True

        if gemm_type == GemmType.DENSE:
            num_blocks_n = layer_config.shape_n // config["block_shape"][1]
            num_blocks_k = layer_config.shape_k // config["block_shape"][2]
            num_blocks_nk = num_blocks_n * num_blocks_k
            config.pop("num_sms", None)

            is_small_m = shape_m <= config["block_shape"][0] <= 32
            if is_small_m and num_blocks_nk < cls.get_num_sms() * 3:
                config["num_stages"] = 3

            if is_small_m and num_blocks_nk < cls.get_num_sms() * 2:
                config["num_stages"] = 2
                config["use_warp_spec"] = False
                config["use_tma"] = False

            cls._tune_small_m_dense(layer_config, config, shape_m, is_wna16)
            cls._tune_mxmma_dense(layer_config, config, shape_m, is_mxmma, num_blocks_nk)
            config["use_stream_k"] = cls._should_use_stream_k(layer_config, config, shape_m)
            use_stream_k = config["use_stream_k"]
            rebalance_stream_k = cls._should_rebalance_stream_k(layer_config, config, shape_m)
            rebalance_stream_k = rebalance_stream_k and use_stream_k
            block_m, block_n, block_k = config["block_shape"]
            num_output_tiles = math.ceil(shape_m / block_m) * math.ceil(layer_config.shape_n / block_n)
            num_sms = cls.get_num_sms()
            if rebalance_stream_k and block_m <= 16 and num_output_tiles * 3 >= num_sms:
                config["use_stream_k"] = False

            moderate_nk_grid = cls.sm_version == 120 and num_sms * 2 <= num_blocks_nk < num_sms * 3
            if is_small_m and not config["use_stream_k"] and moderate_nk_grid:
                config["use_tma"] = False
                config["use_warp_spec"] = False
                for key in [key for key in config if key.startswith("use_tma_")]:
                    config.pop(key)
                warp_m, warp_n, warp_k = config["warp_shape"]
                config["warp_shape"] = (warp_m, warp_n, min(block_k, warp_k * 2))

            if config["use_stream_k"] and block_m <= 16:
                warp_m, warp_n, warp_k = config["warp_shape"]
                mn_warps = block_m // warp_m * (block_n // warp_n)
                target_k_warps = max(1, 4 // mn_warps)
                config["warp_shape"] = (warp_m, warp_n, max(warp_k, block_k // target_k_warps))

            cls._rebalance_dense_warps(layer_config, config, shape_m)

        return config

    @classmethod
    def _tune_small_m_dense(cls, layer_config, config, shape_m: int, is_wna16: bool) -> None:
        block_m, block_n, block_k = config["block_shape"]
        if shape_m > block_m or block_m > 32 or not is_wna16:
            return

        use_stream_k = config.get("use_stream_k", False)
        shape_k = layer_config.shape_k

        if config.get("use_tma", False):
            cls._use_tma_b_only(config)
        elif not use_stream_k and block_k == 128 and shape_k % (block_k * 2) == 0:
            warp_m, warp_n, _ = config["warp_shape"]
            config["block_shape"] = (block_m, block_n, block_k * 2)
            config["warp_shape"] = (warp_m, warp_n, 64)
            config["num_stages"] = 3

    @classmethod
    def _tune_mxmma_dense(cls, layer_config, config, shape_m, is_mxmma, num_blocks_nk) -> None:
        block_m, block_n, _ = config["block_shape"]
        num_b_bits = layer_config.b_dtype.num_bits
        num_a_bits = layer_config.a_dtype.num_bits
        if not is_mxmma or num_a_bits != 8 or num_b_bits >= 8:
            return

        shape_n = layer_config.shape_n
        shape_k = layer_config.shape_k
        if shape_m < 128 or block_n >= 256 or shape_n % 256 or shape_k % 64:
            return

        block_m = max(block_m, 96)
        config["block_shape"] = (block_m, 256, 64)
        config["warp_shape"] = (block_m, 32, 64)
        config["num_stages"] = 4 if num_blocks_nk >= cls.get_num_sms() * 3 else 2

    @classmethod
    def _should_use_stream_k(cls, layer_config, config, shape_m: int) -> bool:
        return config.get("use_stream_k", False)

    @classmethod
    def _should_rebalance_stream_k(cls, layer_config, config, shape_m: int) -> bool:
        block_m, block_n, block_k = config["block_shape"]
        num_blocks_m = math.ceil(shape_m / block_m)
        num_blocks_n = math.ceil(layer_config.shape_n / block_n)
        num_k_tiles = layer_config.shape_k // block_k
        num_ctas = cls.get_num_sms() * config.get("num_ctas_per_sm", 1)
        return num_blocks_m * num_blocks_n * num_k_tiles < num_ctas * config["num_stages"] * 2

    @classmethod
    def _rebalance_dense_warps(cls, layer_config, config, shape_m: int) -> None:
        use_stream_k = config.get("use_stream_k", False)
        block_m, block_n, _ = config["block_shape"]
        _, warp_n, _ = config["warp_shape"]
        mma_k = 1024 // layer_config.a_dtype.num_bits
        block_k = min(layer_config.shape_k, mma_k * 2)
        if layer_config.shape_k % block_k:
            return

        num_warps_n = block_n // warp_n
        num_warps_m = min(2, block_m // 16, max(1, 4 // num_warps_n))
        if num_warps_m * num_warps_n < 4:
            return

        warp_m = block_m // num_warps_m
        if warp_m % 16:
            return

        num_k_tiles = layer_config.shape_k // block_k
        avoid_small_warp_m = cls.sm_version == 120 and shape_m <= block_m and warp_m < 32 and num_k_tiles > 4
        if not use_stream_k and avoid_small_warp_m:
            return

        candidate = config | {
            "block_shape": (block_m, block_n, block_k),
            "warp_shape": (warp_m, warp_n, block_k),
        }
        if all(candidate[key] == config[key] for key in ("block_shape", "warp_shape")):
            return

        if warp_m < 64:
            candidate["use_tma"] = False
            candidate["use_warp_spec"] = False
            for key in [key for key in candidate if key.startswith("use_tma_")]:
                candidate.pop(key)

        max_stages = min(4, candidate["num_stages"])
        for num_stages in range(max_stages, 1, -1):
            smem = estimate_smem_size_layer(
                layer_config,
                candidate["block_shape"],
                GemmType.DENSE,
                num_stages,
                warp_shape=candidate["warp_shape"],
                reduce_overlap_last_stage_only=candidate.get("reduce_overlap_last_stage_only", False),
                use_mbarrier=True,
                use_warp_spec=candidate.get("use_warp_spec", False),
                num_write_splits=candidate.get("num_write_splits", 1),
            )
            if smem <= cls.max_smem_size:
                candidate["num_stages"] = num_stages
                config.clear()
                config.update(candidate)
                return

    @staticmethod
    def _use_tma_b_only(config) -> None:
        config.update(
            use_tma=True,
            use_warp_spec=False,
            use_tma_a=False,
            use_tma_as=False,
            use_tma_b=True,
            use_tma_c=False,
            use_tma_bs=False,
            use_tma_bs2=False,
            use_tma_bzp=False,
            use_tma_bias=False,
        )

    @classmethod
    def _use_m_tile(cls, layer_config, config, gemm_type, block_m: int) -> None:
        _, block_n, block_k = config["block_shape"]
        _, warp_n, warp_k = config["warp_shape"]
        config["block_shape"] = (block_m, block_n, block_k)
        config["warp_shape"] = (block_m, warp_n, warp_k)
        config["num_stages"] = cls._fit_num_stages(layer_config, config, gemm_type, reduce_overlap=False)

    @classmethod
    def _fit_num_stages(cls, layer_config, config, gemm_type, reduce_overlap: bool) -> int:
        best = 2
        for num_stages in range(3, 6 if cls.sm_version == 121 else 5):
            smem = estimate_smem_size_layer(
                layer_config,
                config["block_shape"],
                gemm_type,
                num_stages,
                warp_shape=config["warp_shape"],
                reduce_overlap_last_stage_only=reduce_overlap,
                use_mbarrier=True,
                use_warp_spec=config["use_warp_spec"],
                num_write_splits=config.get("num_write_splits", 1),
            )
            if smem <= cls.max_smem_size:
                best = num_stages
        return best
