import math

import numpy as np

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.tune.base import DeviceHeuristics
from humming.utils.smem import estimate_smem_size_layer


class Sm90Heuristics(DeviceHeuristics):
    max_smem_size: int = 227 * 1024
    # Keep two-CTA configs at 256 threads; 512 would force a 64-register limit.
    max_indexed_threads_for_two_ctas: int = 256
    max_indexed_threads_for_three_ctas: int = 256
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [
        dtypes.int8,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    ]
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 90

    @classmethod
    def _estimate_indexed_ctas_per_sm(
        cls,
        layer_config: LayerConfig,
        block_shape: tuple[int, int, int],
        warp_shape: tuple[int, int, int],
        num_stages: int,
        num_output_tiles: int,
        num_sms: int,
    ) -> int:
        num_threads = math.prod(block_shape) // math.prod(warp_shape) * 32
        smem_size = estimate_smem_size_layer(
            layer_config,
            block_shape,
            GemmType.INDEXED,
            num_stages,
            warp_shape=warp_shape,
        )
        resource_limit = 1
        if num_threads <= cls.max_indexed_threads_for_two_ctas and smem_size * 2 <= cls.max_smem_size:
            resource_limit = 2
        # The measured A16/B4 M8 kernels tolerate the 85-register 3-CTA bound.
        if (
            layer_config.a_dtype.num_bits == 16
            and layer_config.b_dtype.num_bits == 4
            and block_shape[0] == 8
            and num_threads <= cls.max_indexed_threads_for_three_ctas
            and smem_size * 3 <= cls.max_smem_size
        ):
            resource_limit = 3
        grid_limit = math.ceil(num_output_tiles / num_sms)
        return max(1, min(resource_limit, grid_limit))

    @classmethod
    def _fit_indexed_a16_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        config: dict,
    ) -> dict:
        block_shape = tuple(config["block_shape"])
        warp_shape = tuple(config["warp_shape"])
        num_stages = config["num_stages"]
        num_sms = cls.get_num_sms()
        num_output_tiles = (
            layer_config.shape_n
            // block_shape[1]
            * cls.estimate_num_blocks_m(layer_config, shape_m, block_shape[0])
        )
        num_ctas_per_sm = cls._estimate_indexed_ctas_per_sm(
            layer_config,
            block_shape,
            warp_shape,
            num_stages,
            num_output_tiles,
            num_sms,
        )

        # A smaller K tile is useful only when it admits another resident CTA.
        smaller_block_k = block_shape[2] // 2
        scale_group_sizes = (
            layer_config.input_scale_group_size,
            layer_config.weight_scale_group_size,
        )
        # K tiles must nest cleanly with every active scale group.
        scale_groups_align = all(
            not group_size or group_size % smaller_block_k == 0 or smaller_block_k % group_size == 0
            for group_size in scale_group_sizes
        )
        if num_ctas_per_sm == 1 and block_shape[2] >= warp_shape[2] * 2 and scale_groups_align:
            smaller_block_shape = (*block_shape[:2], smaller_block_k)
            smaller_warp_shape = (
                *warp_shape[:2],
                min(warp_shape[2], smaller_block_k),
            )
            smaller_num_ctas = cls._estimate_indexed_ctas_per_sm(
                layer_config,
                smaller_block_shape,
                smaller_warp_shape,
                num_stages,
                num_output_tiles,
                num_sms,
            )
            if smaller_num_ctas > num_ctas_per_sm:
                block_shape = smaller_block_shape
                warp_shape = smaller_warp_shape
                num_ctas_per_sm = smaller_num_ctas

        if (
            layer_config.a_dtype.num_bits == 16
            and layer_config.b_dtype.num_bits == 4
            and block_shape[1] >= 256
            and block_shape[2] == 64
        ):
            candidate_block_shape = (
                block_shape[0],
                block_shape[1] // 2,
                block_shape[2] * 2,
            )
            candidate_warp_shape = warp_shape
            candidate_legal = (
                layer_config.shape_n % candidate_block_shape[1] == 0
                and layer_config.shape_k % candidate_block_shape[2] == 0
            )
            if candidate_legal:
                candidate_tiles = num_output_tiles * 2
                candidate_ctas = cls._estimate_indexed_ctas_per_sm(
                    layer_config,
                    candidate_block_shape,
                    candidate_warp_shape,
                    num_stages,
                    candidate_tiles,
                    num_sms,
                )
                current_waves = math.ceil(num_output_tiles / (num_sms * num_ctas_per_sm))
                candidate_waves = math.ceil(candidate_tiles / (num_sms * candidate_ctas))
                # Trade a wider K tile for more N tiles only within the same grid wave.
                if candidate_ctas > num_ctas_per_sm and candidate_waves <= current_waves:
                    block_shape = candidate_block_shape
                    warp_shape = candidate_warp_shape
                    num_ctas_per_sm = candidate_ctas
                    num_output_tiles = candidate_tiles

        # Avoid Stream-K locks once direct output tiles fill an SM wave.
        return config | {
            "block_shape": block_shape,
            "warp_shape": warp_shape,
            "num_ctas_per_sm": num_ctas_per_sm,
            "use_stream_k": config["use_stream_k"] and num_output_tiles < num_sms,
        }

    @classmethod
    def get_config1(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        tune_indexed_a16 = (
            gemm_type == GemmType.INDEXED and layer_config.a_dtype.num_bits == 16 and not use_batch_invariant
        )
        if layer_config.use_packed_k_layout:
            max_block_m = 128
        elif use_f16_accum:
            max_block_m = 256
        else:
            max_block_m = 176

        if tune_indexed_a16:
            # Bound padding when only a few routed rows land on each expert.
            tokens_per_expert = shape_m / layer_config.num_experts
            first_threshold = 1.01 if layer_config.b_dtype.num_bits == 4 else 0.7
            moe_block_size_configs = (
                (8, first_threshold),
                (16, 0.7),
                (24, 0.8),
                (32, 0.9),
                (48, 0.9),
                (64, 0.9),
            )
            for block_shape_m, threshold in moe_block_size_configs:
                if tokens_per_expert / block_shape_m < threshold:
                    break
        else:
            num_blocks_list = cls.calc_num_block_list(
                layer_config,
                shape_m,
                max_block_m,
            )
            block_shape_m = np.argmin(num_blocks_list).item() * 8 + 8
        warp_shape_n = 32
        warp_shape_k = 1024 // layer_config.a_dtype.num_bits

        # Long-K layers need more routed rows before wider N tiles pay off.
        wide_tile_min_shape_m = 64 if layer_config.shape_k > 4096 else 16
        use_wide_indexed_tile = tune_indexed_a16 and block_shape_m <= 64 and shape_m >= wide_tile_min_shape_m
        if use_wide_indexed_tile:
            warp_shape_n = 64
            if layer_config.shape_k <= 512 and layer_config.shape_n >= 2048:
                # Shallow K needs more N work per CTA to amortize scheduling.
                block_shape_n = 512
                block_shape_k = 64
            else:
                block_shape_n = 256
                block_shape_k = 128
        elif layer_config.shape_n <= 4096 and not use_batch_invariant and block_shape_m <= 64:
            block_shape_n = 128
            block_shape_k = warp_shape_k * 2
            if block_shape_m <= 32:
                block_shape_k = block_shape_k * 2
            if block_shape_k > 256:
                block_shape_k = block_shape_k // 2
                warp_shape_k = warp_shape_k // 2

            while layer_config.shape_k % block_shape_k != 0:
                block_shape_k = block_shape_k // 2
        else:
            block_shape_n = 256
            block_shape_k = warp_shape_k
            if block_shape_m <= 32 and layer_config.b_dtype.num_bits <= 6:
                block_shape_k = block_shape_k * 2
            elif block_shape_m <= 32:
                warp_shape_k = warp_shape_k // 2

        min_warp_shape_n = 32 if layer_config.a_dtype.num_bits == 16 else 16
        # Keep a complete four-warp WGMMA group while fitting output width.
        while layer_config.shape_n % block_shape_n != 0:
            block_shape_n //= 2
            assert block_shape_n >= min_warp_shape_n * 4
        warp_shape_n = min(warp_shape_n, block_shape_n // 4)

        # Earlier shape fitting can reduce block K below the initial warp K.
        warp_shape_k = min(warp_shape_k, block_shape_k)
        while layer_config.shape_k % block_shape_k != 0:
            block_shape_k = block_shape_k // 2
            warp_shape_k = min(warp_shape_k, block_shape_k)
            assert block_shape_k >= warp_shape_k

        dense_small_fp4 = (
            gemm_type == GemmType.DENSE
            and layer_config.a_dtype.num_bits == 16
            and layer_config.b_dtype.num_bits == 4
            and shape_m <= 128
            and layer_config.shape_n % 128 == 0
            and layer_config.shape_k % 64 == 0
        )
        if dense_small_fp4:
            block_shape_n = 128
            block_shape_k = 64
            warp_shape_n = 32
            warp_shape_k = 64
        config = {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (block_shape_m, warp_shape_n, warp_shape_k),
            "use_stream_k": not use_batch_invariant,
            "use_f16_accum": use_f16_accum,
            "num_stages": 4,
        }

        if gemm_type != GemmType.INDEXED:
            config["use_warp_spec"] = True
            config["use_tma"] = True
            config["use_mbarrier"] = True
            if dense_small_fp4:
                config["num_ctas_per_sm"] = 2

            if layer_config.shape_n % (block_shape_n * 2) == 0 and shape_m / block_shape_m >= 4:
                if gemm_type == GemmType.DENSE:
                    config["multi_cast_size_a"] = 2

        return config

    @classmethod
    def get_config2(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if use_f16_accum:
            max_block_m = 256
        elif layer_config.input_scale_group_size > 0:
            max_block_m = 160
        elif layer_config.weight_scale_group_size < 128:
            max_block_m = 192
        else:
            max_block_m = 200

        num_blocks_list = cls.calc_num_block_list(layer_config, shape_m, max_block_m)
        block_shape_m = np.argmin(num_blocks_list).item() * 8 + 8

        block_shape_k = 256 if block_shape_m <= 32 else 128
        if layer_config.shape_k % 256 != 0:
            block_shape_k = 128

        # N32 avoids the 512-thread grouped-scale CTA when an N128 tile fits.
        block_shape_n = 128
        warp_shape_n = 32
        while layer_config.shape_n % block_shape_n != 0:
            block_shape_n //= 2
            warp_shape_n = min(warp_shape_n, block_shape_n // 4)
            assert warp_shape_n >= 16

        config = {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (block_shape_m, warp_shape_n, 128),
            "use_stream_k": not use_batch_invariant,
            "use_f16_accum": use_f16_accum,
            "num_stages": 4,
        }

        if gemm_type != GemmType.INDEXED:
            config["use_warp_spec"] = True
            config["use_tma"] = True
            config["use_mbarrier"] = True

            if shape_m / block_shape_m >= 4 and gemm_type == GemmType.DENSE:
                config["multi_cast_size_a"] = 2

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
            for i in range(max_block_m // 8):
                block_m = i * 8 + 8
                num_blocks_list.append(math.ceil(shape_m / block_m))
        else:
            random_state = np.random.RandomState(seed=0)
            samples = random_state.randint(0, layer_config.num_experts, size=shape_m)
            counts = np.bincount(samples)
            for i in range(max_block_m // 8):
                block_m = i * 8 + 8
                num_blocks = int(np.ceil(counts * 1.1 / block_m).sum().item())
                num_blocks_list.append(num_blocks)

        for i in range(max_block_m // 8):
            num_blocks = num_blocks_list[i]
            block_m = i * 8 + 8
            if layer_config.a_dtype == dtypes.int8 and block_m % 16 == 8 and block_m > 32:
                num_blocks_list[i] = 1000000

        return num_blocks_list

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if layer_config.a_dtype.num_bits == 16:
            func = cls.get_config1
        elif layer_config.use_packed_k_layout:
            func = cls.get_config1
        elif layer_config.input_scale_group_size == 0 and layer_config.weight_scale_group_size == 0:
            func = cls.get_config1
        elif layer_config.use_fused_e8m0_scale and layer_config.input_scale_group_size == 0:
            func = cls.get_config1
        else:
            func = cls.get_config2

        config = func(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        if gemm_type == GemmType.INDEXED and layer_config.a_dtype.num_bits == 16 and not use_batch_invariant:
            config = cls._fit_indexed_a16_config(
                layer_config,
                shape_m,
                config,
            )

        while config["num_stages"] > 3:
            smem_size = estimate_smem_size_layer(
                layer_config,
                config["block_shape"],
                gemm_type,
                config["num_stages"],
                warp_shape=config["warp_shape"],
                reduce_overlap_last_stage_only=config.get("reduce_overlap_last_stage_only", False),
                use_mbarrier=config.get("use_mbarrier", False),
                use_warp_spec=config.get("use_warp_spec", False),
                num_write_splits=config.get("num_write_splits", 1),
                mma_accum_bits=16 if use_f16_accum else 32,
            )
            if smem_size <= cls.max_smem_size:
                break
            config["num_stages"] -= 1

        return config
