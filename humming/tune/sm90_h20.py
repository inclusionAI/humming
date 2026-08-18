import math

import numpy as np

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.tune.base import DeviceHeuristics
from humming.tune.candidate import TuningProblem
from humming.tune.sm90_h20_families import (
    fused_e8m0_dense_in_scope,
    fused_e8m0_moe_in_scope,
    make_h20_device_profile,
    select_fused_e8m0_dense,
    select_fused_e8m0_moe,
)
from humming.utils.smem import estimate_smem_size_layer


class Sm90H20Heuristics(DeviceHeuristics):
    max_smem_size: int = 227 * 1024
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 90

    @classmethod
    def _get_small_m_dense_override(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        block_shape_m: int,
    ) -> dict | None:
        """Pick a small-M dense tile from resource and parallelism estimates."""
        a_bits = layer_config.a_dtype.num_bits
        b_bits = layer_config.b_dtype.num_bits
        shape_n, shape_k = layer_config.shape_n, layer_config.shape_k
        if shape_m > block_shape_m or a_bits not in (8, 16) or shape_n % 128:
            return None
        if a_bits == 8 and (not layer_config.b_dtype.is_integer_type or b_bits > 4):
            return None

        num_sms = cls.get_num_sms()
        warp_k = 512 // a_bits
        reference_k = 2 * warp_k
        reference_tiles = math.ceil(shape_m / block_shape_m) * (shape_n // 128) * (shape_k // reference_k)
        tiles_per_sm = reference_tiles / num_sms

        def make_config(block_n, block_k, warp_n, warp_k, num_stages, use_tma, overlap=False):
            config = {
                "block_shape": (block_shape_m, block_n, block_k),
                "warp_shape": (block_shape_m, warp_n, warp_k),
                "use_stream_k": True,
                "num_sms": num_sms,
                "num_stages": num_stages,
                "num_ctas_per_sm": 1,
                "use_tma": use_tma,
                "use_warp_spec": use_tma,
                "use_mbarrier": use_tma,
            }
            if overlap:
                config["reduce_overlap_last_stage_only"] = True
            return config

        max_output_values = 6 * 1024
        wide_k = 2 * reference_k if a_bits == 16 or b_bits >= 4 else reference_k
        wide_num_iters = math.ceil(shape_m / block_shape_m) * (shape_n // 256) * (shape_k // wide_k)
        stage4_slice = max(4, math.ceil(wide_num_iters / num_sms / 4) * 4)
        stage4_active_ctas = min(num_sms, math.ceil(wide_num_iters / stage4_slice))
        if (
            block_shape_m * 256 <= max_output_values
            and shape_n % 256 == 0
            and shape_k % wide_k == 0
            and shape_k // wide_k >= 4 * 4
            and stage4_active_ctas * 2 >= num_sms
        ):
            block_shape = (block_shape_m, 256, wide_k)
            stage_scores = []
            for num_stages in range(4, 7):
                smem_size = estimate_smem_size_layer(layer_config, block_shape, GemmType.DENSE, num_stages)
                if smem_size > cls.max_smem_size * 0.8:
                    continue
                slice_iters = math.ceil(wide_num_iters / num_sms / num_stages) * num_stages
                active_ctas = min(num_sms, math.ceil(wide_num_iters / slice_iters))
                pipeline_gain = 0.05 * min(b_bits, 4) / 4
                stage_scores.append((active_ctas * (1 + pipeline_gain * num_stages), num_stages))
            if stage_scores:
                return make_config(256, wide_k, 64, warp_k, max(stage_scores)[1], True, overlap=True)

        if a_bits != 16 or block_shape_m * 128 > max_output_values:
            return None
        k_pipeline_turns = shape_k // warp_k
        if k_pipeline_turns <= 32:
            return None
        if k_pipeline_turns <= 64 and tiles_per_sm <= 6 and shape_k % warp_k == 0:
            return make_config(128, warp_k, 32, warp_k, 4, False)
        if shape_k % reference_k == 0 and shape_n // 128 <= math.ceil(num_sms / 8):
            return make_config(128, reference_k, 32, reference_k, 4, True)
        return None

    @classmethod
    def _fit_dense_block_m_to_output_grid(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        block_m: int,
        block_n: int,
        block_k: int,
    ) -> int:
        """Expose enough dense output tiles without sacrificing M reuse."""
        num_n_tiles = layer_config.shape_n // block_n
        current_m_tiles = math.ceil(shape_m / block_m)

        # Stream-K slices provide less useful parallelism than distinct output tiles.
        stream_k_grid_gain = 1.0
        current_output_tiles = current_m_tiles * num_n_tiles
        if layer_config.shape_k <= 1024:
            if current_output_tiles >= math.ceil(cls.get_num_sms() * 0.5):
                return block_m
        else:
            if current_output_tiles >= math.ceil(cls.get_num_sms() * 0.2):
                return block_m
            stream_k_grid_gain = min(4.5, layer_config.shape_k / (12 * block_k))
        target_wave_fraction = 0.8
        if layer_config.shape_k > 1024:
            target_wave_fraction = max(0.5, 1 - layer_config.shape_k / (8 * 1024))
        target_output_tiles = math.ceil(cls.get_num_sms() * target_wave_fraction / stream_k_grid_gain)
        if current_output_tiles >= target_output_tiles:
            return block_m

        current_padded_rows = current_m_tiles * block_m
        candidates = []
        padding_safe_candidates = []
        min_stream_k_block_m = 16 if layer_config.shape_k > 1024 and block_m >= 32 else 8
        for candidate_m in range(8, block_m, 8):
            if layer_config.a_dtype == dtypes.int8 and candidate_m > 32 and candidate_m % 16:
                continue
            candidate_m_tiles = math.ceil(shape_m / candidate_m)
            padded_rows = candidate_m_tiles * candidate_m
            if padded_rows > current_padded_rows * 1.05:
                continue
            padding_safe_candidates.append(candidate_m)
            if candidate_m_tiles * num_n_tiles < target_output_tiles:
                continue
            if candidate_m < min_stream_k_block_m:
                continue
            candidates.append(candidate_m)
        if candidates:
            return max(candidates)
        if layer_config.shape_k <= 1024 and padding_safe_candidates:
            return min(padding_safe_candidates)
        if layer_config.shape_k > 1024 and block_m >= 32:
            reuse_candidates = [candidate_m for candidate_m in padding_safe_candidates if candidate_m >= 16]
            if reuse_candidates:
                return max(reuse_candidates)
        return block_m

    @classmethod
    def _tune_long_k_moe_residency(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        gemm_type: GemmType,
        config: dict,
    ) -> None:
        """Trade excess pipeline storage for more resident MoE CTAs."""
        block_m, block_n, block_k = config["block_shape"]
        if layer_config.shape_k <= 1024 or block_m > 32:
            return

        num_stages = 4
        warp_shape = config["warp_shape"]
        smem_size = estimate_smem_size_layer(
            layer_config,
            config["block_shape"],
            gemm_type,
            num_stages,
            warp_shape=warp_shape,
            mma_accum_bits=16 if config["use_f16_accum"] else 32,
        )
        num_threads = math.prod(config["block_shape"]) // math.prod(warp_shape) * 32
        # Cap residency at three CTAs to preserve register headroom.
        num_experts = layer_config.num_experts
        if shape_m < num_experts:
            estimated_m_blocks = shape_m
        else:
            blocks_per_expert = math.ceil(shape_m / num_experts / block_m)
            estimated_m_blocks = num_experts * blocks_per_expert
        num_sms_physical = cls.get_num_sms()

        if layer_config.shape_n >= 1024 and layer_config.shape_n % 512 == 0:
            wide_block_n = 512
        elif layer_config.shape_n >= 512 and layer_config.shape_n % 256 == 0:
            wide_block_n = 256
        else:
            wide_block_n = 0
        wide_block_k = 64
        wide_warp_n = 64
        wide_num_stages = 3
        wide_output_tiles = 0
        if wide_block_n:
            wide_output_tiles = estimated_m_blocks * (layer_config.shape_n // wide_block_n)
        expert_tile_fill = shape_m / (estimated_m_blocks * block_m)
        has_wide_grid = wide_output_tiles >= 2 * num_sms_physical
        underfilled_expert_tiles = expert_tile_fill < 0.5 or (
            expert_tile_fill <= 0.5 and (has_wide_grid or layer_config.b_dtype.num_bits < 4)
        )
        wide_k_tiles = layer_config.shape_k // wide_block_k
        stream_k_grid_gain = min(16, max(4, wide_k_tiles // (2 * wide_num_stages)))
        stream_k_can_fill_grid = (
            layer_config.b_dtype.num_bits <= 4
            and wide_output_tiles * stream_k_grid_gain >= 3 * num_sms_physical
            and wide_k_tiles >= 64
        )
        has_wide_tile = wide_block_n > 0 and block_m * wide_block_n <= 8 * 1024
        wide_n_aligned = wide_block_n > 0 and layer_config.shape_n % wide_block_n == 0
        wide_k_aligned = layer_config.shape_k % wide_block_k == 0
        has_wide_parallelism = has_wide_grid or stream_k_can_fill_grid
        is_wide_tile_legal = has_wide_tile and wide_n_aligned and wide_k_aligned
        use_wide_moe_tile = is_wide_tile_legal and underfilled_expert_tiles and has_wide_parallelism
        if use_wide_moe_tile:
            wide_block_shape = (block_m, wide_block_n, wide_block_k)
            wide_warp_shape = (block_m, wide_warp_n, wide_block_k)
            wide_smem_size = estimate_smem_size_layer(
                layer_config,
                wide_block_shape,
                gemm_type,
                wide_num_stages,
                warp_shape=wide_warp_shape,
                mma_accum_bits=16 if config["use_f16_accum"] else 32,
            )
            wide_num_threads = math.prod(wide_block_shape) // math.prod(wide_warp_shape) * 32
            wide_num_ctas = min(3, cls.max_smem_size // wide_smem_size, 1024 // wide_num_threads)
            if wide_num_ctas >= 1:
                config.update(
                    block_shape=wide_block_shape,
                    warp_shape=wide_warp_shape,
                    num_stages=wide_num_stages,
                    num_ctas_per_sm=wide_num_ctas,
                    num_sms=num_sms_physical,
                )
                return

        num_output_tiles = estimated_m_blocks * (layer_config.shape_n // block_n)
        num_ctas_per_sm = min(3, cls.max_smem_size // smem_size, 1024 // num_threads)
        if num_output_tiles < num_sms_physical:
            num_ctas_per_sm = min(num_ctas_per_sm, 2)
        if num_ctas_per_sm < 1:
            return

        k_tiles = layer_config.shape_k // block_k
        useful_ctas = num_output_tiles * math.ceil(k_tiles / num_stages)
        num_sms = min(num_sms_physical, math.ceil(useful_ctas / num_ctas_per_sm))
        config.update(num_stages=num_stages, num_ctas_per_sm=num_ctas_per_sm, num_sms=max(1, num_sms))

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
        is_moe = gemm_type != GemmType.DENSE
        if a_dtype.num_bits == 16:
            return {
                "block_shape": (64, 256, 512 // a_dtype.num_bits),
                "warp_shape": (64, 64, 512 // a_dtype.num_bits),
                "num_ctas_per_sm": 2,
            }
        elif use_fused_e8m0_scale and not is_moe:
            return {
                "block_shape": (128, 128, 1024 // a_dtype.num_bits),
                "warp_shape": (128, 32, 1024 // a_dtype.num_bits),
                "num_ctas_per_sm": 2,
            }
        elif use_fused_e8m0_scale and is_moe:
            return {
                "block_shape": (64, 128, 1024 // a_dtype.num_bits),
                "warp_shape": (64, 32, 1024 // a_dtype.num_bits),
                "num_ctas_per_sm": 3,
            }
        elif group_size == 0 and not is_moe:
            return {
                "block_shape": (64, 256, 512 // a_dtype.num_bits),
                "warp_shape": (64, 64, 512 // a_dtype.num_bits),
                "num_ctas_per_sm": 2,
            }
        elif group_size == 0 and is_moe:
            return {
                "block_shape": (64, 128, 512 // a_dtype.num_bits),
                "warp_shape": (64, 32, 512 // a_dtype.num_bits),
                "num_ctas_per_sm": 3,
            }
        elif group_size >= 128 and shape_k > 512:
            return {
                "block_shape": (64, 128, 1024 // a_dtype.num_bits),
                "warp_shape": (64, 16, 1024 // a_dtype.num_bits),
                "num_ctas_per_sm": 2,
            }
        else:
            return {
                "block_shape": (64, 128, 512 // a_dtype.num_bits),
                "warp_shape": (64, 32, 512 // a_dtype.num_bits),
                "num_ctas_per_sm": 3 if is_moe else 2,
            }

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        moe_in_scope = fused_e8m0_moe_in_scope(
            layer_config, shape_m, gemm_type, use_batch_invariant
        )
        dense_in_scope = fused_e8m0_dense_in_scope(
            layer_config, shape_m, gemm_type, use_batch_invariant
        )
        if moe_in_scope or dense_in_scope:
            problem = TuningProblem(
                layer_config=layer_config,
                shape_m=shape_m,
                gemm_type=gemm_type,
                device=make_h20_device_profile(cls.get_num_sms()),
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
            )
            if moe_in_scope:
                return select_fused_e8m0_moe(problem).to_config()
            return select_fused_e8m0_dense(problem).to_config()

        return cls._get_config_legacy(
            layer_config,
            shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )

    @classmethod
    def _get_config_legacy(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        group_size = layer_config.input_scale_group_size or layer_config.weight_scale_group_size
        is_moe = gemm_type != GemmType.DENSE
        a_dtype = layer_config.a_dtype
        config = cls.get_base_config(
            a_dtype,
            layer_config.b_dtype,
            group_size,
            use_f16_accum,
            layer_config.use_fused_e8m0_scale,
            gemm_type,
            layer_config.shape_k,
        )
        block_shape_m, block_shape_n, block_shape_k = config["block_shape"]
        num_ctas_per_sm = config.get("num_ctas_per_sm", 1)
        warp_shape_m, warp_shape_n, warp_shape_k = config["warp_shape"]
        if layer_config.use_packed_k_layout:
            warp_shape_n = max(warp_shape_n, 32)
        num_stages = 3
        min_warp_shape_n = 32 if a_dtype.num_bits == 16 or layer_config.use_packed_k_layout else 16
        while layer_config.shape_n % block_shape_n:
            block_shape_n //= 2
            warp_shape_n = min(warp_shape_n, block_shape_n // 4)
        assert warp_shape_n >= min_warp_shape_n

        if not layer_config.num_experts:
            if shape_m <= block_shape_m:
                block_shape_m = math.ceil(shape_m / 8) * 8
            else:
                blocks = [math.ceil(shape_m / ((i + 1) * 8)) for i in range(block_shape_m // 8)]
                block_shape_m = np.argmin(blocks).item() * 8 + 8
            if layer_config.a_dtype == dtypes.int8 and block_shape_m > 32 and block_shape_m % 16 != 0:
                block_shape_m = math.ceil(block_shape_m / 16) * 16
        else:
            block_size_configs = [(8, 0.7), (16, 0.8), (32, 0.9), (48, 0.9), (64, 0.9)]
            for moe_block_size, threshold in block_size_configs:
                if shape_m / layer_config.num_experts / moe_block_size < threshold:
                    break

            new_shape_m = int(shape_m / layer_config.num_experts / 0.9)
            new_shape_m = max(new_shape_m, 1)
            if block_shape_m == 128:
                if np.ceil(new_shape_m / 96) * 96 < np.ceil(new_shape_m / 64) * 64:
                    block_shape_m = 96
                elif np.ceil(new_shape_m / 128) * 128 < np.ceil(new_shape_m / 64) * 64 * 1.05:
                    block_shape_m = 128
                else:
                    block_shape_m = moe_block_size
            elif new_shape_m >= 64 and new_shape_m < 96:
                block_shape_m = 48
            else:
                block_shape_m = moe_block_size

        warp_shape_m = block_shape_m
        num_blocks_n = layer_config.shape_n // block_shape_n
        num_blocks_m = cls.estimate_num_blocks_m(layer_config, shape_m, block_shape_m)

        num_sms = cls.get_num_sms()
        while num_blocks_n * num_blocks_m * 2 < num_sms * num_ctas_per_sm:
            if warp_shape_n == 64:
                warp_shape_n = warp_shape_n // 2
                block_shape_n = block_shape_n // 2
                num_blocks_n = num_blocks_n * 2
                if num_ctas_per_sm == 2:
                    num_ctas_per_sm = 3
                continue
            elif num_ctas_per_sm > 1:
                num_ctas_per_sm = num_ctas_per_sm - 1
                continue
            else:
                break

        num_warps_m = block_shape_m // warp_shape_m
        num_warps_n = block_shape_n // warp_shape_n
        num_warps_k = block_shape_k // warp_shape_k
        num_warps = num_warps_m * num_warps_n * num_warps_k * num_ctas_per_sm

        if num_warps == 4:
            warp_shape_k = 512 // layer_config.a_dtype.num_bits
            block_shape_k = warp_shape_k * 2

        if num_warps <= 8 and block_shape_m <= 32:
            if is_moe and warp_shape_n == 64:
                warp_shape_n = warp_shape_n // 2
            else:
                num_warps_k = block_shape_k // warp_shape_k
                warp_shape_k = 512 // layer_config.a_dtype.num_bits
                block_shape_k = warp_shape_k * num_warps_k * 2

        if is_moe and layer_config.shape_k <= 512 and layer_config.shape_n >= 2048 and block_shape_m <= 32:
            if block_shape_n == 256:
                warp_shape_n = 32
                block_shape_n = 128
                num_blocks_n = num_blocks_n * 2

            if num_blocks_n * num_blocks_m >= num_sms * 4:
                num_ctas_per_sm = 4

        if warp_shape_k == block_shape_k and warp_shape_k == 512 // layer_config.a_dtype.num_bits:
            block_shape = (block_shape_m, block_shape_n, block_shape_k * 2)
            smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, num_stages)
            if smem_size * num_ctas_per_sm < cls.max_smem_size:
                block_shape_k = block_shape_k * 2
                warp_shape_k = warp_shape_k * 2

        max_num_stages = 4
        for num_stages_new in range(num_stages + 1, max_num_stages + 1):
            block_shape = (block_shape_m, block_shape_n, block_shape_k)
            smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, num_stages_new)
            if smem_size * num_ctas_per_sm < cls.max_smem_size:
                num_stages = num_stages_new

        if not is_moe:
            block_shape_m = cls._fit_dense_block_m_to_output_grid(
                layer_config,
                shape_m,
                block_shape_m,
                block_shape_n,
                block_shape_k,
            )
            warp_shape_m = block_shape_m
            num_blocks_m = math.ceil(shape_m / block_shape_m)

        if num_ctas_per_sm == 1:
            factor = min(4.5, layer_config.shape_k / (3 * block_shape_k))
            if layer_config.shape_k > 1024:
                # Keep at least two stage-4 turns per Stream-K slice.
                factor = min(9, max(factor, layer_config.shape_k / (8 * block_shape_k)))
            num_sms = min(num_sms, math.ceil(num_blocks_n * num_blocks_m * factor))

        while layer_config.shape_k % block_shape_k != 0:
            warp_shape_k = 512 // layer_config.a_dtype.num_bits
            block_shape_k = block_shape_k // 2
            assert block_shape_k >= warp_shape_k

        if (
            layer_config.a_dtype.num_bits == 8
            and layer_config.input_scale_group_size > 0
            and gemm_type != GemmType.GROUPED_MASKED
            and shape_m >= 6144
        ):
            num_ctas_per_sm = min(num_ctas_per_sm, 2)

        config = {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (warp_shape_m, warp_shape_n, warp_shape_k),
            "use_stream_k": layer_config.shape_k > 1024,
            "use_f16_accum": use_f16_accum,
            "num_sms": num_sms,
            "num_stages": num_stages,
            "num_ctas_per_sm": num_ctas_per_sm,
        }

        if layer_config.shape_k <= 512 and is_moe and shape_m >= 2048:
            config["use_tma"] = True
            config["use_mbarrier"] = True
            if gemm_type == GemmType.INDEXED:
                config["use_tma_a"] = False
                config["use_tma_c"] = False

            # num_sms is a launch grid factor here and may exceed the physical SM count.
            if config["num_ctas_per_sm"] > 1 and shape_m >= 24576:
                tiles_per_cta = 5
                block_m, block_n, _ = config["block_shape"]
                num_tiles = (layer_config.shape_n // block_n) * (shape_m // block_m)
                sms_target = num_tiles / (config["num_ctas_per_sm"] * tiles_per_cta)
                config["num_sms"] = max(config["num_sms"], 1 << round(math.log2(sms_target)))

        has_tma_tile = block_shape_m >= 48
        has_tma_resources = num_ctas_per_sm <= 2 and num_warps <= 8
        has_tma_pipeline = layer_config.shape_k // block_shape_k >= 24
        use_dense_tma = not is_moe and has_tma_tile and has_tma_resources and has_tma_pipeline
        if use_dense_tma:
            config["use_tma"] = True
            config["use_warp_spec"] = True
            config["use_mbarrier"] = True
            config["num_stages"] = 3
        elif config["num_stages"] == 4 and block_shape_m <= 32:
            block_shape = (block_shape_m, block_shape_n, block_shape_k)
            smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, 5)
            if smem_size * num_ctas_per_sm < cls.max_smem_size and not config["use_stream_k"]:
                config["num_stages"] = 5

        if not is_moe and not use_batch_invariant:
            config.update(cls._get_small_m_dense_override(layer_config, shape_m, block_shape_m) or {})
        elif is_moe and not use_batch_invariant:
            cls._tune_long_k_moe_residency(layer_config, shape_m, gemm_type, config)

        if use_batch_invariant:
            warp_shape_k = 512 // layer_config.a_dtype.num_bits
            block_shape_k = 512 // layer_config.a_dtype.num_bits
            config["block_shape"] = (block_shape_m, block_shape_n, block_shape_k)
            config["warp_shape"] = (warp_shape_m, warp_shape_n, warp_shape_k)
            # TODO: check if TMA / cp.async affect batch invariance
            config["use_tma"] = False
            config["use_warp_spec"] = False
            config["use_mbarrier"] = False
            config["use_stream_k"] = False

        return config
