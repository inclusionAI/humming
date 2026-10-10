import functools
import math

import numpy as np

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import current_device
from humming.utils.math import round_up
from humming.utils.smem import estimate_smem_size_layer


def _estimate_compute_bound_threshold(layer_config: LayerConfig, use_f16_accum: bool) -> float:
    info = current_device
    dtype = str(layer_config.a_dtype)
    if "float16" not in info.tensorcore_tops:
        raise RuntimeError(f"unknown FP16 Tensor Core throughput for sm{info.sm_version}")

    max_tops = info.tensorcore_tops[dtype]
    max_bandwidth = info.memory_bandwidth_gbps
    if info.sm_version in (75, 86, 89) and "float" in dtype and use_f16_accum:
        max_tops *= 2

    weight_nbytes = layer_config.weight_nbytes // (layer_config.num_experts or 1)
    shape_n = layer_config.shape_n
    shape_k = layer_config.shape_k
    left_bias = weight_nbytes / max_bandwidth
    left_factor = shape_k * layer_config.a_dtype.num_bits / 8 / max_bandwidth
    right_factor = shape_n * shape_k * 2 / max_tops
    return left_bias / (right_factor - left_factor) * 1e3


class DeviceHeuristics:
    max_smem_size: int = 0
    b16_allowed_dtypes: list[dtypes.DataType] = []
    b8_allowed_dtypes: list[dtypes.DataType] = []
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 0
    expert_probability_cv: float = 0.25

    @classmethod
    def should_use_pdl_for_input(cls, layer_config: LayerConfig, shape_m: int) -> bool:
        return False

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
        raise NotImplementedError

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
        compute_bound_min_shape_m = _estimate_compute_bound_threshold(layer_config, use_f16_accum)
        is_moe_a16 = gemm_type != GemmType.DENSE and layer_config.a_dtype.num_bits == 16

        # 1. base config
        group_size = layer_config.input_scale_group_size or layer_config.weight_scale_group_size
        config = cls.get_base_config(
            layer_config.a_dtype,
            layer_config.b_dtype,
            group_size,
            use_f16_accum,
            layer_config.use_fused_e8m0_scale,
            gemm_type,
            layer_config.shape_k,
        )
        block_shape_m, block_shape_n, block_shape_k = config["block_shape"]
        warp_shape_m, warp_shape_n, warp_shape_k = config["warp_shape"]
        num_ctas_per_sm = config.get("num_ctas_per_sm", 1)
        num_stages = config.get("num_stages", 3 if cls.sm_version != 75 else 2)
        output_chunk_rows = config.get("output_chunk_rows", 0)
        max_num_stages = 5 if cls.sm_version == 80 else 3
        num_warps_m = block_shape_m // warp_shape_m

        # 2. block_shape_m and warp_shape_m
        if not layer_config.num_experts:
            if shape_m <= block_shape_m:
                block_shape_m = round_up(shape_m, 16)
            else:
                blocks = [math.ceil(shape_m / ((i + 1) * 16)) for i in range(block_shape_m // 16)]
                block_shape_m = np.argmin(blocks).item() * 16 + 16
        else:
            for moe_block_size in [16, 32, 48, 64]:
                if shape_m / layer_config.num_experts / moe_block_size < 0.9:
                    break

            new_shape_m = int(shape_m / layer_config.num_experts / 0.9)
            new_shape_m = max(new_shape_m, 1)
            if block_shape_m == 128:
                if round_up(new_shape_m, 96) < round_up(new_shape_m, 64):
                    block_shape_m = 96
                elif round_up(new_shape_m, 128) < round_up(new_shape_m, 64) * 1.05:
                    block_shape_m = 128
                else:
                    block_shape_m = moe_block_size
            elif new_shape_m >= 64 and new_shape_m < 96:
                block_shape_m = 48
            else:
                block_shape_m = moe_block_size

        assert num_warps_m <= 2
        if num_warps_m == 2 and block_shape_m >= 64:
            block_shape_m = round_up(block_shape_m, 32)
            warp_shape_m = block_shape_m // 2
        elif num_warps_m == 2 and block_shape_m % 32 == 0:
            warp_shape_m = block_shape_m // 2
        else:
            warp_shape_m = block_shape_m
            num_warps_m = 1

        while layer_config.shape_n % block_shape_n != 0:
            assert block_shape_n > 64
            block_shape_n = block_shape_n // 2
            if warp_shape_n > layer_config.a_dtype.num_bits * 4:
                warp_shape_n = warp_shape_n // 2

        num_blocks_n = layer_config.shape_n // block_shape_n
        num_blocks_m = cls.estimate_num_blocks_m(layer_config, shape_m, block_shape_m)

        num_sms = current_device.sm_count
        min_grid_blocks = math.ceil(num_sms * num_ctas_per_sm / 2)
        # Large expert tiles need more than half a wave to amortize their
        # register footprint; small M tiles benefit more from a wider N.
        is_short_moe = is_moe_a16 and layer_config.shape_k <= 1024
        if is_short_moe and block_shape_m >= 64 and cls.sm_version < 100:
            min_grid_blocks = max(min_grid_blocks, math.ceil(num_sms * 2 / 3))
        while num_blocks_n * num_blocks_m < min_grid_blocks:
            prefer_m_split = shape_m > block_shape_m >= block_shape_n and num_blocks_m < num_blocks_n
            fitted_block_m = cls._fit_dense_block_m_to_grid(
                (block_shape_m, block_shape_n),
                layer_config,
                shape_m,
                gemm_type,
                num_ctas_per_sm,
            )
            if prefer_m_split and fitted_block_m != block_shape_m:
                break
            if warp_shape_n > layer_config.a_dtype.num_bits * 4 and block_shape_n > 64:
                warp_shape_n = warp_shape_n // 2
                block_shape_n = block_shape_n // 2
                num_blocks_n = num_blocks_n * 2
                continue
            elif block_shape_n > 64:
                block_shape_n = block_shape_n // 2
                num_blocks_n = num_blocks_n * 2
            elif num_ctas_per_sm > 1:
                num_ctas_per_sm = num_ctas_per_sm - 1
                continue
            else:
                break

        if block_shape_n < 256 and warp_shape_k == 1024 // layer_config.a_dtype.num_bits:
            block_shape_k = block_shape_k // 2
            warp_shape_k = warp_shape_k // 2

        num_warps_m = block_shape_m // warp_shape_m
        num_warps_n = block_shape_n // warp_shape_n
        num_warps_k = block_shape_k // warp_shape_k
        num_warps = num_warps_m * num_warps_n * num_warps_k * num_ctas_per_sm

        if num_warps < 8:
            block_shape = (block_shape_m, block_shape_n, block_shape_k)
            smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, num_stages)
            has_one_to_two_waves = num_sms <= num_blocks_n * num_blocks_m < 2 * num_sms
            prefer_two_ctas = num_ctas_per_sm == 1 and has_one_to_two_waves
            prefer_two_ctas &= not is_moe_a16 or block_shape_m < 64
            if prefer_two_ctas and smem_size * 2 <= cls.max_smem_size:
                num_ctas_per_sm = 2
                num_warps *= 2

            while num_warps < 8:
                if layer_config.shape_k % (block_shape_k * 2) != 0:
                    break
                block_shape_new = (block_shape_m, block_shape_n, block_shape_k * 2)
                smem_size = estimate_smem_size_layer(
                    layer_config,
                    block_shape_new,
                    gemm_type,
                    num_stages,
                )
                if smem_size * num_ctas_per_sm > cls.max_smem_size:
                    break
                block_shape = block_shape_new
                block_shape_k = block_shape_k * 2
                num_warps = num_warps * 2

        if num_warps < 8 and warp_shape_m % 32 == 0:
            warp_shape_m = warp_shape_m // 2
            num_warps = num_warps * 2

        if num_warps < 8 and num_ctas_per_sm == 1 and num_blocks_n * num_blocks_m >= num_sms:
            smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, num_stages)
            if smem_size * 2 <= cls.max_smem_size:
                num_ctas_per_sm = 2

        if shape_m < compute_bound_min_shape_m:
            b_block_bits = block_shape_n * block_shape_k * layer_config.b_dtype.num_bits
            b_load_iters = b_block_bits / 128 / (num_warps * 32 / num_ctas_per_sm)
            if warp_shape_k % (1024 // layer_config.a_dtype.num_bits) == 0 and b_load_iters >= 4:
                warp_shape_k = warp_shape_k // 2
                block_shape_k = block_shape_k // 2

        fitted_block_m = cls._fit_dense_block_m_to_grid(
            (block_shape_m, block_shape_n),
            layer_config,
            shape_m,
            gemm_type,
            num_ctas_per_sm,
        )
        # SM120 and later select MoE M tiles in their own architecture policy.
        has_enough_tiles = num_blocks_n * num_blocks_m * 2 >= num_sms * num_ctas_per_sm
        has_heavy_weights = layer_config.shape_n >= 4096 and layer_config.b_dtype.num_bits >= 8
        has_heavy_weights &= block_shape_m <= 32
        has_deep_pipeline = max_num_stages >= 4 and block_shape_m >= 64
        prefer_weight_reuse = has_enough_tiles and layer_config.shape_k > 4096
        prefer_weight_reuse &= has_heavy_weights or has_deep_pipeline
        if is_moe_a16 and cls.sm_version < 120 and not prefer_weight_reuse:
            tokens_per_expert = shape_m / layer_config.num_experts
            if tokens_per_expert <= 64:
                max_block_m = 16 if tokens_per_expert <= 16 else 32
                fitted_block_m = min(fitted_block_m, max_block_m)

        use_dense_output_grid = False
        if fitted_block_m != block_shape_m:
            fitted_warp_shape_n = warp_shape_n
            if layer_config.a_dtype.num_bits == 16:
                fitted_warp_shape_n = min(warp_shape_n, 32)
            num_warps_n = block_shape_n // fitted_warp_shape_n
            num_warps_m = 2 if fitted_block_m > 32 and fitted_block_m % 32 == 0 else 1
            target_k_warps = max(1, 4 // (num_warps_m * num_warps_n))
            min_warp_shape_k = 1024 // layer_config.a_dtype.num_bits
            if layer_config.a_dtype.num_bits == 16:
                min_warp_shape_k = min(min_warp_shape_k, warp_shape_k)
            fitted_warp_shape_k = min(
                block_shape_k,
                max(min_warp_shape_k, block_shape_k // target_k_warps),
            )
            num_warps_k = block_shape_k // fitted_warp_shape_k
            if num_warps_m * num_warps_n * num_warps_k < 4:
                fitted_block_m = block_shape_m
            else:
                block_shape_m = fitted_block_m
                num_blocks_m = cls.estimate_num_blocks_m(layer_config, shape_m, block_shape_m)
                warp_shape_m = block_shape_m // num_warps_m
                warp_shape_n = fitted_warp_shape_n
                warp_shape_k = fitted_warp_shape_k
                min_grid_blocks = math.ceil(num_sms * num_ctas_per_sm / 2)
                has_long_a16_k = layer_config.a_dtype.num_bits == 16 and layer_config.shape_k > 4096
                use_dense_output_grid = gemm_type == GemmType.DENSE and cls.sm_version >= 120
                use_dense_output_grid &= not has_long_a16_k and num_blocks_n * num_blocks_m >= min_grid_blocks

        for num_stages_new in range(num_stages + 1, max_num_stages + 1):
            block_shape = (block_shape_m, block_shape_n, block_shape_k)
            smem_size = estimate_smem_size_layer(
                layer_config,
                block_shape,
                gemm_type,
                num_stages_new,
            )
            if smem_size * num_ctas_per_sm < cls.max_smem_size:
                num_stages = num_stages_new

        use_stream_k = True
        if use_batch_invariant:
            warp_shape_k = 512 // layer_config.a_dtype.num_bits
            block_shape_k = 512 // layer_config.a_dtype.num_bits
            use_stream_k = False

            if cls.sm_version != 75:
                num_warps_m = block_shape_m // warp_shape_m
                warp_shape_m = round_up(warp_shape_m, 16)
                block_shape_m = num_warps_m * warp_shape_m

        while layer_config.shape_k % block_shape_k != 0:
            block_shape_k = block_shape_k // 2
            if use_batch_invariant:
                warp_shape_k = block_shape_k
            else:
                warp_shape_k = 512 // layer_config.a_dtype.num_bits
                assert block_shape_k >= warp_shape_k

        use_stream_k = layer_config.shape_k > 1024 and use_stream_k and not use_dense_output_grid
        use_fp32_accum = layer_config.a_dtype.is_floating_point_type and not use_f16_accum
        can_use_output_grid = gemm_type == GemmType.DENSE or is_moe_a16 and layer_config.shape_k > 4096
        if use_fp32_accum and can_use_output_grid:
            blocks_per_wave = num_sms * num_ctas_per_sm
            num_output_tiles = num_blocks_m * num_blocks_n
            num_waves = math.ceil(num_output_tiles / blocks_per_wave)
            wave_utilization = num_output_tiles / (num_waves * blocks_per_wave)
            has_full_wave = num_waves == 1 and wave_utilization >= 0.9
            if is_moe_a16 and num_waves == 1 and num_stages >= 4:
                # Account for Scheduler's aligned slices and pipeline startup,
                # as in SM100's work estimate, before splitting a single wave.
                stream_stages = min(num_stages, max_num_stages - 1) if block_shape_m >= 64 else num_stages
                k_iters = layer_config.shape_k // block_shape_k
                scale_group = max(layer_config.input_scale_group_size, layer_config.weight_scale_group_size)
                alignment = math.lcm(max(1, scale_group // block_shape_k), stream_stages)
                slice_iters = math.ceil(num_output_tiles * k_iters / blocks_per_wave)
                slice_iters = max(slice_iters, math.ceil((k_iters - 1) / 9))
                slice_iters = round_up(slice_iters, alignment)
                has_full_wave = slice_iters + 2 * stream_stages >= k_iters * 0.9
            has_many_full_waves = gemm_type == GemmType.DENSE and num_waves >= 8 and wave_utilization >= 0.98
            if has_full_wave or has_many_full_waves:
                use_stream_k = False

        has_small_dense_tile = gemm_type == GemmType.DENSE and block_shape_m * block_shape_n < 16384
        has_long_wide_weights = layer_config.shape_k > 4096 and layer_config.b_dtype.num_bits >= 7
        keep_full_pipeline = has_small_dense_tile and has_long_wide_weights and max_num_stages == 3
        has_large_a16_tile = layer_config.a_dtype.num_bits == 16 and block_shape_m >= 64
        has_large_a16_tile &= not keep_full_pipeline
        if use_fp32_accum and use_stream_k and has_large_a16_tile:
            num_stages = min(num_stages, max_num_stages - 1)
        if is_moe_a16 and use_stream_k:
            # A fifth stage needs enough K work to amortize its startup and
            # Scheduler's coarser stage-aligned Stream-K slices.
            num_stages = min(num_stages, max(4, layer_config.shape_k // block_shape_k // 2))
        if use_batch_invariant:
            assert not use_stream_k
            assert block_shape_k == warp_shape_k

        if not use_stream_k:
            max_blocks_m = num_blocks_m
            if gemm_type != GemmType.DENSE:
                # Expert boundaries can each add a partially filled M tile.
                max_blocks_m = math.ceil(shape_m / block_shape_m) + layer_config.num_experts - 1
                max_blocks_m = min(shape_m, max_blocks_m)
            num_sms = min(num_sms, math.ceil(num_blocks_n * max_blocks_m / num_ctas_per_sm))

        return {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (warp_shape_m, warp_shape_n, warp_shape_k),
            "use_stream_k": use_stream_k,
            "use_f16_accum": use_f16_accum,
            "num_sms": num_sms,
            "num_stages": num_stages,
            "num_ctas_per_sm": num_ctas_per_sm,
            "output_chunk_rows": output_chunk_rows,
            "use_pdl": cls.sm_version >= 90,
        }

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _sample_expert_rows(shape_m: int, num_experts: int, probability_cv: float) -> np.ndarray:
        """Sample total routed rows, including top-k, without changing global RNG state."""
        if not math.isfinite(probability_cv) or probability_cv < 0:
            raise ValueError("expert probability CV must be finite and nonnegative")
        random_state = np.random.RandomState(0)
        counts = []
        for _ in range(16):
            if probability_cv == 0:
                probabilities = np.full(num_experts, 1 / num_experts)
            else:
                weights = random_state.gamma(1 / probability_cv**2, size=num_experts)
                probabilities = weights / weights.sum()
            counts.append(random_state.multinomial(shape_m, probabilities))
        result = np.asarray(counts)
        result.flags.writeable = False
        return result

    @classmethod
    def estimate_num_blocks_m(cls, layer_config: LayerConfig, shape_m: int, block_shape_m: int):
        if not layer_config.num_experts:
            return math.ceil(shape_m / block_shape_m)

        # Round each expert separately to account for empty experts and uneven padding.
        counts = cls._sample_expert_rows(shape_m, layer_config.num_experts, cls.expert_probability_cv)
        m_tiles = ((counts + block_shape_m - 1) // block_shape_m).sum(axis=1)
        return math.ceil(float(m_tiles.mean()))

    @classmethod
    def _fit_dense_block_m_to_grid(
        cls,
        block_shape: tuple[int, int],
        layer_config: LayerConfig,
        shape_m: int,
        gemm_type: GemmType,
        num_ctas_per_sm: int,
    ) -> int:
        block_m, block_n = block_shape
        is_dense_a16 = gemm_type == GemmType.DENSE and layer_config.a_dtype.num_bits == 16
        if not is_dense_a16 or block_m <= 16 or shape_m <= block_m <= 32:
            return block_m

        num_blocks_m = math.ceil(shape_m / block_m)
        num_blocks_n = layer_config.shape_n // block_n
        is_short_k = layer_config.shape_k <= 1024
        wave_fraction = 2 if is_short_k else 4
        target_blocks = math.ceil(current_device.sm_count * num_ctas_per_sm / wave_fraction)
        if num_blocks_m * num_blocks_n >= target_blocks:
            return block_m

        max_blocks_m = num_blocks_m * (8 if is_short_k else 4)
        target_blocks = min(target_blocks, max_blocks_m * num_blocks_n)
        candidates = [
            candidate
            for candidate in range(16, block_m, 16)
            if num_blocks_n * math.ceil(shape_m / candidate) >= target_blocks
            and math.ceil(shape_m / candidate) <= max_blocks_m
        ]
        return min(
            candidates,
            key=lambda candidate: (round_up(shape_m, candidate), -candidate),
            default=block_m,
        )

    @classmethod
    def get_configs(
        cls,
        layer_config: LayerConfig,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_m_major_input_scale: bool = False,
    ):
        a_dtype = layer_config.a_dtype
        if a_dtype.num_bits == 16:
            assert a_dtype in cls.b16_allowed_dtypes
        elif a_dtype.num_bits == 8:
            assert a_dtype in cls.b8_allowed_dtypes
        elif a_dtype.num_bits == 4:
            assert a_dtype in cls.b4_allowed_dtypes
        else:
            raise AssertionError(f"unsupported a_dtype {a_dtype} on sm{cls.sm_version}")

        last_shape_m = 0
        configs: list[list[int | dict]] = []
        last_config_str: str = ""

        if not layer_config.num_experts:
            max_shape_m = 8192
        else:
            max_shape_m = 65536

        shape_m_candidates = [1, 2, 4, 8]
        if cls.sm_version == 90:
            shape_m_candidates += list(range(8, max_shape_m, 8))
        else:
            shape_m_candidates += list(range(16, max_shape_m, 16))

        for shape_m in shape_m_candidates:
            if shape_m > 1024 and shape_m % 16 != 0:
                continue
            if shape_m > 2048 and shape_m % 32 != 0:
                continue
            if shape_m > 4096 and shape_m % 64 != 0:
                continue
            if shape_m > 16384 and shape_m % 128 != 0:
                continue

            config = cls.get_config(
                layer_config=layer_config,
                shape_m=shape_m,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
                use_m_major_input_scale=use_m_major_input_scale,
            )
            config_str = str(config)

            if last_config_str == config_str:
                configs[-1][1] = shape_m
            else:
                configs.append([last_shape_m, shape_m, config])

            last_config_str = config_str
            last_shape_m = shape_m

        configs[-1][1] = 1 << 30

        return configs
