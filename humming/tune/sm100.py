import dataclasses
import functools
import math

import numpy as np

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType, SmemReuseMode
from humming.device import current_device
from humming.tune.base import DeviceHeuristics
from humming.tune.sm8x import Sm80Heuristics
from humming.utils.math import round_up
from humming.utils.smem import estimate_smem_size_layer


class Sm100MmaHeuristics(Sm80Heuristics):
    max_smem_size: int = 227 * 1024
    sm_version: int = 100
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
        config = super().get_config(layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type)
        block_m, block_n, _ = config["block_shape"]
        warp_m = config["warp_shape"][0]
        if use_batch_invariant:
            return config

        warp_k = 1024 // layer_config.a_dtype.num_bits
        num_stages = config["num_stages"]
        # Long Stream-K slices can amortize a wider weight tile and a deeper
        # pipeline. Keep four warps, including when the inherited M tile was
        # split between two warps, to avoid extra intra-CTA partitioning.
        candidate_n, candidate_k, candidate_stages = 128, 2 * warp_k, 4
        has_aligned_tile = layer_config.shape_n % candidate_n == 0
        has_aligned_tile = has_aligned_tile and layer_config.shape_k % candidate_k == 0
        if config["use_stream_k"] and block_m <= 32 and has_aligned_tile:
            m_tiles = cls.estimate_num_blocks_m(layer_config, shape_m, block_m)
            output_tiles = m_tiles * (layer_config.shape_n // candidate_n)
            k_iters = layer_config.shape_k // candidate_k
            resident_ctas = config["num_sms"] * 2
            slice_iters = math.ceil(output_tiles * k_iters / resident_ctas)
            if min(slice_iters, k_iters) >= 2 * candidate_stages:
                block_n = candidate_n
                warp_m = block_m
                num_stages = candidate_stages

        if block_m != warp_m:
            return config

        # Small M needs little math parallelism. Use four warps per CTA and
        # two resident CTAs instead of expanding K to fill a single CTA.
        # Start from the existing M/N grid and Stream-K policy for dense and MoE.
        warp_n = min(64, block_n // 2)
        n_warps = block_n // warp_n
        block_k = (4 // n_warps) * warp_k
        if layer_config.shape_k % block_k:
            return config

        if config["use_stream_k"] and block_n < 128:
            m_tiles = cls.estimate_num_blocks_m(layer_config, shape_m, block_m)
            output_tiles = m_tiles * (layer_config.shape_n // block_n)
            k_iters = layer_config.shape_k // block_k
            resident_ctas = config["num_sms"] * 2
            slice_iters = math.ceil(output_tiles * k_iters / resident_ctas)
            if slice_iters <= num_stages and layer_config.shape_n % (2 * block_n) == 0:
                # Very short Stream-K slices spend much of their time on
                # startup and reduction. Merge adjacent N tiles and shorten
                # the pipeline instead of creating more tiny slices.
                block_n *= 2
                warp_n *= 2
                num_stages = 2

        block_shape = (block_m, block_n, block_k)
        warp_shape = (warp_m, warp_n, warp_k)
        smem_size = estimate_smem_size_layer(
            layer_config,
            block_shape,
            gemm_type,
            num_stages,
            warp_shape=warp_shape,
            num_write_splits=config["num_write_splits"],
            mma_accum_bits=16 if use_f16_accum else 32,
        )
        if smem_size * 2 > cls.max_smem_size:
            return config
        return config | {
            "block_shape": block_shape,
            "warp_shape": warp_shape,
            "num_ctas_per_sm": 2,
            "num_stages": num_stages,
        }


class Sm100UmmaHeuristics(DeviceHeuristics):
    max_smem_size = 227 * 1024
    sm_version = 100
    expert_probability_cv = 0.25
    b16_allowed_dtypes = [dtypes.float16, dtypes.bfloat16]

    @classmethod
    def _fits_resources(cls, layer_config, block_shape, num_stages, num_ctas_per_sm):
        block_m, block_n, block_k = block_shape
        output_groups = math.ceil(block_n / 128)
        operand_columns = block_k // 2
        stage_columns = output_groups * (num_stages * operand_columns + block_m)
        tmem_columns = 1 << (stage_columns - 1).bit_length()
        if tmem_columns * num_ctas_per_sm > 512:
            # UMMA falls back to two operand buffers when stage-local operands
            # do not fit. Account for that allocation before checking residency.
            buffered_columns = output_groups * (2 * operand_columns + block_m)
            tmem_columns = 1 << (buffered_columns - 1).bit_length()
        if tmem_columns * num_ctas_per_sm > 512:
            return False

        smem_size = estimate_smem_size_layer(
            layer_config,
            block_shape,
            GemmType.DENSE,
            num_stages,
            warp_shape=(block_m, 32, block_k),
            smem_reuse_mode=SmemReuseMode.NONE,
        )
        return smem_size * num_ctas_per_sm <= cls.max_smem_size

    @staticmethod
    def _get_stream_k_work(layer_config, output_tiles, k_iters, block_k, num_stages, resident_ctas):
        # Mirror Scheduler's tail-wave selection and scale/stage alignment.
        stream_tiles = output_tiles
        if output_tiles > resident_ctas:
            stream_tiles = output_tiles % resident_ctas
            if stream_tiles and stream_tiles * 10 <= resident_ctas:
                stream_tiles += resident_ctas
        data_parallel_waves = (output_tiles - stream_tiles) // resident_ctas
        scale_group = max(layer_config.input_scale_group_size, layer_config.weight_scale_group_size, 1)
        blocks_per_group = max(scale_group // block_k, 1)
        alignment = math.lcm(blocks_per_group, num_stages)
        max_slices = 1 << (layer_config.c_dtype.mantissa_bits // 2)
        min_slice_iters = math.ceil((k_iters - 1) / (max_slices - 1))
        slice_iters = max(math.ceil(stream_tiles * k_iters / resident_ctas), min_slice_iters)
        slice_iters = round_up(slice_iters, alignment) if stream_tiles else 0
        return data_parallel_waves * k_iters + slice_iters

    @classmethod
    def _select_m_tile(cls, layer_config, shape_m, config, use_batch_invariant):
        block_m, block_n, block_k = config["block_shape"]
        num_stages = config["num_stages"]
        num_ctas = config["num_ctas_per_sm"]
        num_sms = current_device.sm_count
        resident_ctas = num_sms * num_ctas
        n_blocks = layer_config.shape_n // block_n
        m_blocks = math.ceil(shape_m / block_m)
        if m_blocks * n_blocks >= resident_ctas:
            return config

        k_iters = layer_config.shape_k // block_k
        # Proxy for repeated B traffic: a converted TMEM operand plus its
        # compressed input and, when present, per-K-group scales.
        weight_bits = layer_config.b_dtype.num_bits
        if layer_config.is_group_weight_scale:
            weight_bits += layer_config.bs_dtype.num_bits / layer_config.weight_scale_group_size
        weight_cost = 1 + weight_bits / layer_config.a_dtype.num_bits
        if not layer_config.use_native_dequant and not layer_config.use_fused_e8m0_scale:
            # Software unpacking and conversion add register work to every
            # repeated B tile. This is a coarse cost, not an instruction count.
            weight_cost += 2

        def evaluate(candidate_m):
            tiles = math.ceil(shape_m / candidate_m) * n_blocks
            waves = math.ceil(tiles / resident_ctas)
            work = waves * k_iters
            stream_work = cls._get_stream_k_work(
                layer_config, tiles, k_iters, block_k, num_stages, resident_ctas
            )
            use_stream_k = not use_batch_invariant and stream_work + 2 * num_stages < work
            if use_stream_k:
                work = stream_work + 2 * num_stages
            # TMEM output is processed in 32-row chunks. Penalize repeated B
            # conversion as M tiles multiply, even before the grid is full.
            row_chunks = math.ceil(candidate_m / 32)
            latency = work * (1 + row_chunks / 4) + waves * row_chunks
            weight_work = tiles * k_iters * weight_cost / num_sms
            return latency + weight_work, use_stream_k

        baseline_score, _ = evaluate(block_m)
        best_score = baseline_score
        best = config
        candidate_m = block_m
        while candidate_m > 8:
            candidate_m = round_up(math.ceil(candidate_m / 2), 8)
            shape = (candidate_m, block_n, block_k)
            if not cls._fits_resources(layer_config, shape, num_stages, num_ctas):
                continue
            score, use_stream_k = evaluate(candidate_m)
            if score < best_score:
                best_score = score
                best = config | {
                    "block_shape": shape,
                    "warp_shape": (candidate_m, 32, block_k),
                    "use_stream_k": use_stream_k,
                }
        # Small predicted differences do not justify additional tiles; retain
        # the baseline on ties and near ties rather than always splitting M.
        return best if best_score < baseline_score * 0.9 else config

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if layer_config.num_experts:
            return cls._get_moe_config(layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type)
        if gemm_type != GemmType.DENSE:
            raise ValueError("non-dense UMMA requires experts")
        if use_f16_accum:
            raise ValueError("UMMA requires FP32 accumulation")
        if shape_m <= 0:
            raise ValueError("shape_m must be positive")
        layer_config = dataclasses.replace(layer_config, mma_type=MmaType.UMMA)

        # Balance the final M tile before considering additional parallelism.
        m_blocks = math.ceil(shape_m / 256)
        block_m = round_up(math.ceil(shape_m / m_blocks), 8)
        shape_n, shape_k = layer_config.shape_n, layer_config.shape_k
        num_sms = current_device.sm_count

        for block_k in (64, 32):
            if shape_k % block_k:
                continue
            k_iters = shape_k // block_k
            for block_n in (256, 128, 64):
                if shape_n % block_n:
                    continue
                block_shape = (block_m, block_n, block_k)
                output_tiles = m_blocks * (shape_n // block_n)
                # Narrow short-K grids need more N tiles because splitting K
                # cannot provide enough work for all SMs.
                can_fill_grid = output_tiles * k_iters >= num_sms * 4
                if block_n > 64 and output_tiles < num_sms and not can_fill_grid:
                    continue

                for num_ctas_per_sm in (2, 1):
                    resident_ctas = num_sms * num_ctas_per_sm
                    # Two resident CTAs hide latency with a short pipeline.
                    # With one CTA, wider N provides more work per stage.
                    output_groups = math.ceil(block_n / 128)
                    target_stages = 5 - output_groups - (num_ctas_per_sm - 1)
                    target_stages = min(target_stages, max(2, k_iters))
                    for num_stages in range(target_stages, 1, -1):
                        if not cls._fits_resources(layer_config, block_shape, num_stages, num_ctas_per_sm):
                            continue
                        stream_k_work = cls._get_stream_k_work(
                            layer_config, output_tiles, k_iters, block_k, num_stages, resident_ctas
                        )
                        data_parallel_work = math.ceil(output_tiles / resident_ctas) * k_iters
                        use_stream_k = (
                            not use_batch_invariant and stream_k_work + 2 * num_stages < data_parallel_work
                        )
                        if num_ctas_per_sm == 2 and output_tiles < resident_ctas:
                            active_ctas = math.ceil(output_tiles * k_iters / max(stream_k_work, 1))
                            if not use_stream_k or active_ctas <= num_sms:
                                continue
                        # For a single short wave, cp.async avoids TMA setup
                        # without sacrificing overlap across persistent tiles.
                        is_short_wave = output_tiles <= resident_ctas and k_iters <= num_stages
                        use_tma = not is_short_wave
                        config = {
                            "mma_type": MmaType.UMMA.value,
                            "block_shape": block_shape,
                            "warp_shape": (block_m, 32, block_k),
                            "num_stages": num_stages,
                            "num_ctas_per_sm": num_ctas_per_sm,
                            "smem_reuse_mode": SmemReuseMode.NONE.value,
                            "use_warp_spec": True,
                            "use_tma": use_tma,
                            "use_tma_a": use_tma,
                            "use_tma_c": use_tma,
                            "use_stream_k": use_stream_k,
                            "use_pdl": False,
                            "raster_group_m": 1,
                        }
                        return cls._select_m_tile(layer_config, shape_m, config, use_batch_invariant)

        raise ValueError("no resource-feasible dense UMMA tile for this layer")

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
    @functools.lru_cache(maxsize=64)
    def _get_moe_candidates(cls, layer_config: LayerConfig, gemm_type: GemmType) -> tuple:
        candidates = []
        indexed = gemm_type == GemmType.INDEXED
        for block_n in (256, 128) if layer_config.shape_n % 128 == 0 else (64,):
            if layer_config.shape_n % block_n:
                continue
            output_groups = math.ceil(block_n / 128)
            for block_k in (64,) if layer_config.shape_k % 64 == 0 else (32,):
                if layer_config.shape_k % block_k:
                    continue
                k_iters = layer_config.shape_k // block_k
                for num_ctas in (2, 1) if block_n == 256 else (2,):
                    max_block_m = min(256, 512 // (output_groups * num_ctas) - block_k)
                    for block_m in range(8, max_block_m + 1, 8):
                        for stages in range(min(4, max(2, k_iters)), 1, -1):
                            stage_columns = output_groups * (block_m + stages * block_k // 2)
                            tmem_columns = 1 << (stage_columns - 1).bit_length()
                            buffers = stages
                            if tmem_columns * num_ctas > 512:
                                buffers = 2
                                columns = output_groups * (block_m + buffers * block_k // 2)
                                tmem_columns = 1 << (columns - 1).bit_length()
                            if tmem_columns * num_ctas > 512:
                                continue
                            shape = (block_m, block_n, block_k)
                            smem_size = estimate_smem_size_layer(
                                layer_config,
                                shape,
                                gemm_type,
                                stages,
                                warp_shape=(block_m, 32, block_k),
                                smem_reuse_mode=SmemReuseMode.NONE,
                                use_mbarrier=True,
                                use_warp_spec=True,
                            )
                            if smem_size * num_ctas > cls.max_smem_size:
                                continue
                            config = {
                                "mma_type": MmaType.UMMA.value,
                                "block_shape": shape,
                                "warp_shape": (block_m, 32, block_k),
                                "num_stages": stages,
                                "num_ctas_per_sm": num_ctas,
                                "smem_reuse_mode": SmemReuseMode.NONE.value,
                                "use_warp_spec": True,
                                "use_tma": True,
                                "use_tma_a": not indexed,
                                "use_tma_c": not indexed,
                                "use_stream_k": False,
                                "use_pdl": False,
                                "raster_group_m": 1,
                            }
                            candidates.append(config)
                            # Use the deepest legal pipeline for each tile.
                            break
        return tuple(candidates)

    @classmethod
    def _get_moe_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.INDEXED,
    ) -> dict:
        if not layer_config.num_experts or gemm_type == GemmType.DENSE:
            raise ValueError("UMMA MoE requires an expert GEMM")
        if shape_m <= 0:
            raise ValueError("shape_m must be positive")
        if use_f16_accum:
            raise ValueError("UMMA requires FP32 accumulation")
        layer_config = dataclasses.replace(layer_config, mma_type=MmaType.UMMA)
        # The launcher selects grouped kernels using valid_shape_m when supplied.
        counts = cls._sample_expert_rows(shape_m, layer_config.num_experts, cls.expert_probability_cv)
        num_sms = current_device.sm_count
        best_by_residency = {}
        for config in cls._get_moe_candidates(layer_config, gemm_type):
            block_m, block_n, _ = config["block_shape"]
            num_ctas = config["num_ctas_per_sm"]
            resident_ctas = num_sms * num_ctas
            m_tiles = ((counts + block_m - 1) // block_m).sum(axis=1)
            tiles = m_tiles * (layer_config.shape_n // block_n)
            waves = (tiles + resident_ctas - 1) // resident_ctas

            # Balance serial tile rounds against padded row work per SM.
            # N normalizes the amount of weight processed by each tile.
            # Counting full waves also penalizes insufficient parallelism.
            tile_rounds = waves * block_n
            padded_work = tile_rounds * num_ctas * block_m
            score = float(np.sqrt(tile_rounds * padded_work).mean())
            # Larger M can force a shallower pipeline at the same residency.
            # Charge for the reduced load lookahead instead of comparing only
            # tile counts and padding. This keeps the tradeoff continuous.
            pipeline_penalty = 1 + 1 / config["num_stages"]
            score *= pipeline_penalty
            previous = best_by_residency.get(num_ctas)
            if previous is None or score < previous[0]:
                best_by_residency[num_ctas] = (score, config, tiles)

        best = best_by_residency.get(2, best_by_residency.get(1))
        single_cta = best_by_residency.get(1)
        # Require a clear improvement before giving up the second resident CTA.
        if best is not None and single_cta is not None and single_cta[0] < best[0] * 0.95:
            best = single_cta
        if best is None:
            raise ValueError("no resource-feasible UMMA MoE tile for this layer")
        _, config, tiles = best
        use_stream_k = False
        if not use_batch_invariant:
            block_k = config["block_shape"][2]
            stages = config["num_stages"]
            resident_ctas = num_sms * config["num_ctas_per_sm"]
            k_iters = layer_config.shape_k // block_k
            data_parallel_work = ((tiles + resident_ctas - 1) // resident_ctas) * k_iters
            stream_work_samples = [
                cls._get_stream_k_work(layer_config, int(tile_count), k_iters, block_k, stages, resident_ctas)
                for tile_count in tiles
            ]
            stream_work = np.asarray(stream_work_samples)
            stream_work = stream_work + 2 * stages
            use_stream_k = bool(stream_work.mean() < data_parallel_work.mean() * 0.9)
        return config | {"use_stream_k": use_stream_k}


class Sm100Heuristics(Sm100MmaHeuristics):
    @classmethod
    def _should_use_mma(cls, layer_config: LayerConfig, shape_m: int) -> bool:
        effective_m = float(shape_m)
        if layer_config.num_experts:
            counts = Sm100UmmaHeuristics._sample_expert_rows(
                shape_m, layer_config.num_experts, Sm100UmmaHeuristics.expert_probability_cv
            )
            # Weight experts by their routed work, so empty experts do not
            # make a few large experts look like a small-M workload.
            effective_m = float((counts * counts).sum() / counts.sum())
        # With many experts, UMMA becomes useful sooner than in a sparse
        # dense grid. Keep MMA while the weighted expert load fits M16.
        if layer_config.num_experts:
            max_m = 16
        else:
            # Dense measurements favor MMA through M48 with long K, and
            # through M64 with short K.
            max_m = 64 if layer_config.shape_k <= 512 else 48
        return effective_m <= max_m

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if shape_m <= 0:
            raise ValueError("shape_m must be positive")
        if layer_config.mma_type == MmaType.UMMA:
            if not use_f16_accum and not cls._should_use_mma(layer_config, shape_m):
                return Sm100UmmaHeuristics.get_config(
                    layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type
                )
            layer_config = dataclasses.replace(layer_config, mma_type=MmaType.MMA)
        config = Sm100MmaHeuristics.get_config(
            layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type
        )
        return config | {"mma_type": layer_config.mma_type.value}

    @classmethod
    def get_umma_config(cls, layer_config: LayerConfig, shape_m: int, gemm_type: GemmType):
        # Explicit UMMA entry point bypasses automatic backend selection.
        return Sm100UmmaHeuristics.get_config(layer_config, shape_m, gemm_type=gemm_type)
