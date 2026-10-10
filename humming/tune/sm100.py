import functools
import math
import os

import numpy as np

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType, SmemReuseMode
from humming.config.mma import get_default_mma_type
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
        use_m_major_input_scale: bool = False,
    ):
        config = super().get_config(
            layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type, use_m_major_input_scale
        )
        block_m, block_n, _ = config["block_shape"]
        warp_m = config["warp_shape"][0]
        if use_batch_invariant:
            return config

        if block_m > 32:
            is_a16_fp32 = layer_config.a_dtype.num_bits == 16 and not use_f16_accum
            use_long_stream_k = layer_config.shape_k > 4096 and config["use_stream_k"]
            if not is_a16_fp32 or not use_long_stream_k or block_m < 64:
                return config

            # Long K benefits from K-parallel warps and a deeper pipeline after splitting M.
            block_k = config["block_shape"][2]
            warp_k = 32
            if block_m < block_n and block_k < 128 and layer_config.shape_k % 128 == 0:
                block_k = 128
                warp_k = 64
            candidate = config | {
                "block_shape": (block_m, block_n, block_k),
                "warp_shape": (warp_m, min(block_n, 64), warp_k),
                "num_stages": 3,
            }
            num_warps = block_m * block_n * block_k // math.prod(candidate["warp_shape"])
            smem_size = estimate_smem_size_layer(
                layer_config,
                candidate["block_shape"],
                gemm_type,
                3,
                mma_type=MmaType.MMA,
                warp_shape=candidate["warp_shape"],
                output_chunk_rows=config["output_chunk_rows"],
            )
            if num_warps <= 8 and smem_size * config["num_ctas_per_sm"] <= cls.max_smem_size:
                return candidate
            return config

        # This four-warp schedule targets small M tiles.
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
            mma_type=MmaType.MMA,
            warp_shape=warp_shape,
            output_chunk_rows=config["output_chunk_rows"],
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
    b16_allowed_dtypes = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2, dtypes.float8e3m4]
    b4_allowed_dtypes = [dtypes.float4e2m1, dtypes.float4e0m3]

    @staticmethod
    def _get_tmem_columns(layer_config, block_shape, num_stages, num_ctas_per_sm, cta_group_size=1):
        block_m, block_n, block_k = block_shape
        output_groups = block_n // 128
        operand_columns = 0 if layer_config.use_raw_weight else block_k * layer_config.a_dtype.num_bits // 32
        constant_scale_columns = 0
        if layer_config.use_block_scaled_mma:
            scale_group_size = layer_config.mma_scale_group_size
            scale_words = math.ceil(block_k / (4 * scale_group_size))
            input_scale_stride = max(4, 1 << (math.ceil(block_m / 32) - 1).bit_length())
            if layer_config.is_group_weight_scale:
                operand_columns += scale_words * 4
            if layer_config.is_group_input_scale:
                operand_columns += scale_words * input_scale_stride
            if not layer_config.is_group_input_scale or not layer_config.is_group_weight_scale:
                constant_scale_columns = 16
            operand_columns = round_up(operand_columns, 16)
        accumulator_columns = (
            round_up(block_m, 32) if layer_config.use_raw_weight and cta_group_size == 2 else block_m
        )
        buffers = 1 if layer_config.use_raw_weight else (4 if cta_group_size == 2 else num_stages)
        columns = constant_scale_columns + output_groups * (buffers * operand_columns + accumulator_columns)
        tmem_columns = 1 << (columns - 1).bit_length()
        if not layer_config.use_raw_weight and tmem_columns * num_ctas_per_sm > 512:
            # TS falls back to two operand buffers; SS copies scales on the
            # issuer stream and needs only one, independently of stage count.
            columns = constant_scale_columns + output_groups * (2 * operand_columns + accumulator_columns)
            tmem_columns = 1 << (columns - 1).bit_length()
        return tmem_columns

    @classmethod
    def _fits_resources(
        cls,
        layer_config,
        block_shape,
        num_stages,
        num_ctas_per_sm,
        gemm_type=GemmType.DENSE,
        cta_group_size=1,
        output_chunk_rows=0,
        use_m_major_input_scale=False,
    ):
        tmem_columns = cls._get_tmem_columns(
            layer_config, block_shape, num_stages, num_ctas_per_sm, cta_group_size
        )
        if tmem_columns * num_ctas_per_sm > 512:
            return False
        block_m, _, block_k = block_shape
        if layer_config.a_dtype == dtypes.int8 and block_m > 32 and block_m % 16:
            return False
        smem_size = estimate_smem_size_layer(
            layer_config,
            block_shape,
            gemm_type,
            num_stages,
            mma_type=MmaType.UMMA,
            warp_shape=(block_m, 32, block_k),
            smem_reuse_mode=SmemReuseMode.NONE,
            use_warp_spec=True,
            umma_cta_group_size=cta_group_size,
            output_chunk_rows=output_chunk_rows,
            use_tma_c=gemm_type != GemmType.INDEXED,
            use_m_major_input_scale=use_m_major_input_scale,
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
    def _select_m_tile(
        cls, layer_config, shape_m, config, use_batch_invariant, use_m_major_input_scale=False
    ):
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
            fits_resources = cls._fits_resources(
                layer_config, shape, num_stages, num_ctas, use_m_major_input_scale=use_m_major_input_scale
            )
            if not fits_resources:
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
    def _select_low_bit_m_tile(cls, layer_config, shape_m, config, use_m_major_input_scale=False):
        a_dtype = layer_config.a_dtype
        if a_dtype.num_bits == 16:
            return config
        stage_k = 1024 // a_dtype.num_bits
        k_iters = layer_config.shape_k // stage_k
        # Every M tile re-reads the weights; long K keeps the Stream-K tile.
        if layer_config.shape_k % stage_k or k_iters >= 64:
            return config

        def align_m(rows):
            # INT8 UMMA needs M divisible by 16 above M32.
            alignment = 16 if a_dtype == dtypes.int8 and rows > 32 else 8
            return round_up(rows, alignment)

        num_sms = current_device.sm_count
        n_blocks = layer_config.shape_n // 128
        full_m = align_m(math.ceil(shape_m / math.ceil(shape_m / 256)))
        if math.ceil(shape_m / full_m) * n_blocks >= num_sms:
            return config

        # Halve M while one data-parallel wave still holds every tile.
        block_m = full_m
        candidate_m = full_m
        while candidate_m > 16:
            candidate_m = align_m(math.ceil(candidate_m / 2))
            if math.ceil(shape_m / candidate_m) * n_blocks > num_sms:
                break
            block_m = candidate_m

        # Short stages leave the tensor cores idle between loads: double K when four stages fit.
        for block_k, num_stages in ((2 * stage_k, 4), (stage_k, 4), (stage_k, 3), (stage_k, 2)):
            if layer_config.shape_k % block_k:
                continue
            block_shape = (block_m, 128, block_k)
            fits_resources = cls._fits_resources(
                layer_config, block_shape, num_stages, 1, use_m_major_input_scale=use_m_major_input_scale
            )
            if fits_resources:
                return config | {
                    "block_shape": block_shape,
                    "warp_shape": (block_m, 32, block_k),
                    "num_stages": num_stages,
                    "num_ctas_per_sm": 1,
                    "use_stream_k": False,
                    "use_tma": True,
                    "use_tma_a": True,
                    "use_tma_c": True,
                }
        return config

    @classmethod
    def _select_cooperative_ctas(cls, layer_config, shape_m, config, use_m_major_input_scale=False):
        activation_bits = layer_config.a_dtype.num_bits
        is_low_bit = activation_bits < 16
        # Low-bit cooperative instructions take native operands only.
        if is_low_bit and not layer_config.use_raw_weight:
            return config
        block_m, block_n, block_k = config["block_shape"]
        has_cooperative_tile = block_m >= 128 and block_m % 32 == 0
        has_cooperative_tile &= block_n == 128 and block_k == 1024 // activation_bits
        has_cooperative_tile &= layer_config.shape_n % (2 * block_n) == 0
        uses_tma = config["use_tma"] and config["use_tma_a"] and config["use_tma_c"]
        if not has_cooperative_tile or not uses_tma or config["num_ctas_per_sm"] != 1:
            return config
        if layer_config.is_block_weight_scale:
            return config

        # Stream-K balances partial waves across CTA pairs. Without it, retain
        # independent CTAs when cooperative output waves would be underfilled.
        output_tiles = math.ceil(shape_m / block_m) * (layer_config.shape_n // block_n)
        num_sms = current_device.sm_count // 2 * 2
        if not num_sms:
            return config
        scheduled_tiles = math.ceil(output_tiles / num_sms) * num_sms
        if not config["use_stream_k"] and output_tiles < 0.8 * scheduled_tiles:
            return config

        # Wider low-bit stages may not fit six; FP16 keeps its measured depth.
        min_stages = 4 if is_low_bit else 6
        k_iters = layer_config.shape_k // block_k
        pipeline_iters = k_iters
        if is_low_bit and config["use_stream_k"]:
            # An underfilled Stream-K grid gives each CTA only a slice of K.
            pipeline_iters = min(k_iters, output_tiles * k_iters // num_sms)
        for num_stages in range(6, min_stages - 1, -1):
            if pipeline_iters < 4 * num_stages:
                continue
            smem_size = estimate_smem_size_layer(
                layer_config,
                config["block_shape"],
                GemmType.DENSE,
                num_stages,
                mma_type=MmaType.UMMA,
                warp_shape=config["warp_shape"],
                smem_reuse_mode=SmemReuseMode.NONE,
                umma_cta_group_size=2,
                output_chunk_rows=32,
                use_m_major_input_scale=use_m_major_input_scale,
            )
            tmem_columns = cls._get_tmem_columns(layer_config, config["block_shape"], num_stages, 1, 2)
            if smem_size <= cls.max_smem_size and (not is_low_bit or tmem_columns <= 512):
                return config | {
                    "umma_cta_group_size": 2,
                    "output_chunk_rows": 32,
                    "num_stages": num_stages,
                }
        return config

    @classmethod
    def _select_stream_k(cls, layer_config, shape_m, config):
        if not config["use_stream_k"]:
            return config
        block_m, block_n, _ = config["block_shape"]
        cta_group_size = config.get("umma_cta_group_size", 1)
        resident_ctas = (current_device.sm_count // cta_group_size) * config["num_ctas_per_sm"]
        output_tiles = math.ceil(shape_m / block_m) * (layer_config.shape_n // (block_n * cta_group_size))
        num_waves = math.ceil(output_tiles / resident_ctas)
        last_wave_tiles = output_tiles % resident_ctas or resident_ctas
        # Many waves with a nearly full tail leave Stream-K little idle time to recover.
        if num_waves > 10 and last_wave_tiles > 0.75 * resident_ctas:
            return config | {"use_stream_k": False}
        return config

    @classmethod
    def _select_dequant_warpgroups(cls, layer_config, shape_m, config):
        # Native conversion keeps up with one warpgroup; register dequantization does not.
        if layer_config.use_raw_weight or layer_config.use_native_dequant:
            return config
        # Aligned integers convert to FP16/BF16 with one logic op and one subtraction.
        a_dtype, b_dtype = layer_config.a_dtype, layer_config.b_dtype
        is_aligned_integer = b_dtype.is_integer_type and b_dtype.num_bits in (1, 2, 4, 8)
        fits_mantissa = a_dtype.num_bits == 16 and b_dtype.num_bits <= a_dtype.mantissa_bits
        has_stage_scale = layer_config.is_group_weight_scale or layer_config.is_block_weight_scale
        if is_aligned_integer and fits_mantissa and not has_stage_scale:
            return config
        # Two resident CTAs on a single output group already contend for the SM.
        has_narrow_resident_pair = config["block_shape"][1] == 128 and config["num_ctas_per_sm"] == 2
        if has_narrow_resident_pair:
            return config
        return config | {"umma_num_dequant_warpgroups": 2}

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
        if layer_config.num_experts:
            return cls._get_moe_config(
                layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type, use_m_major_input_scale
            )
        if gemm_type != GemmType.DENSE:
            raise ValueError("non-dense UMMA requires experts")
        if use_f16_accum:
            raise ValueError("UMMA requires FP32 accumulation")
        if shape_m <= 0:
            raise ValueError("shape_m must be positive")

        # Balance the final M tile before considering additional parallelism.
        m_blocks = math.ceil(shape_m / 256)
        block_m = round_up(math.ceil(shape_m / m_blocks), 8)
        if layer_config.a_dtype == dtypes.int8 and block_m > 32:
            block_m = round_up(block_m, 16)
        shape_n, shape_k = layer_config.shape_n, layer_config.shape_k
        num_sms = current_device.sm_count

        for block_k in (1024 // layer_config.a_dtype.num_bits, 512 // layer_config.a_dtype.num_bits):
            if shape_k % block_k:
                continue
            k_iters = shape_k // block_k
            for block_n in (256, 128):
                if shape_n % block_n:
                    continue
                block_shape = (block_m, block_n, block_k)
                output_tiles = m_blocks * (shape_n // block_n)
                # Narrow short-K grids need more N tiles because splitting K
                # cannot provide enough work for all SMs.
                can_fill_grid = output_tiles * k_iters >= num_sms * 4
                if block_n > 128 and output_tiles < num_sms and not can_fill_grid:
                    continue

                for num_ctas_per_sm in (2, 1):
                    resident_ctas = num_sms * num_ctas_per_sm
                    # Two resident CTAs hide latency with a short pipeline.
                    # With one CTA, wider N provides more work per stage.
                    output_groups = block_n // 128
                    target_stages = 5 - output_groups - (num_ctas_per_sm - 1)
                    target_stages = min(target_stages, max(2, k_iters))
                    for num_stages in range(target_stages, 1, -1):
                        fits_resources = cls._fits_resources(
                            layer_config,
                            block_shape,
                            num_stages,
                            num_ctas_per_sm,
                            use_m_major_input_scale=use_m_major_input_scale,
                        )
                        if not fits_resources:
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
                        use_tma = layer_config.use_raw_weight or not is_short_wave
                        config = {
                            "mma_type": MmaType.UMMA.value,
                            "block_shape": block_shape,
                            "warp_shape": (block_m, 32, block_k),
                            "num_stages": num_stages,
                            "producer_stage_unroll": 4,
                            "consumer_stage_unroll": 4,
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
                        config = cls._select_m_tile(
                            layer_config, shape_m, config, use_batch_invariant, use_m_major_input_scale
                        )
                        if not use_batch_invariant:
                            config = cls._select_low_bit_m_tile(
                                layer_config, shape_m, config, use_m_major_input_scale
                            )
                        config = cls._select_cooperative_ctas(
                            layer_config, shape_m, config, use_m_major_input_scale
                        )
                        config = cls._select_stream_k(layer_config, shape_m, config)
                        return cls._select_dequant_warpgroups(layer_config, shape_m, config)

        raise ValueError("no resource-feasible dense UMMA tile for this layer")

    @classmethod
    @functools.lru_cache(maxsize=64)
    def _get_moe_candidates(
        cls, layer_config: LayerConfig, gemm_type: GemmType, use_m_major_input_scale: bool = False
    ) -> tuple:
        candidates = []
        indexed = gemm_type == GemmType.INDEXED
        for block_n in (256, 128):
            if layer_config.shape_n % block_n:
                continue
            output_groups = block_n // 128
            preferred_k = 1024 // layer_config.a_dtype.num_bits
            for block_k in (preferred_k,) if layer_config.shape_k % preferred_k == 0 else (preferred_k // 2,):
                if layer_config.shape_k % block_k:
                    continue
                k_iters = layer_config.shape_k // block_k
                residency_choices = (2, 1) if block_n == 256 or layer_config.use_raw_weight else (2,)
                for num_ctas in residency_choices:
                    operand_columns = (
                        0 if layer_config.use_raw_weight else block_k * layer_config.a_dtype.num_bits // 32
                    )
                    max_block_m = min(256, 512 // (output_groups * num_ctas) - 2 * operand_columns)
                    min_stages = min(3, max(2, k_iters))
                    max_stages = min(5 if layer_config.use_raw_weight else 4, max(2, k_iters))
                    can_cooperate = layer_config.use_raw_weight and num_ctas == 1
                    can_cooperate &= layer_config.shape_n % (2 * block_n) == 0
                    for cta_group_size in (1, 2) if can_cooperate else (1,):
                        for block_m in range(8 * cta_group_size, max_block_m + 1, 8 * cta_group_size):
                            use_chunks = layer_config.use_raw_weight and (block_m > 32 or cta_group_size == 2)
                            output_chunk_rows = 32 if use_chunks else 0
                            shape = (block_m, block_n, block_k)
                            for stages in range(max_stages, min_stages - 1, -1):
                                if not cls._fits_resources(
                                    layer_config,
                                    shape,
                                    stages,
                                    num_ctas,
                                    gemm_type,
                                    cta_group_size,
                                    output_chunk_rows,
                                    use_m_major_input_scale,
                                ):
                                    continue
                                config = {
                                    "mma_type": MmaType.UMMA.value,
                                    "block_shape": shape,
                                    "warp_shape": (block_m, 32, block_k),
                                    "num_stages": stages,
                                    "producer_stage_unroll": 4,
                                    "consumer_stage_unroll": 4,
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
                                if layer_config.use_raw_weight:
                                    config.update(
                                        umma_cta_group_size=cta_group_size,
                                        output_chunk_rows=output_chunk_rows,
                                    )
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
        use_m_major_input_scale: bool = False,
    ) -> dict:
        if not layer_config.num_experts or gemm_type == GemmType.DENSE:
            raise ValueError("UMMA MoE requires an expert GEMM")
        if shape_m <= 0:
            raise ValueError("shape_m must be positive")
        if use_f16_accum:
            raise ValueError("UMMA requires FP32 accumulation")
        # The launcher selects grouped kernels using valid_shape_m when supplied.
        counts = cls._sample_expert_rows(shape_m, layer_config.num_experts, cls.expert_probability_cv)
        num_sms = current_device.sm_count
        typical_expert_rows = float(np.median(counts))
        best_by_residency = {}
        for config in cls._get_moe_candidates(layer_config, gemm_type, use_m_major_input_scale):
            block_m, block_n, block_k = config["block_shape"]
            num_ctas = config["num_ctas_per_sm"]
            if layer_config.use_raw_weight and num_ctas == 1 and typical_expert_rows < block_m:
                # Give up the second resident CTA only when a typical expert
                # fills the wider M tile, rather than mostly padding it.
                continue
            cta_group_size = config.get("umma_cta_group_size", 1)
            resident_ctas = (num_sms // cta_group_size) * num_ctas
            m_tiles = ((counts + block_m - 1) // block_m).sum(axis=1)
            tiles = m_tiles * (layer_config.shape_n // (block_n * cta_group_size))
            waves = (tiles + resident_ctas - 1) // resident_ctas

            k_iters = layer_config.shape_k // block_k
            stages = config["num_stages"]
            data_parallel_work = waves * k_iters
            if cta_group_size == 2 and data_parallel_work.mean() < 4 * stages:
                continue
            work = data_parallel_work
            use_stream_k = False
            if not use_batch_invariant:
                stream_work_samples = [
                    cls._get_stream_k_work(
                        layer_config, int(tile_count), k_iters, block_k, stages, resident_ctas
                    )
                    for tile_count in tiles
                ]
                stream_work = np.asarray(stream_work_samples)
                stream_work = stream_work + 2 * stages
                # Reduction and tile transitions are expensive for short slices.
                # Require a substantial saving, and compare M/N with Stream-K
                # already included rather than enabling it after choosing a tile.
                use_stream_k = bool(stream_work.mean() < data_parallel_work.mean() * 0.5)
                if use_stream_k:
                    work = stream_work

            # Each tile pays shared activation/synchronization work in addition
            # to N-dependent conversion. A wider N amortizes that fixed cost.
            tile_rounds = work / k_iters * (block_n + 32)
            padded_work = tile_rounds * num_ctas * block_m
            score = float(np.sqrt(tile_rounds * padded_work).mean())
            pipeline_penalty = 1 + 1 / stages
            score *= pipeline_penalty
            config = config | {"use_stream_k": use_stream_k}
            previous = best_by_residency.get(num_ctas)
            if previous is None or score < previous[0]:
                best_by_residency[num_ctas] = (score, config)

        best = best_by_residency.get(2, best_by_residency.get(1))
        single_cta = best_by_residency.get(1)
        # Require a clear improvement before giving up the second resident CTA.
        if best is not None and single_cta is not None and single_cta[0] < best[0] * 0.95:
            best = single_cta
        if best is None:
            raise ValueError("no resource-feasible UMMA MoE tile for this layer")
        return best[1]


class Sm100Heuristics(Sm100MmaHeuristics):
    b8_allowed_dtypes = [*Sm100MmaHeuristics.b8_allowed_dtypes, dtypes.float8e3m4]
    b4_allowed_dtypes = [dtypes.float4e2m1, dtypes.float4e0m3]

    @classmethod
    def _should_use_mma(cls, layer_config: LayerConfig, shape_m: int) -> bool:
        # INT8 UMMA exists on SM100/SM110 only; other layers without UMMA use MMA.
        if not layer_config.is_umma_supported:
            return True
        # All supported UMMA N tiles are multiples of 128.
        if layer_config.shape_n % 128:
            return True
        a_dtype = layer_config.a_dtype
        if a_dtype.num_bits < 16:
            # FP8/FP4 measurements favor UMMA at every M, dense and MoE,
            # whenever K divides into UMMA stages. INT8 follows FP8.
            return layer_config.shape_k % (512 // a_dtype.num_bits) != 0
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
        use_m_major_input_scale: bool = False,
    ):
        if shape_m <= 0:
            raise ValueError("shape_m must be positive")
        default_test_source = "heuristic" if "PYTEST_CURRENT_TEST" in os.environ else ""
        test_sources = os.environ.get("HUMMING_TEST_TUNING_SOURCE", default_test_source).split("+")
        is_heuristic_test = "heuristic" in (source.strip() for source in test_sources)
        prefer_umma = is_heuristic_test and layer_config.is_umma_supported and not use_f16_accum
        prefer_umma &= layer_config.shape_n % 128 == 0
        prefer_umma &= layer_config.shape_k % (512 // layer_config.a_dtype.num_bits) == 0
        if prefer_umma or get_default_mma_type(layer_config) == MmaType.UMMA:
            has_native_mixed_operands = layer_config.a_dtype.num_bits == 8
            has_native_mixed_operands &= layer_config.b_dtype.is_floating_point_type
            has_native_mixed_operands &= layer_config.b_dtype != layer_config.a_dtype
            has_hidden_fp8 = dtypes.float8e3m4 in (layer_config.a_dtype, layer_config.b_dtype)
            requires_umma = layer_config.use_block_scaled_mma or has_native_mixed_operands or has_hidden_fp8
            has_mixed_raw_weights = layer_config.use_raw_weight and (
                layer_config.a_dtype.num_bits != layer_config.b_dtype.num_bits
            )
            requires_umma |= has_mixed_raw_weights
            if use_f16_accum and has_mixed_raw_weights:
                raise ValueError("native mixed weight layout requires UMMA with FP32 accumulation")
            keep_umma = requires_umma or prefer_umma or not cls._should_use_mma(layer_config, shape_m)
            if not use_f16_accum and keep_umma:
                return Sm100UmmaHeuristics.get_config(
                    layer_config,
                    shape_m,
                    use_f16_accum,
                    use_batch_invariant,
                    gemm_type,
                    use_m_major_input_scale,
                )
        config = Sm100MmaHeuristics.get_config(
            layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type, use_m_major_input_scale
        )
        return config | {"mma_type": MmaType.MMA.value}

    @classmethod
    def get_umma_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        gemm_type: GemmType,
        use_m_major_input_scale: bool = False,
    ):
        # Explicit UMMA entry point bypasses automatic backend selection.
        return Sm100UmmaHeuristics.get_config(
            layer_config, shape_m, gemm_type=gemm_type, use_m_major_input_scale=use_m_major_input_scale
        )
