import dataclasses
import math
from typing import Literal

import numpy as np

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType, TuningConfig
from humming.config.mma import get_register_budget_error
from humming.device import current_device
from humming.tune.base import DeviceHeuristics
from humming.tune.candidate import (
    CandidateAnalysis,
    ScheduleCandidate,
    TuningDecision,
    TuningProblem,
    analyze_candidate,
    fit_pipeline_stages,
    get_problem_rejection_reasons,
)
from humming.utils.smem import estimate_smem_size_layer


def get_sm90_specialized_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    use_m_major_input_scale: bool = False,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> dict | None:
    short_config = get_short_k_config(
        layer_config,
        shape_m,
        gemm_type,
        use_f16_accum,
        use_batch_invariant,
        is_h20=is_h20,
        use_m_major_input_scale=use_m_major_input_scale,
    )
    if short_config is not None:
        return short_config
    packed_config = get_packed_wna8_config(
        layer_config,
        shape_m,
        gemm_type,
        use_f16_accum,
        use_batch_invariant,
        is_h20=is_h20,
        use_m_major_input_scale=use_m_major_input_scale,
        expert_probability_cv=expert_probability_cv,
    )
    if packed_config is not None:
        return packed_config
    block_scaled_config = get_block_scaled_moe_config(
        layer_config,
        shape_m,
        gemm_type,
        use_f16_accum,
        use_batch_invariant,
        is_h20=is_h20,
        expert_probability_cv=expert_probability_cv,
    )
    if block_scaled_config is not None:
        return block_scaled_config
    return get_scaled_w8a8_config(
        layer_config,
        shape_m,
        gemm_type,
        use_f16_accum,
        use_batch_invariant,
        is_h20=is_h20,
        use_m_major_input_scale=use_m_major_input_scale,
        expert_probability_cv=expert_probability_cv,
    )


def get_short_k_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    use_m_major_input_scale: bool = False,
) -> dict | None:
    if gemm_type != GemmType.DENSE or use_f16_accum or use_batch_invariant:
        return None
    if layer_config.a_dtype.num_bits != 8 or layer_config.shape_k > 512:
        return None
    if layer_config.shape_n % 128 or layer_config.shape_k % 128:
        return None
    if get_problem_rejection_reasons(layer_config, MmaType.WGMMA):
        return None
    if is_h20 and layer_config.use_raw_weight and layer_config.shape_k > 256 and shape_m > 64:
        return None
    if layer_config.use_fused_e8m0_scale and layer_config.use_packed_k_layout and shape_m > 64:
        return None

    alignment = 16 if layer_config.a_dtype.is_integer_type else 8
    n_tiles = layer_config.shape_n // 128
    block_m = int(shape_m * n_tiles / current_device.sm_count) // alignment * alignment
    block_m = min(64, max(16, block_m))
    has_unpacked_group_weights = (
        not layer_config.use_raw_weight
        and not layer_config.use_packed_k_layout
        and layer_config.is_group_weight_scale
        and not layer_config.use_fused_e8m0_scale
    )
    warp_n = 32 if has_unpacked_group_weights else 16
    config = {
        "mma_type": MmaType.WGMMA.value,
        "block_shape": (block_m, 128, 128),
        "warp_shape": (block_m, warp_n, 128),
        "num_stages": 3,
        "num_ctas_per_sm": 2,
        "use_tma": False,
        "use_warp_spec": False,
        "use_stream_k": False,
        "smem_reuse_mode": "all_stages",
        "raster_group_m": 1,
    }
    tuning = TuningConfig(**config)
    if get_register_budget_error(layer_config, tuning, registers_per_sm=65536):
        return None
    smem_size = estimate_smem_size_layer(
        layer_config,
        config["block_shape"],
        gemm_type,
        config["num_stages"],
        warp_shape=config["warp_shape"],
        use_tma=False,
        use_warp_spec=False,
        mma_type=MmaType.WGMMA,
        use_m_major_input_scale=use_m_major_input_scale,
    )
    return config if smem_size * config["num_ctas_per_sm"] <= 227 * 1024 else None


def get_block_scaled_moe_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> dict | None:
    fp8_dtypes = (dtypes.float8e4m3, dtypes.float8e5m2)
    has_fp8_operands = layer_config.a_dtype in fp8_dtypes and layer_config.b_dtype == layer_config.a_dtype
    has_block_scales = layer_config.is_block_weight_scale
    is_grouped = gemm_type in (GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED)
    if not (has_fp8_operands and has_block_scales and is_grouped):
        return None
    if use_f16_accum or use_batch_invariant or not layer_config.num_experts:
        return None
    if get_problem_rejection_reasons(layer_config, MmaType.WGMMA):
        return None

    shape_n, shape_k = layer_config.shape_n, layer_config.shape_k
    if shape_n % 128 or shape_k % 128:
        return None

    # Separate output storage lets the stage ring continue across expert tiles.
    # H20 favors two resident CTAs; the full Hopper compute throughput benefits
    # from a wider tile with more accumulator registers per CTA.
    block_m, block_n, warp_n = (48, 128, 16) if is_h20 else (80, 256, 32)
    stages, resident_ctas = (3, 2) if is_h20 else (4, 1)
    use_warp_spec = True
    use_stream_k = False
    short_k = shape_k <= 512
    if short_k:
        block_m = 64 if is_h20 else 96
        stages = 3
    elif is_h20 and shape_n <= 1024 and shape_k >= 4096 and shape_n % 256 == 0:
        block_m, block_n, warp_n = 64, 256, 32
        resident_ctas = 1
        use_stream_k = True
    elif is_h20 and shape_k <= 2048:
        stages = 4

    # Narrow output widths cannot form the wider tile. Removing the producer
    # warpgroup makes two smaller CTAs fit without sacrificing register space.
    if shape_n % block_n:
        block_m, block_n, warp_n = 80, 128, 16
        stages, resident_ctas = 3, 2
        use_warp_spec = False

    # Use the same routing samples as the SM100 MoE policy. Round each expert
    # independently: empty experts, padding and incomplete waves all affect
    # whether these larger tiles can amortize their pipeline and output storage.
    counts = DeviceHeuristics._sample_expert_rows(shape_m, layer_config.num_experts, expert_probability_cv)
    resident_blocks = current_device.sm_count * resident_ctas
    block_m_candidates = (80, 96) if block_m == 96 else (block_m,)
    best = None
    for candidate_m in block_m_candidates:
        m_tiles = ((counts + candidate_m - 1) // candidate_m).sum(axis=1)
        tiles = m_tiles * (shape_n // block_n)
        utilization = counts.sum(axis=1) / (m_tiles * candidate_m)
        if utilization.mean() < 2 / 3 or tiles.mean() < resident_blocks * 2 / 3:
            continue
        waves = (tiles + resident_blocks - 1) // resident_blocks
        # As in SM100, balance tile rounds against padded math per tile.
        score = float(waves.mean() * math.sqrt(candidate_m))
        if best is None or score < best[0]:
            best = score, candidate_m
    if best is None:
        return None
    block_m = best[1]

    config = {
        "mma_type": MmaType.WGMMA.value,
        "block_shape": (block_m, block_n, 128),
        "warp_shape": (block_m, warp_n, 128),
        "num_stages": stages,
        "num_ctas_per_sm": resident_ctas,
        "use_stream_k": use_stream_k,
        "use_warp_spec": use_warp_spec,
        "use_tma": True,
        "use_tma_c": use_warp_spec and not short_k,
        "smem_reuse_mode": "none",
        "wgmma_split_issue_wait": not use_warp_spec,
        "raster_group_m": 1,
    }
    if short_k or use_stream_k:
        config["producer_stage_unroll"] = 2
        config["consumer_stage_unroll"] = 2 if short_k else 1

    smem_size = estimate_smem_size_layer(
        layer_config,
        config["block_shape"],
        gemm_type,
        stages,
        warp_shape=config["warp_shape"],
        smem_reuse_mode="none",
        use_tma=True,
        use_warp_spec=use_warp_spec,
    )
    if smem_size * resident_ctas > 227 * 1024:
        return None
    return config


@dataclasses.dataclass(frozen=True, slots=True)
class Sm90CandidatePolicy:
    max_indexed_threads_for_two_ctas: int = 256
    max_indexed_threads_for_three_ctas: int = 256


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class _IndexedOption:
    candidate: ScheduleCandidate
    transform: Literal["base", "half_k", "split_n_widen_k"]
    priority: int
    parent: "_IndexedOption | None" = None


def calc_sm90_num_block_list(
    layer_config: LayerConfig,
    shape_m: int,
    max_block_m: int,
) -> list[int]:
    num_blocks_list = []
    if not layer_config.num_experts:
        for block_m in range(8, max_block_m + 1, 8):
            num_blocks_list.append(math.ceil(shape_m / block_m))
    else:
        random_state = np.random.RandomState(seed=0)
        samples = random_state.randint(0, layer_config.num_experts, size=shape_m)
        counts = np.bincount(samples)
        for block_m in range(8, max_block_m + 1, 8):
            num_blocks = int(np.ceil(counts * 1.1 / block_m).sum().item())
            num_blocks_list.append(num_blocks)

    for index, block_m in enumerate(range(8, max_block_m + 1, 8)):
        if layer_config.a_dtype == dtypes.int8 and block_m % 16 == 8 and block_m > 32:
            num_blocks_list[index] = 1000000

    return num_blocks_list


def _select_sm90_block_m(
    layer_config: LayerConfig,
    shape_m: int,
    max_block_m: int,
) -> int:
    num_blocks_list = calc_sm90_num_block_list(
        layer_config,
        shape_m,
        max_block_m,
    )
    return np.argmin(num_blocks_list).item() * 8 + 8


def build_sm90_seed_config(problem: TuningProblem) -> dict:
    """Build the sparse seed config shared by legacy and indexed-A16 paths."""
    layer_config = problem.layer_config
    tune_indexed_a16 = (
        problem.gemm_type == GemmType.INDEXED
        and layer_config.a_dtype.num_bits == 16
        and not problem.use_batch_invariant
    )
    if layer_config.use_packed_k_layout:
        max_block_m = 128
    elif problem.use_f16_accum:
        max_block_m = 256
    else:
        max_block_m = 176

    if tune_indexed_a16:
        # Bound padding when only a few routed rows land on each expert.
        tokens_per_expert = problem.shape_m / layer_config.num_experts
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
        block_shape_m = _select_sm90_block_m(
            layer_config,
            problem.shape_m,
            max_block_m,
        )
    warp_shape_n = 32
    warp_shape_k = 1024 // layer_config.a_dtype.num_bits

    # Long-K layers need more routed rows before wider N tiles pay off.
    wide_tile_min_shape_m = 64 if layer_config.shape_k > 4096 else 16
    use_wide_indexed_tile = (
        tune_indexed_a16 and block_shape_m <= 64 and problem.shape_m >= wide_tile_min_shape_m
    )
    if use_wide_indexed_tile:
        warp_shape_n = 64
        # N=512 spills its accumulator at two-CTA residency from M=48 onward.
        if layer_config.shape_k <= 512 and layer_config.shape_n >= 2048 and block_shape_m < 48:
            block_shape_n = 512
            block_shape_k = 64
        else:
            block_shape_n = 256
            block_shape_k = 128
    elif layer_config.shape_n <= 4096 and not problem.use_batch_invariant and block_shape_m <= 64:
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

    if layer_config.use_packed_k_layout:
        warp_shape_k = 128
        block_shape_k = max(block_shape_k, warp_shape_k)

    if problem.gemm_type == GemmType.INDEXED and layer_config.use_packed_k_layout:
        while block_shape_n // warp_shape_n * (block_shape_k // warp_shape_k) > 8:
            block_shape_k //= 2

    # Extra K partitions inflate the producer and dequantization state without
    # enough reduction work to amortize them. Preserve the long-K schedules.
    needs_smaller_k_tile = layer_config.shape_k <= 2048 and block_shape_k > 64
    dense_small_wna16 = (
        problem.gemm_type == GemmType.DENSE
        and layer_config.a_dtype.num_bits == 16
        and layer_config.b_dtype.num_bits < 16
        and (layer_config.b_dtype.num_bits == 4 or needs_smaller_k_tile)
        and problem.shape_m <= 128
        and layer_config.shape_n % 128 == 0
        and layer_config.shape_k % 64 == 0
    )
    if dense_small_wna16:
        block_shape_n = 128
        block_shape_k = 64
        warp_shape_n = 32
        warp_shape_k = 64
    if problem.use_batch_invariant:
        # Keep one K partition and the same reduction tile for every batch size.
        batch_invariant_k = 128 if layer_config.use_packed_k_layout else 1024 // layer_config.a_dtype.num_bits
        block_shape_k = min(batch_invariant_k, layer_config.shape_k & -layer_config.shape_k)
        warp_shape_k = block_shape_k

    config = {
        "block_shape": (block_shape_m, block_shape_n, block_shape_k),
        "warp_shape": (block_shape_m, warp_shape_n, warp_shape_k),
        "use_stream_k": not problem.use_batch_invariant,
        "use_f16_accum": problem.use_f16_accum,
        "num_stages": 4,
    }

    if problem.gemm_type != GemmType.INDEXED:
        config["use_warp_spec"] = True
        config["use_tma"] = True
        if dense_small_wna16:
            config["num_ctas_per_sm"] = 2

        if (
            layer_config.shape_n % (block_shape_n * 2) == 0
            and problem.shape_m / block_shape_m >= 4
            and problem.gemm_type == GemmType.DENSE
        ):
            config["multi_cast_size_a"] = 2

    return config


def select_grouped_scale(
    problem: TuningProblem,
) -> TuningDecision:
    layer_config = problem.layer_config
    if problem.use_f16_accum:
        max_block_m = 256
    elif layer_config.input_scale_group_size > 0:
        max_block_m = 160
    elif layer_config.weight_scale_group_size < 128:
        max_block_m = 192
    else:
        max_block_m = 200
    block_shape_m = _select_sm90_block_m(
        layer_config,
        problem.shape_m,
        max_block_m,
    )
    if problem.use_batch_invariant:
        block_ks = (min(128, layer_config.shape_k & -layer_config.shape_k),)
    else:
        block_ks = (256, 128, 64) if block_shape_m <= 32 else (128, 64)
    use_multicast = problem.gemm_type == GemmType.DENSE and problem.shape_m / block_shape_m >= 4

    candidates = []
    # Candidate order records measured preference; legality supplies fallbacks.
    tile_shapes = ((128, 32), (128, 16), (64, 16)) if layer_config.use_raw_weight else ((128, 32), (64, 16))
    for block_shape_n, warp_shape_n in tile_shapes:
        for block_shape_k in block_ks:
            multicast_values = (True, False) if use_multicast else (False,)
            for multicast in multicast_values:
                config = {
                    "block_shape": (
                        block_shape_m,
                        block_shape_n,
                        block_shape_k,
                    ),
                    "warp_shape": (
                        block_shape_m,
                        warp_shape_n,
                        min(128, block_shape_k),
                    ),
                    "use_stream_k": not problem.use_batch_invariant,
                    "use_f16_accum": problem.use_f16_accum,
                    "num_stages": 4,
                }
                if problem.gemm_type != GemmType.INDEXED:
                    config["use_warp_spec"] = True
                    config["use_tma"] = True
                if multicast:
                    config["multi_cast_size_a"] = 2
                candidate = ScheduleCandidate.from_config(
                    "grouped_scale_"
                    f"n{block_shape_n}_wn{warp_shape_n}_k{block_shape_k}_"
                    f"{'multicast' if multicast else 'direct'}",
                    config,
                )
                candidates.append(fit_pipeline_stages(problem, candidate))

    analyses = tuple(analyze_candidate(problem, candidate) for candidate in candidates)
    selected = next(
        (analysis for analysis in analyses if analysis.legal),
        None,
    )
    if selected is None:
        rejected = {analysis.candidate.candidate_id: analysis.rejection_reasons for analysis in analyses}
        raise AssertionError(f"no legal grouped-scale SM90 schedule: {rejected}")

    return TuningDecision(
        problem=problem,
        family="grouped_scale",
        selected=selected.candidate,
        considered=analyses,
        reason="selected the first legal measured-priority candidate",
    )


def get_scaled_w8a8_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    use_m_major_input_scale: bool = False,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> dict | None:
    has_raw_w8a8 = layer_config.a_dtype.num_bits == 8 and layer_config.a_dtype == layer_config.b_dtype
    has_group_scales = layer_config.is_group_input_scale or layer_config.is_group_weight_scale
    has_group_scales |= layer_config.is_block_weight_scale
    if not has_raw_w8a8 or not has_group_scales:
        return None
    if gemm_type not in (GemmType.DENSE, GemmType.INDEXED) or use_f16_accum or use_batch_invariant:
        return None
    if layer_config.shape_n % 128 or layer_config.shape_k % 128:
        return None
    if get_problem_rejection_reasons(layer_config, MmaType.WGMMA):
        return None

    num_sms = current_device.sm_count
    alignment = 16 if layer_config.a_dtype.is_integer_type else 8
    short_k = layer_config.shape_k <= 512

    def make_config(block_m, block_n, warp_n, use_warp_spec, use_stream_k, stages=4, ctas=1, ring=True):
        use_ring = use_warp_spec and ring
        return {
            "mma_type": MmaType.WGMMA.value,
            "block_shape": (block_m, block_n, 128),
            "warp_shape": (block_m, warp_n, 128),
            "num_stages": stages,
            "num_ctas_per_sm": ctas,
            "use_tma": use_warp_spec,
            "use_warp_spec": use_warp_spec,
            "use_stream_k": bool(use_stream_k),
            "smem_reuse_mode": "none" if use_ring else "all_stages",
            "producer_stage_unroll": 2 if use_ring else None,
            "raster_group_m": 1,
        }

    def fits_resources(config):
        tuning = TuningConfig(**config)
        if get_register_budget_error(layer_config, tuning, registers_per_sm=65536):
            return False
        smem_size = estimate_smem_size_layer(
            layer_config,
            config["block_shape"],
            gemm_type,
            config["num_stages"],
            warp_shape=config["warp_shape"],
            smem_reuse_mode=config["smem_reuse_mode"],
            use_tma=config["use_tma"],
            use_warp_spec=config["use_warp_spec"],
            mma_type=MmaType.WGMMA,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        return smem_size * config["num_ctas_per_sm"] <= 227 * 1024

    if gemm_type == GemmType.DENSE:
        if short_k:
            return None
        n_tiles = layer_config.shape_n // 128
        has_wide_short_grid = layer_config.shape_k <= 1024 and n_tiles >= num_sms / 4 and shape_m <= 128
        if is_h20 and shape_m < 32:
            return None
        if has_wide_short_grid and shape_m >= 32:
            m_tiles = max(1, math.ceil(num_sms * 2 / 3 / n_tiles))
            block_m = min(64, math.ceil(shape_m / m_tiles / alignment) * alignment)
            config = make_config(block_m, 128, 16, True, False)
        elif has_wide_short_grid:
            block_m = int(shape_m * n_tiles / num_sms) // alignment * alignment
            block_m = min(64, max(16, block_m))
            config = make_config(block_m, 128, 16, False, False, stages=3, ctas=2)
        elif is_h20:
            if shape_m < 32 or (layer_config.shape_k >= 4096 and n_tiles < num_sms / 4):
                return None
            m_tiles = max(1, math.ceil(num_sms * 2 / 3 / n_tiles))
            block_m = min(64, math.ceil(shape_m / m_tiles / alignment) * alignment)
            if math.ceil(shape_m / block_m) * n_tiles < num_sms * 2 / 3:
                return None
            config = make_config(block_m, 128, 16, True, False)
        else:
            if shape_m <= 64:
                return None
            wide_tiles = math.ceil(shape_m / 64) * (layer_config.shape_n // 256)
            if layer_config.shape_n % 256 == 0 and wide_tiles >= num_sms * 2 / 3:
                config = make_config(64, 256, 32, True, False)
            else:
                config = make_config(64, 128, 32, True, True, ring=False)
        return config if fits_resources(config) else None

    counts = DeviceHeuristics._sample_expert_rows(shape_m, layer_config.num_experts, expert_probability_cv)
    if is_h20 and layer_config.shape_k < 4096:
        full_tiles = (counts // 32).sum(axis=1) * (layer_config.shape_n // 256)
        if short_k or full_tiles.mean() < num_sms:
            return None
    candidates = []
    for block_m in (8, 16, 24, 32, 48, 64, 80):
        if is_h20 and block_m > 32:
            continue
        if layer_config.a_dtype.is_integer_type and block_m > 32 and block_m % 16:
            continue
        m_tiles = ((counts + block_m - 1) // block_m).sum(axis=1)
        wide_tiles = m_tiles * (layer_config.shape_n // 256)
        use_wide_tile = layer_config.shape_n % 256 == 0 and wide_tiles.mean() >= num_sms * 2 / 3
        block_n = 256 if use_wide_tile else 128
        tiles = m_tiles * (layer_config.shape_n // block_n)
        use_stream_k = tiles.mean() < num_sms * 2 / 3 and layer_config.shape_k >= 2048
        config = make_config(block_m, block_n, block_n // 8, True, use_stream_k, stages=3 if short_k else 4)
        if fits_resources(config):
            score = float(np.ceil(tiles / num_sms).mean() * (block_m + 32) * block_n / 128)
            candidates.append((score, config))
    if not candidates:
        return None
    config = min(candidates, key=lambda item: item[0])[1]
    block_m, block_n, _ = config["block_shape"]
    if is_h20 and block_n == 256 and block_m <= 16:
        full_tiles = (counts // block_m).sum(axis=1) * (layer_config.shape_n // block_n)
        if full_tiles.mean() < num_sms:
            return None
    return config


def get_packed_wna8_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    use_m_major_input_scale: bool = False,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> dict | None:
    """Select packed weight tiles from routing work and SM90 resources."""
    if not layer_config.use_packed_k_layout:
        return None
    if is_h20 and gemm_type == GemmType.DENSE and not layer_config.use_fused_e8m0_scale:
        return None
    if use_f16_accum or use_batch_invariant:
        return None
    if layer_config.shape_n % 128 or layer_config.shape_k % 128:
        return None

    # Match the M sampling grid in DeviceHeuristics.get_configs so that direct
    # queries and precomputed ranges use the same routing sample at boundaries.
    if shape_m <= 8:
        shape_m = 1 << (max(shape_m, 1) - 1).bit_length()
    else:
        step = 8 if shape_m <= 1024 else 16 if shape_m <= 2048 else 32 if shape_m <= 4096 else 64
        if shape_m > 16384:
            step = 128
        shape_m = math.ceil(shape_m / step) * step

    if gemm_type == GemmType.DENSE:
        # Use one K warp to limit register pressure on small grids,
        # and fit M to the rows to reduce padding.
        block_m = math.ceil(shape_m / math.ceil(shape_m / 64))
        alignment = 16 if layer_config.a_dtype.is_integer_type else 8
        block_m = math.ceil(block_m / alignment) * alignment
        tiles = math.ceil(shape_m / block_m) * (layer_config.shape_n // 128)
        can_split_k = not is_h20 and shape_m >= 32 and layer_config.shape_k >= 2048
        if can_split_k and tiles < current_device.sm_count * 2 / 3:
            return {
                "mma_type": MmaType.WGMMA.value,
                "block_shape": (block_m, 128, 128),
                "warp_shape": (block_m, 16, 128),
                "num_stages": 4,
                "num_ctas_per_sm": 1,
                "use_tma": True,
                "use_warp_spec": True,
                "use_stream_k": True,
                "wgmma_split_issue_wait": True,
            }
        if shape_m <= 64:
            return None

    if gemm_type == GemmType.DENSE:
        counts = np.asarray([[shape_m]])
    elif gemm_type == GemmType.GROUPED_MASKED:
        # Here M describes allocated expert capacity, not a routed token count.
        counts = np.full((1, layer_config.num_experts), math.ceil(shape_m / layer_config.num_experts))
    else:
        counts = DeviceHeuristics._sample_expert_rows(
            shape_m, layer_config.num_experts, expert_probability_cv
        )

    has_indexed_weight_groups = (
        gemm_type == GemmType.INDEXED
        and layer_config.is_group_weight_scale
        and not layer_config.use_fused_e8m0_scale
    )
    max_block_m = 64 if is_h20 else 160
    num_sms = current_device.sm_count
    best_config = None
    best_score = math.inf
    large_configs = []

    def estimate_work(config, expert_rows):
        block_m, block_n, _ = config["block_shape"]
        resident_ctas = config["num_ctas_per_sm"]
        m_tiles = ((expert_rows + block_m - 1) // block_m).sum(axis=1)
        tiles = m_tiles * (layer_config.shape_n // block_n)
        waves = np.ceil(tiles / (num_sms * resident_ctas))
        return float(waves.mean() * resident_ctas * (block_m + 32) * block_n / 128)

    for block_m in range(8, max_block_m + 1, 8):
        if layer_config.a_dtype.is_integer_type and block_m % 16:
            continue
        is_small_tile = block_m <= 32
        block_n = 256 if is_small_tile and layer_config.shape_n % 256 == 0 else 128
        warp_n = 32 if block_n == 256 else 16
        can_use_warp_spec = gemm_type != GemmType.INDEXED or has_indexed_weight_groups
        use_warp_spec = not is_h20 and not is_small_tile and can_use_warp_spec
        resident_ctas = 1 if use_warp_spec else 2
        stages = 3 if is_small_tile else (5 if is_h20 else 4)
        config = {
            "mma_type": MmaType.WGMMA.value,
            "block_shape": (block_m, block_n, 128),
            "warp_shape": (block_m, warp_n, 128),
            "num_stages": stages,
            "num_ctas_per_sm": resident_ctas,
            "use_tma": use_warp_spec,
            "use_warp_spec": use_warp_spec,
            "use_stream_k": False,
            "wgmma_use_late_as": layer_config.is_group_input_scale,
            "wgmma_split_issue_wait": not (use_warp_spec and has_indexed_weight_groups),
            "smem_reuse_mode": "none" if use_warp_spec else "all_stages",
            "raster_group_m": 1,
        }
        if use_warp_spec:
            config["producer_stage_unroll"] = 2
        tuning = TuningConfig(**config)
        if get_register_budget_error(layer_config, tuning, registers_per_sm=65536):
            continue
        smem_size = estimate_smem_size_layer(
            layer_config,
            config["block_shape"],
            gemm_type,
            stages,
            warp_shape=config["warp_shape"],
            smem_reuse_mode=config["smem_reuse_mode"],
            use_tma=use_warp_spec,
            use_warp_spec=use_warp_spec,
            mma_type=MmaType.WGMMA,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        if smem_size * resident_ctas > 227 * 1024:
            continue

        # Balance padded math against repeated weight loading and dequantization.
        # Round each sampled expert before accounting for incomplete waves.
        score = estimate_work(config, counts)
        if block_m >= 128:
            large_configs.append((config, score))
        if score < best_score:
            best_score = score
            best_config = config
    has_routed_rows = gemm_type in (GemmType.GROUPED_CONTIGUOUS, GemmType.INDEXED)
    if has_routed_rows and best_config is not None and best_config["block_shape"][0] > 128:
        # Prefer narrower RS tiles when nominal work is close and skewed work improves.
        # Rank by nominal work to preserve gains near tile boundaries.
        skewed_rows = DeviceHeuristics._sample_expert_rows(
            shape_m, layer_config.num_experts, max(1.0, expert_probability_cv)
        )
        skewed_work = estimate_work(best_config, skewed_rows)
        safer_configs = [
            (config, score)
            for config, score in large_configs
            if config["block_shape"][0] < best_config["block_shape"][0]
            and score <= best_score * 1.1
            and estimate_work(config, skewed_rows) < skewed_work
        ]
        if safer_configs:
            best_config = min(safer_configs, key=lambda item: item[1])[0]

    if best_config is not None:
        block_m, block_n, _ = best_config["block_shape"]
        # Preserve the existing Stream-K decode schedules when little work can
        # amortize a full packed stage or the indexed row-gathering setup.
        if gemm_type == GemmType.INDEXED and block_m < 32:
            return None
        if is_h20 and block_m == 8 and block_n == 256:
            return None
        if gemm_type == GemmType.DENSE or not layer_config.use_fused_e8m0_scale:
            m_tiles = ((counts + block_m - 1) // block_m).sum(axis=1)
            tiles = m_tiles * (layer_config.shape_n // block_n)
            resident_ctas = 1 if layer_config.use_fused_e8m0_scale else best_config["num_ctas_per_sm"]
            if tiles.mean() < num_sms * resident_ctas / 2:
                return None
    return best_config


def _indexed_a16_ctas_per_sm(
    problem: TuningProblem,
    analysis: CandidateAnalysis,
    policy: Sm90CandidatePolicy,
) -> int:
    # Thread caps stand in for the measured register launch-bound cliffs.
    resource_limit = 1
    if (
        analysis.num_threads <= policy.max_indexed_threads_for_two_ctas
        and analysis.smem_size * 2 <= problem.device.resident_smem_size
    ):
        resource_limit = 2
    block_shape = analysis.candidate.block_shape
    if (
        problem.layer_config.a_dtype.num_bits == 16
        and problem.layer_config.b_dtype.num_bits == 4
        and block_shape[0] == 8
        and analysis.num_threads <= policy.max_indexed_threads_for_three_ctas
        and analysis.smem_size * 3 <= problem.device.resident_smem_size
    ):
        resource_limit = 3

    resource_limit = min(resource_limit, analysis.thread_smem_cta_limit)
    if problem.device.num_sms is None:
        raise ValueError("indexed-A16 selection requires a device SM count")
    grid_limit = math.ceil(analysis.num_output_tiles / problem.device.num_sms)
    return max(1, min(resource_limit, grid_limit))


def _analyze_indexed_a16_candidate(
    problem: TuningProblem,
    candidate: ScheduleCandidate,
    policy: Sm90CandidatePolicy,
) -> CandidateAnalysis:
    analysis = analyze_candidate(problem, candidate)
    if not analysis.legal:
        return analysis
    candidate = candidate.with_updates(
        num_ctas_per_sm=_indexed_a16_ctas_per_sm(
            problem,
            analysis,
            policy,
        )
    )
    return analyze_candidate(problem, candidate)


def _half_k_candidate(
    problem: TuningProblem,
    source: ScheduleCandidate,
) -> ScheduleCandidate | None:
    block_shape = source.block_shape
    warp_shape = source.warp_shape
    smaller_block_k = block_shape[2] // 2
    scale_groups_align = all(
        not group_size or group_size % smaller_block_k == 0 or smaller_block_k % group_size == 0
        for group_size in (
            problem.layer_config.input_scale_group_size,
            problem.layer_config.weight_scale_group_size,
        )
    )
    if block_shape[2] < warp_shape[2] * 2 or not scale_groups_align:
        return None

    config = source.to_config()
    config.pop("num_ctas_per_sm", None)
    return ScheduleCandidate.from_config(
        "indexed_a16_half_k",
        config,
    ).with_updates(
        block_shape=(*block_shape[:2], smaller_block_k),
        warp_shape=(
            *warp_shape[:2],
            min(warp_shape[2], smaller_block_k),
        ),
    )


def _split_n_widen_k_candidate(
    problem: TuningProblem,
    source: ScheduleCandidate,
    *,
    candidate_id: str,
) -> ScheduleCandidate | None:
    block_shape = source.block_shape
    if not (
        problem.layer_config.a_dtype.num_bits == 16
        and problem.layer_config.b_dtype.num_bits == 4
        and block_shape[1] >= 256
        and block_shape[2] == 64
    ):
        return None

    config = source.to_config()
    config.pop("num_ctas_per_sm", None)
    return ScheduleCandidate.from_config(candidate_id, config).with_updates(
        block_shape=(
            block_shape[0],
            block_shape[1] // 2,
            block_shape[2] * 2,
        ),
    )


def select_indexed_a16(
    problem: TuningProblem,
    policy: Sm90CandidatePolicy,
) -> TuningDecision:
    if problem.device.num_sms is None:
        raise ValueError("indexed-A16 selection requires a device SM count")
    base = fit_pipeline_stages(
        problem,
        ScheduleCandidate.from_config(
            "indexed_a16_base",
            build_sm90_seed_config(problem),
        ),
    )
    base_option = _IndexedOption(base, "base", 0)
    options = [base_option]
    half_k = _half_k_candidate(problem, base)
    half_option = None
    if half_k is not None:
        half_option = _IndexedOption(half_k, "half_k", 1, base_option)
        options.append(half_option)
    for source, candidate_id in (
        (base_option, "indexed_a16_split_n_widen_k_from_base"),
        (half_option, "indexed_a16_split_n_widen_k"),
    ):
        if source is None:
            continue
        split = _split_n_widen_k_candidate(
            problem,
            source.candidate,
            candidate_id=candidate_id,
        )
        if split is not None:
            options.append(_IndexedOption(split, "split_n_widen_k", 2, source))

    analyses = {
        option: _analyze_indexed_a16_candidate(
            problem,
            option.candidate,
            policy,
        )
        for option in options
    }
    base_analysis = analyses[base_option]
    if not base_analysis.legal:
        raise AssertionError(base_analysis.rejection_reasons)
    eligible_reasons = {
        base_option: "selected the base indexed-A16 schedule",
    }

    if half_option is not None:
        half_analysis = analyses[half_option]
        if (
            base_analysis.candidate.num_ctas_per_sm == 1
            and half_analysis.legal
            and half_analysis.candidate.num_ctas_per_sm > base_analysis.candidate.num_ctas_per_sm
        ):
            eligible_reasons[half_option] = "halved K because it increased CTA residency"

    for option in options:
        if option.transform != "split_n_widen_k":
            continue
        assert option.parent is not None
        parent_reason = eligible_reasons.get(option.parent)
        analysis = analyses[option]
        parent_analysis = analyses[option.parent]
        if (
            parent_reason is not None
            and analysis.legal
            and analysis.candidate.num_ctas_per_sm > parent_analysis.candidate.num_ctas_per_sm
            and analysis.waves is not None
            and parent_analysis.waves is not None
            and analysis.waves <= parent_analysis.waves
        ):
            eligible_reasons[option] = f"{parent_reason}; split N and widened K without adding a grid wave"
    selected = max(
        eligible_reasons,
        key=lambda option: option.priority,
    )

    final_candidate = analyses[selected].candidate.with_updates(
        use_stream_k=(
            bool(analyses[selected].candidate.get("use_stream_k", True))
            and analyses[selected].num_output_tiles < problem.device.num_sms
        )
    )
    final_analysis = analyze_candidate(problem, final_candidate)
    if not final_analysis.legal:
        raise AssertionError(final_analysis.rejection_reasons)
    return TuningDecision(
        problem=problem,
        family="indexed_a16",
        selected=final_candidate,
        considered=tuple(final_analysis if option is selected else analyses[option] for option in options),
        reason=eligible_reasons[selected],
    )
