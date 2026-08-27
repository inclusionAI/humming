import dataclasses
import math
from typing import Literal

import numpy as np

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.tune.candidate import (
    CandidateAnalysis,
    ScheduleCandidate,
    TuningDecision,
    TuningProblem,
    analyze_candidate,
    fit_pipeline_stages,
)


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
        tune_indexed_a16
        and block_shape_m <= 64
        and problem.shape_m >= wide_tile_min_shape_m
    )
    if use_wide_indexed_tile:
        warp_shape_n = 64
        # N=512 spills its accumulator at two-CTA residency from M=48 onward.
        if (
            layer_config.shape_k <= 512
            and layer_config.shape_n >= 2048
            and block_shape_m < 48
        ):
            block_shape_n = 512
            block_shape_k = 64
        else:
            block_shape_n = 256
            block_shape_k = 128
    elif (
        layer_config.shape_n <= 4096
        and not problem.use_batch_invariant
        and block_shape_m <= 64
    ):
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
        problem.gemm_type == GemmType.DENSE
        and layer_config.a_dtype.num_bits == 16
        and layer_config.b_dtype.num_bits == 4
        and problem.shape_m <= 128
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
        "use_stream_k": not problem.use_batch_invariant,
        "use_f16_accum": problem.use_f16_accum,
        "num_stages": 4,
    }

    if problem.gemm_type != GemmType.INDEXED:
        config["use_warp_spec"] = True
        config["use_tma"] = True
        config["use_mbarrier"] = True
        if dense_small_fp4:
            config["num_ctas_per_sm"] = 2

        if (
            layer_config.shape_n % (block_shape_n * 2) == 0
            and problem.shape_m / block_shape_m >= 4
            and problem.gemm_type == GemmType.DENSE
        ):
            config["multi_cast_size_a"] = 2

    return config


# --- MXFP4 weights dequantised into 8-bit MMA inputs (fused e8m0 scales) ---

# Measured on SM90 (H200, GLM-5.2 MoE shapes): this family is bound by the
# accumulator register footprint of the math warps, so every schedule keeps
# exactly eight of them (two WGMMA warpgroups). One warpgroup owning the whole
# tile, or four warpgroups splitting it, both lose 2-4x.
_FP4_MATH_WARPS = 8
# An M block re-reads the whole expert weight slab, so while an expert still
# fits in one block the tile is sized to hold the *largest* expert rather than
# the average one: undersizing costs 25-35%, oversizing only ~5%. Past the
# point where no legal tile can hold an expert, padding is unavoidable and the
# tile settles on its own throughput optimum instead.
_FP4_LARGE_BLOCK_M = 72
_FP4_MAX_SMALL_BLOCK_M = 80
_FP4_MIN_BLOCK_M = 16
# Standard deviations of routing imbalance the tile is sized to absorb.
_FP4_IMBALANCE_SIGMAS = 2.5
_FP4_NUM_STAGES = 4


def uses_fused_e8m0_fp4(layer_config: LayerConfig) -> bool:
    """Whether FP4 weights are dequantised into 8-bit MMA inputs via e8m0."""
    return (
        bool(layer_config.use_fused_e8m0_scale)
        and layer_config.b_dtype.num_bits == 4
        and layer_config.a_dtype.num_bits == 8
        and layer_config.input_scale_group_size > 0
    )


def _fused_e8m0_fp4_block_m(problem: TuningProblem) -> int:
    layer_config = problem.layer_config
    if problem.gemm_type == GemmType.DENSE or not layer_config.num_experts:
        # No routing padding to absorb, so the tile only has to avoid
        # overshooting the rows that exist.
        return min(_FP4_LARGE_BLOCK_M, max(8, math.ceil(problem.shape_m / 8) * 8))

    tokens_per_expert = problem.shape_m / layer_config.num_experts
    if tokens_per_expert >= _FP4_MAX_SMALL_BLOCK_M:
        return _FP4_LARGE_BLOCK_M

    # Multinomial routing spreads expert counts by about sqrt(tokens_per_expert).
    busiest_expert = tokens_per_expert + _FP4_IMBALANCE_SIGMAS * math.sqrt(
        tokens_per_expert
    )
    block_shape_m = math.ceil(busiest_expert / 8) * 8
    return min(max(block_shape_m, _FP4_MIN_BLOCK_M), _FP4_MAX_SMALL_BLOCK_M)


def _fused_e8m0_fp4_num_blocks_m(problem: TuningProblem, block_shape_m: int) -> int:
    num_blocks_m = math.ceil(problem.shape_m / block_shape_m)
    if problem.gemm_type == GemmType.DENSE or not problem.layer_config.num_experts:
        return num_blocks_m
    # Every routed expert takes at least one padded M block of its own.
    return max(problem.layer_config.num_experts, num_blocks_m)


def select_fused_e8m0_fp4(problem: TuningProblem) -> TuningDecision:
    if problem.device.num_sms is None:
        raise ValueError("fused-e8m0 FP4 selection requires a device SM count")
    preferred_block_m = _fused_e8m0_fp4_block_m(problem)

    candidates = []
    # Candidate order records measured preference; legality supplies fallbacks.
    for block_shape_n, warp_shape_n in ((256, 32), (128, 32), (64, 16)):
        for block_shape_k in (128, 64):
            warp_shape_k = min(128, block_shape_k)
            num_warps_nk = (block_shape_n // warp_shape_n) * (
                block_shape_k // warp_shape_k
            )
            num_warps_m = max(1, _FP4_MATH_WARPS // num_warps_nk)
            # Round the tile up until warp_m is a whole number of eight rows.
            m_granularity = num_warps_m * 8
            block_shape_m = (
                math.ceil(preferred_block_m / m_granularity) * m_granularity
            )
            num_output_tiles = (
                problem.layer_config.shape_n
                // block_shape_n
                * _fused_e8m0_fp4_num_blocks_m(problem, block_shape_m)
            )
            config = {
                "block_shape": (block_shape_m, block_shape_n, block_shape_k),
                "warp_shape": (
                    block_shape_m // num_warps_m,
                    warp_shape_n,
                    warp_shape_k,
                ),
                # Stream-K only pays for itself while the grid cannot fill the
                # device; past that its reduction pass is measured overhead.
                "use_stream_k": (
                    not problem.use_batch_invariant
                    and num_output_tiles < problem.device.num_sms
                ),
                "use_f16_accum": problem.use_f16_accum,
                "num_stages": _FP4_NUM_STAGES,
            }
            if problem.gemm_type != GemmType.INDEXED:
                config["use_warp_spec"] = True
                config["use_tma"] = True
                config["use_mbarrier"] = True
            candidate = ScheduleCandidate.from_config(
                f"fused_e8m0_fp4_n{block_shape_n}_k{block_shape_k}",
                config,
            )
            candidates.append(fit_pipeline_stages(problem, candidate))

    analyses = tuple(analyze_candidate(problem, candidate) for candidate in candidates)
    selected = next(
        (analysis for analysis in analyses if analysis.legal),
        None,
    )
    if selected is None:
        rejected = {
            analysis.candidate.candidate_id: analysis.rejection_reasons
            for analysis in analyses
        }
        raise AssertionError(f"no legal fused-e8m0 FP4 SM90 schedule: {rejected}")

    return TuningDecision(
        problem=problem,
        family="fused_e8m0_fp4",
        selected=selected.candidate,
        considered=analyses,
        reason="selected the first legal measured-priority candidate",
    )


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
    block_ks = (256, 128, 64) if block_shape_m <= 32 else (128, 64)
    use_multicast = (
        problem.gemm_type == GemmType.DENSE and problem.shape_m / block_shape_m >= 4
    )

    candidates = []
    # Candidate order records measured preference; legality supplies fallbacks.
    for block_shape_n, warp_shape_n in ((128, 32), (64, 16)):
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
                    config["use_mbarrier"] = True
                if multicast:
                    config["multi_cast_size_a"] = 2
                candidate = ScheduleCandidate.from_config(
                    "grouped_scale_"
                    f"n{block_shape_n}_k{block_shape_k}_"
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
        rejected = {
            analysis.candidate.candidate_id: analysis.rejection_reasons
            for analysis in analyses
        }
        raise AssertionError(f"no legal grouped-scale SM90 schedule: {rejected}")

    return TuningDecision(
        problem=problem,
        family="grouped_scale",
        selected=selected.candidate,
        considered=analyses,
        reason="selected the first legal measured-priority candidate",
    )


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
        not group_size
        or group_size % smaller_block_k == 0
        or smaller_block_k % group_size == 0
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
            and half_analysis.candidate.num_ctas_per_sm
            > base_analysis.candidate.num_ctas_per_sm
        ):
            eligible_reasons[half_option] = (
                "halved K because it increased CTA residency"
            )

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
            and analysis.candidate.num_ctas_per_sm
            > parent_analysis.candidate.num_ctas_per_sm
            and analysis.waves is not None
            and parent_analysis.waves is not None
            and analysis.waves <= parent_analysis.waves
        ):
            eligible_reasons[option] = (
                f"{parent_reason}; split N and widened K without adding a grid wave"
            )
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
        considered=tuple(
            final_analysis if option is selected else analyses[option]
            for option in options
        ),
        reason=eligible_reasons[selected],
    )
