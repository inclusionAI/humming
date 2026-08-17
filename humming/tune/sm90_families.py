import dataclasses
import math
from typing import Literal

from humming.config import GemmType
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


@dataclasses.dataclass(frozen=True, slots=True)
class _IndexedOption:
    candidate: ScheduleCandidate
    transform: Literal["base", "half_k", "split_n_widen_k"]
    priority: int
    parent: "_IndexedOption | None" = None


@dataclasses.dataclass(slots=True)
class _IndexedEvaluation:
    option: _IndexedOption
    analysis: CandidateAnalysis
    reason: str | None = None

    @property
    def eligible(self) -> bool:
        return self.reason is not None


def _evaluation_for(
    evaluations: list[_IndexedEvaluation],
    option: _IndexedOption,
) -> _IndexedEvaluation:
    return next(evaluation for evaluation in evaluations if evaluation.option is option)


def select_grouped_scale(
    problem: TuningProblem,
    block_shape_m: int,
) -> TuningDecision:
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
    config: dict,
    policy: Sm90CandidatePolicy,
) -> TuningDecision:
    if problem.device.num_sms is None:
        raise ValueError("indexed-A16 selection requires a device SM count")
    base = fit_pipeline_stages(
        problem,
        ScheduleCandidate.from_config("indexed_a16_base", config),
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

    evaluations = [
        _IndexedEvaluation(
            option,
            _analyze_indexed_a16_candidate(problem, option.candidate, policy),
        )
        for option in options
    ]
    base_evaluation = _evaluation_for(evaluations, base_option)
    if not base_evaluation.analysis.legal:
        raise AssertionError(base_evaluation.analysis.rejection_reasons)
    base_evaluation.reason = "selected the base indexed-A16 schedule"

    if half_option is not None:
        half_evaluation = _evaluation_for(evaluations, half_option)
        if (
            base_evaluation.analysis.candidate.num_ctas_per_sm == 1
            and half_evaluation.analysis.legal
            and half_evaluation.analysis.candidate.num_ctas_per_sm
            > base_evaluation.analysis.candidate.num_ctas_per_sm
        ):
            half_evaluation.reason = "halved K because it increased CTA residency"

    for evaluation in evaluations:
        option = evaluation.option
        if option.transform != "split_n_widen_k":
            continue
        assert option.parent is not None
        parent = _evaluation_for(evaluations, option.parent)
        parent_reason = parent.reason
        if (
            parent_reason is not None
            and evaluation.analysis.legal
            and evaluation.analysis.candidate.num_ctas_per_sm
            > parent.analysis.candidate.num_ctas_per_sm
            and evaluation.analysis.waves is not None
            and parent.analysis.waves is not None
            and evaluation.analysis.waves <= parent.analysis.waves
        ):
            evaluation.reason = (
                f"{parent_reason}; split N and widened K without adding a grid wave"
            )
    selected = max(
        (evaluation for evaluation in evaluations if evaluation.eligible),
        key=lambda evaluation: evaluation.option.priority,
    )

    final_candidate = selected.analysis.candidate.with_updates(
        use_stream_k=(
            bool(selected.analysis.candidate.get("use_stream_k", True))
            and selected.analysis.num_output_tiles < problem.device.num_sms
        )
    )
    final_analysis = analyze_candidate(problem, final_candidate)
    if not final_analysis.legal:
        raise AssertionError(final_analysis.rejection_reasons)
    selected.analysis = final_analysis
    assert selected.reason is not None
    return TuningDecision(
        problem=problem,
        family="indexed_a16",
        selected=final_candidate,
        considered=tuple(evaluation.analysis for evaluation in evaluations),
        reason=selected.reason,
    )
