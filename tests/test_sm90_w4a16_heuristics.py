"""CPU-only coverage for the SM90 W4A16 indexed policy."""

import dataclasses

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.tune.candidate import (
    DeviceProfile,
    ScheduleCandidate,
    TuningProblem,
    analyze_candidate,
    estimate_indexed_m_blocks_uniform,
)
from humming.tune.sm90 import Sm90Heuristics
from humming.tune.sm90_policies import (
    Sm90CandidatePolicy,
    _select_conservative_indexed_a16_block_m,
    _select_w4a16_indexed_block_m,
    _applies_w4a16_indexed_policy,
    build_sm90_seed_config,
    select_indexed_a16,
)


@pytest.fixture(autouse=True)
def _mock_sm_count(monkeypatch):
    monkeypatch.setattr(
        Sm90Heuristics,
        "get_num_sms",
        classmethod(lambda cls: 132),
    )


def _layer(
    *,
    a_dtype=dtypes.bfloat16,
    b_dtype=dtypes.float4e2m1,
    num_experts: int = 256,
) -> LayerConfig:
    return LayerConfig(
        shape_n=1024,
        shape_k=4096,
        num_experts=num_experts,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=32,
        mma_type=MmaType.WGMMA,
    )


def _problem(
    *,
    shape_m: int = 4096,
    layer: LayerConfig | None = None,
    gemm_type: GemmType = GemmType.INDEXED,
    sm_version: int = 90,
    use_batch_invariant: bool = False,
    use_f16_accum: bool = False,
) -> TuningProblem:
    return TuningProblem(
        layer_config=layer or _layer(),
        shape_m=shape_m,
        gemm_type=gemm_type,
        device=DeviceProfile(
            name=f"sm{sm_version}",
            sm_version=sm_version,
            num_sms=132,
            max_smem_size=227 * 1024,
        ),
        use_batch_invariant=use_batch_invariant,
        use_f16_accum=use_f16_accum,
    )


def _analysis(decision, candidate_id):
    return next(
        analysis
        for analysis in decision.considered
        if analysis.candidate.candidate_id == candidate_id
    )


def test_w4a16_indexed_policy_scope_is_strict():
    assert _applies_w4a16_indexed_policy(_problem())
    assert not _applies_w4a16_indexed_policy(
        _problem(gemm_type=GemmType.DENSE)
    )
    assert not _applies_w4a16_indexed_policy(
        _problem(gemm_type=GemmType.GROUPED_CONTIGUOUS)
    )
    assert not _applies_w4a16_indexed_policy(
        _problem(use_batch_invariant=True)
    )
    assert not _applies_w4a16_indexed_policy(_problem(sm_version=80))
    assert not _applies_w4a16_indexed_policy(
        _problem(layer=_layer(a_dtype=dtypes.float8e4m3))
    )
    assert not _applies_w4a16_indexed_policy(
        _problem(layer=_layer(b_dtype=dtypes.float8e4m3))
    )
    assert not _applies_w4a16_indexed_policy(
        _problem(layer=_layer(num_experts=0))
    )


@pytest.mark.parametrize(
    ("block_m", "expected_m_blocks"),
    [(8, 512), (16, 256), (32, 256), (64, 256)],
)
def test_uniform_indexed_m_block_estimate(block_m, expected_m_blocks):
    assert estimate_indexed_m_blocks_uniform(4096, 256, block_m) == expected_m_blocks


def test_uniform_indexed_m_block_estimate_handles_remainder_experts():
    assert estimate_indexed_m_blocks_uniform(8, 3, 2) == 5


@pytest.mark.parametrize(
    ("routed_m", "expected_bm"),
    [
        (2048, 8),
        (2056, 8),
        (2872, 16),
        (4104, 16),
        (6152, 24),
        (8200, 32),
    ],
)
def test_w4a16_block_m_transition_boundaries(routed_m, expected_bm):
    assert _select_w4a16_indexed_block_m(
        _problem(shape_m=routed_m),
        max_block_m=64,
    ) == expected_bm


def test_seed_config_changes_only_w4a16_indexed_block_m():
    problem = _problem(shape_m=4096)
    seed = build_sm90_seed_config(problem)

    assert _select_conservative_indexed_a16_block_m(problem) == 24
    assert seed["block_shape"] == (16, 256, 128)
    assert seed["warp_shape"] == (16, 64, 64)
    assert seed["num_stages"] == 4
    assert seed["use_stream_k"] is True


def test_routed_m_aware_selector_is_capped_at_bm64():
    default_seed = build_sm90_seed_config(_problem(shape_m=45056))
    f16_accum_seed = build_sm90_seed_config(
        _problem(shape_m=49152, use_f16_accum=True)
    )

    assert default_seed["block_shape"][0] == 64
    assert f16_accum_seed["block_shape"][0] == 64


def test_w4a16_analysis_uses_routed_m_tile_count():
    problem = _problem(shape_m=4100)

    def candidate(block_m):
        return ScheduleCandidate.from_config(
            f"bm{block_m}",
            {
                "block_shape": (block_m, 256, 128),
                "warp_shape": (block_m, 64, 64),
                "num_stages": 4,
                "num_ctas_per_sm": 1,
            },
        )

    bm8 = analyze_candidate(problem, candidate(8))
    bm16 = analyze_candidate(problem, candidate(16))

    assert bm8.legal
    assert bm16.legal
    assert bm8.num_output_tiles == 2064
    assert bm16.num_output_tiles == 1040
    assert bm8.waves == 16
    assert bm16.waves == 8


@pytest.mark.parametrize(
    ("routed_m", "expected_bm"),
    [
        (2056, 8),
        (8192, 32),
    ],
)
def test_routed_m_aware_block_m_flows_through_production(
    routed_m,
    expected_bm,
):
    decision = Sm90Heuristics.get_tuning_decision(
        _layer(),
        shape_m=routed_m,
        gemm_type=GemmType.INDEXED,
    )

    assert decision.family == "indexed_a16"
    assert decision.selected_analysis.legal
    assert decision.selected.block_shape[0] == expected_bm


def test_bm_below_32_keeps_conservative_half_k_policy():
    decision = Sm90Heuristics.get_tuning_decision(
        _layer(),
        shape_m=2048,
        gemm_type=GemmType.INDEXED,
    )

    base = _analysis(decision, "indexed_a16_base")
    half_k = _analysis(decision, "indexed_a16_half_k")
    assert base.candidate.block_shape == (8, 256, 128)
    assert half_k.candidate.block_shape == (8, 256, 64)
    assert base.candidate.num_ctas_per_sm == 2
    assert half_k.candidate.num_ctas_per_sm == 3
    assert decision.selected == base.candidate
    assert decision.reason == "selected the base indexed-A16 schedule"


def test_bm_below_32_uses_half_k_for_conservative_residency_gain():
    problem = _problem(shape_m=128)
    problem = dataclasses.replace(
        problem,
        device=dataclasses.replace(
            problem.device,
            max_smem_per_sm=100_000,
        ),
    )

    decision = select_indexed_a16(problem, Sm90CandidatePolicy())

    base = _analysis(decision, "indexed_a16_base")
    half_k = _analysis(decision, "indexed_a16_half_k")
    assert base.candidate.block_shape[0] == half_k.candidate.block_shape[0] == 8
    assert base.candidate.num_ctas_per_sm == 1
    assert half_k.candidate.num_ctas_per_sm == 2
    assert decision.selected == half_k.candidate
    assert decision.reason == "halved K because it increased CTA residency"


def test_bm_at_least_32_prefers_legal_half_k():
    decision = Sm90Heuristics.get_tuning_decision(
        _layer(),
        shape_m=8192,
        gemm_type=GemmType.INDEXED,
    )

    base = _analysis(decision, "indexed_a16_base")
    half_k = _analysis(decision, "indexed_a16_half_k")
    assert base.candidate.block_shape == (32, 256, 128)
    assert half_k.candidate.block_shape == (32, 256, 64)
    assert half_k.legal
    assert decision.selected == half_k.candidate
    assert decision.reason == "halved K because BM is at least 32"


def test_bm_at_least_32_half_k_does_not_require_residency_gain(monkeypatch):
    import humming.tune.sm90_policies as policies

    original_analyze = policies._analyze_indexed_a16_candidate

    def force_half_k_lower_residency(problem, candidate, policy):
        analysis = original_analyze(problem, candidate, policy)
        if candidate.candidate_id != "indexed_a16_half_k":
            return analysis
        forced_candidate = analysis.candidate.with_updates(num_ctas_per_sm=1)
        return analyze_candidate(problem, forced_candidate)

    monkeypatch.setattr(
        policies,
        "_analyze_indexed_a16_candidate",
        force_half_k_lower_residency,
    )
    decision = select_indexed_a16(
        _problem(shape_m=8192),
        Sm90CandidatePolicy(),
    )

    base = _analysis(decision, "indexed_a16_base")
    half_k = _analysis(decision, "indexed_a16_half_k")
    assert base.candidate.num_ctas_per_sm == 2
    assert half_k.candidate.num_ctas_per_sm == 1
    assert half_k.legal
    assert decision.selected == half_k.candidate


def test_illegal_half_k_falls_back_to_base(monkeypatch):
    import humming.tune.sm90_policies as policies

    original_analyze = policies._analyze_indexed_a16_candidate

    def reject_half_k(problem, candidate, policy):
        analysis = original_analyze(problem, candidate, policy)
        if candidate.candidate_id == "indexed_a16_half_k":
            return dataclasses.replace(
                analysis,
                hard_violations=("synthetic half-K rejection",),
            )
        return analysis

    monkeypatch.setattr(
        policies,
        "_analyze_indexed_a16_candidate",
        reject_half_k,
    )
    decision = select_indexed_a16(
        _problem(shape_m=8192),
        Sm90CandidatePolicy(),
    )

    assert not _analysis(decision, "indexed_a16_half_k").legal
    assert decision.selected.candidate_id == "indexed_a16_base"
    assert decision.reason == "selected the base indexed-A16 schedule"


def test_non_w4_indexed_a16_keeps_conservative_block_m():
    problem = _problem(
        shape_m=2048,
        layer=_layer(b_dtype=dtypes.float8e4m3),
    )
    seed = build_sm90_seed_config(problem)

    assert not _applies_w4a16_indexed_policy(problem)
    assert _select_conservative_indexed_a16_block_m(problem) == 16
    assert seed["block_shape"][0] == 16


@pytest.mark.parametrize(
    "problem",
    [
        _problem(layer=_layer(a_dtype=dtypes.float8e4m3)),
        _problem(sm_version=80),
        _problem(use_batch_invariant=True),
    ],
    ids=["a8-input", "non-sm90", "batch-invariant"],
)
def test_non_target_indexed_analysis_keeps_legacy_m_tile_estimate(problem):
    candidate = ScheduleCandidate.from_config(
        "non_target",
        {
            "block_shape": (8, 256, 128),
            "warp_shape": (8, 32, 128),
            "num_stages": 4,
        },
    )

    assert analyze_candidate(problem, candidate).num_output_tiles == 1024


def test_dense_m_tile_estimate_is_unchanged():
    problem = _problem(
        layer=_layer(num_experts=0),
        gemm_type=GemmType.DENSE,
    )

    assert problem.estimate_num_blocks_m(32) == 128
