import dataclasses

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.tune.candidate import (
    DeviceProfile,
    ScheduleCandidate,
    TuningDecision,
    TuningProblem,
    analyze_candidate,
    fit_pipeline_stages,
    get_geometry_rejection_reasons,
    get_problem_rejection_reasons,
)


def _layer(
    *,
    shape_n: int = 512,
    shape_k: int = 256,
    a_dtype=dtypes.bfloat16,
    b_dtype=dtypes.float4e2m1,
    as_dtype=None,
    bs_dtype=dtypes.float8e4m3,
    input_scale_group_size: int = 0,
    weight_scale_group_size: int = 16,
) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=8,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=dtypes.bfloat16,
        as_dtype=as_dtype,
        bs_dtype=bs_dtype,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        mma_type=MmaType.WGMMA,
    )


def _problem(
    *,
    layer_config: LayerConfig | None = None,
    max_smem_size: int = 227 * 1024,
    max_threads_per_sm: int = 2048,
) -> TuningProblem:
    return TuningProblem(
        layer_config=layer_config or _layer(),
        shape_m=4,
        gemm_type=GemmType.INDEXED,
        device=DeviceProfile(
            name="H200",
            sm_version=90,
            num_sms=132,
            max_smem_size=max_smem_size,
            max_threads_per_sm=max_threads_per_sm,
        ),
    )


def _candidate(**updates) -> ScheduleCandidate:
    config = {
        "block_shape": (8, 128, 128),
        "warp_shape": (8, 32, 64),
        "use_stream_k": False,
        "num_stages": 3,
        "num_ctas_per_sm": 2,
    }
    config.update(updates)
    return ScheduleCandidate.from_config("indexed_a16", config)


def test_schedule_candidate_is_immutable_and_updates_config():
    config = {
        "block_shape": (8, 128, 128),
        "use_stream_k": False,
        "warp_shape": (8, 32, 64),
    }
    candidate = ScheduleCandidate.from_config("base", config)
    updated = candidate.with_updates(
        candidate_id="three_stage",
        warp_shape=(8, 32, 32),
        num_stages=3,
    )

    assert candidate.to_config() == config
    assert updated.to_config() == config | {
        "warp_shape": (8, 32, 32),
        "num_stages": 3,
    }
    assert updated.candidate_id == "three_stage"
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.candidate_id = "mutated"

    direct = ScheduleCandidate(
        candidate_id="direct",
        block_shape=(8, 128, 128),
        warp_shape=(8, 32, 64),
    )
    assert direct.to_config()["block_shape"] == (8, 128, 128)


def test_analysis_reports_resources_grid_and_selected_config():
    problem = _problem()
    candidate = _candidate()
    analysis = analyze_candidate(problem, candidate)
    decision = TuningDecision(
        problem=problem,
        family="indexed_a16",
        selected=candidate,
        considered=(analysis,),
        reason="measured small-M schedule",
    )

    assert analysis.legal
    assert analysis.num_math_threads == 256
    assert analysis.num_load_threads == 256
    assert analysis.num_threads == 256
    assert analysis.smem_size > 0
    assert analysis.num_output_tiles == 16
    assert analysis.thread_smem_cta_limit >= 2
    assert analysis.waves == 1
    assert decision.selected_analysis is analysis
    assert decision.to_config() == candidate.to_config()


def test_pipeline_fit_uses_the_shared_smem_analysis():
    problem = _problem()
    candidate = _candidate(num_stages=4, num_ctas_per_sm=1)
    stage_three = candidate.with_updates(num_stages=3)
    stage_three_smem = analyze_candidate(problem, stage_three).smem_size
    constrained = dataclasses.replace(
        problem,
        device=dataclasses.replace(
            problem.device,
            max_smem_size=stage_three_smem,
        ),
    )

    fitted = fit_pipeline_stages(constrained, candidate)

    assert fitted.num_stages == 3


@pytest.mark.parametrize(
    ("block_shape", "warp_shape"),
    [
        ((8, 0, 64), (8, 32, 64)),
        ((8, 192, 64), (8, 32, 64)),
        ((8, 128, 192), (8, 32, 64)),
        ((8, 128, 96), (8, 32, 64)),
        ((8, 384, 64), (8, 32, 64)),
        ((8, 64, 64), (8, 16, 64)),
    ],
)
def test_geometry_validator_rejects_invalid_shapes(block_shape, warp_shape):
    reasons = get_geometry_rejection_reasons(
        _layer(),
        block_shape,
        warp_shape,
    )

    assert reasons


def test_analysis_rejects_scale_group_that_does_not_nest_tile():
    problem = _problem(layer_config=_layer(weight_scale_group_size=96))
    analysis = analyze_candidate(problem, _candidate())

    assert not analysis.launchable


def test_analysis_checks_device_resource_limits():
    problem = _problem(max_threads_per_sm=256)
    analysis = analyze_candidate(problem, _candidate())

    assert analysis.launchable
    assert not analysis.meets_resource_target
    assert analysis.thread_smem_cta_limit < analysis.candidate.num_ctas_per_sm


def test_analysis_treats_per_cta_smem_overflow_as_hard_violation():
    analysis = analyze_candidate(_problem(max_smem_size=1024), _candidate())

    assert not analysis.launchable


def test_geometry_rejects_partial_input_scale_group():
    layer = _layer(
        shape_n=2880,
        shape_k=2880,
        a_dtype=dtypes.float8e4m3,
        as_dtype=dtypes.float32,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=128,
        weight_scale_group_size=32,
    )

    reasons = get_problem_rejection_reasons(layer)

    assert reasons


def test_geometry_rejects_e8m0_scales_on_wgmma():
    layer = _layer(
        a_dtype=dtypes.float8e4m3,
        as_dtype=dtypes.float8e8m0,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=32,
        weight_scale_group_size=32,
    )

    reasons = get_problem_rejection_reasons(layer)

    assert reasons


def test_problem_rejects_scale_groups_smaller_than_mma_k():
    layer = _layer(
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.uint4,
        as_dtype=dtypes.float32,
        bs_dtype=dtypes.float32,
        input_scale_group_size=16,
        weight_scale_group_size=16,
    )

    reasons = get_problem_rejection_reasons(layer)

    assert len(reasons) == 2


def test_geometry_allows_integer_wgmma_warp_m_eight():
    layer = _layer(
        a_dtype=dtypes.int8,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=64,
    )

    reasons = get_geometry_rejection_reasons(
        layer,
        block_shape=(8, 128, 64),
        warp_shape=(8, 32, 64),
    )

    assert not reasons


def test_analysis_rejects_more_than_1024_threads():
    analysis = analyze_candidate(
        _problem(),
        _candidate(
            block_shape=(64, 512, 256),
            warp_shape=(8, 32, 64),
            num_ctas_per_sm=1,
        ),
    )

    assert analysis.num_threads > 1024
    assert not analysis.launchable


@pytest.mark.parametrize(
    "updates",
    [
        {
            "num_stages": 2,
            "use_warp_spec": True,
            "use_tma": True,
            "use_mbarrier": True,
        },
        {"use_warp_spec": True, "use_tma": True, "use_mbarrier": False},
        {"multi_cast_size_a": 2},
        {
            "block_shape": (8, 128, 64),
            "warp_shape": (8, 32, 16),
            "use_warp_spec": True,
            "use_tma": True,
            "use_mbarrier": True,
        },
    ],
    ids=["pipeline-depth", "mbarrier", "indexed-multicast", "warp-iterations"],
)
def test_analysis_rejects_invalid_pipeline_and_transfer_modes(updates):
    analysis = analyze_candidate(_problem(), _candidate(**updates))

    assert not analysis.launchable


def test_decision_rejects_an_illegal_selection():
    problem = _problem()
    candidate = _candidate(block_shape=(8, 64, 128))
    analysis = analyze_candidate(problem, candidate)

    with pytest.raises(ValueError):
        TuningDecision(
            problem=problem,
            family="indexed_a16",
            selected=candidate,
            considered=(analysis,),
            reason="invalid",
        )
