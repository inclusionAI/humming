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


def _layer(*, weight_scale_group_size: int = 16) -> LayerConfig:
    return LayerConfig(
        shape_n=512,
        shape_k=256,
        num_experts=8,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e4m3,
        input_scale_group_size=0,
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


def test_schedule_candidate_is_immutable_and_preserves_config_order():
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

    assert list(candidate.to_config()) == list(config)
    assert list(updated.to_config()) == [
        "block_shape",
        "use_stream_k",
        "warp_shape",
        "num_stages",
    ]
    assert updated.candidate_id == "three_stage"
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.candidate_id = "mutated"


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
    assert analysis.num_load_threads == 0
    assert analysis.num_threads == 256
    assert analysis.smem_size > 0
    assert analysis.num_output_tiles == 16
    assert analysis.thread_smem_cta_limit >= 2
    assert analysis.waves == 1
    assert decision.selected_analysis is analysis
    assert list(decision.to_config()) == list(candidate.to_config())


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
    ("block_shape", "warp_shape", "expected_reason"),
    [
        ((8, 0, 64), (8, 32, 64), "dimensions must be positive"),
        ((8, 192, 64), (8, 32, 64), "shape_n=512 is not divisible"),
        ((8, 128, 192), (8, 32, 64), "shape_k=256 is not divisible"),
        ((8, 128, 96), (8, 32, 64), "does not nest warp_shape"),
        ((8, 128, 128), (8, 24, 64), "warp_n=24 must be a power of two"),
        ((8, 384, 64), (8, 32, 64), "ratios must be powers of two"),
        ((8, 64, 64), (8, 32, 64), "multiple of four warp-N tiles"),
        ((8, 64, 64), (8, 16, 64), "warp_n=16 is smaller than minimum 32"),
        ((8, 128, 64), (8, 32, 16), "warp_k=16 is smaller than minimum 32"),
    ],
)
def test_geometry_validator_explains_rejected_shapes(
    block_shape,
    warp_shape,
    expected_reason,
):
    reasons = get_geometry_rejection_reasons(
        _layer(),
        block_shape,
        warp_shape,
    )

    assert any(expected_reason in reason for reason in reasons)


def test_analysis_checks_scale_nesting_and_device_limits():
    problem = _problem(
        layer_config=_layer(weight_scale_group_size=96),
        max_smem_size=1024,
        max_threads_per_sm=256,
    )
    analysis = analyze_candidate(problem, _candidate())

    assert not analysis.legal
    assert any(
        "weight scale group=96 do not nest" in reason
        for reason in analysis.rejection_reasons
    )
    assert any("smem_size=" in reason for reason in analysis.rejection_reasons)
    assert any("residency limit" in reason for reason in analysis.rejection_reasons)


def test_geometry_rejects_partial_input_scale_group():
    layer = LayerConfig(
        shape_n=2880,
        shape_k=2880,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        as_dtype=dtypes.float32,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=128,
        weight_scale_group_size=32,
        mma_type=MmaType.WGMMA,
    )

    reasons = get_problem_rejection_reasons(layer)

    assert any("input scale group=128" in reason for reason in reasons)


def test_geometry_rejects_e8m0_scales_on_wgmma():
    layer = LayerConfig(
        shape_n=512,
        shape_k=256,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        as_dtype=dtypes.float8e8m0,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=32,
        weight_scale_group_size=32,
        mma_type=MmaType.WGMMA,
    )

    reasons = get_problem_rejection_reasons(layer)

    assert any("must use float32 storage" in reason for reason in reasons)


def test_problem_rejects_scale_groups_smaller_than_mma_k():
    layer = LayerConfig(
        shape_n=512,
        shape_k=256,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16,
        as_dtype=dtypes.float32,
        bs_dtype=dtypes.float32,
        input_scale_group_size=16,
        weight_scale_group_size=16,
        mma_type=MmaType.WGMMA,
    )

    reasons = get_problem_rejection_reasons(layer)

    assert "input scale group=16 is smaller than MMA K=32" in reasons
    assert "weight scale group=16 is smaller than MMA K=32" in reasons


def test_geometry_allows_integer_wgmma_warp_m_eight():
    layer = LayerConfig(
        shape_n=512,
        shape_k=256,
        a_dtype=dtypes.int8,
        b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=64,
        mma_type=MmaType.WGMMA,
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
    assert any("CTA limit 1024" in reason for reason in analysis.rejection_reasons)


def test_analysis_checks_pipeline_and_transfer_legality():
    analysis = analyze_candidate(
        _problem(),
        _candidate(
            num_stages=2,
            use_warp_spec=True,
            use_tma=True,
            use_mbarrier=False,
            multi_cast_size_a=2,
        ),
    )

    assert any(
        "at least three stages" in reason for reason in analysis.rejection_reasons
    )
    assert any("require mbarrier" in reason for reason in analysis.rejection_reasons)
    assert any(
        "multicast requires a dense GEMM" in reason
        for reason in analysis.rejection_reasons
    )


def test_analysis_rejects_indexed_tma_input_and_output_transfers():
    analysis = analyze_candidate(
        _problem(),
        _candidate(
            use_tma=True,
            use_mbarrier=True,
            use_tma_a=True,
            use_tma_as=True,
            use_tma_c=True,
        ),
    )

    assert any(
        "indexed GEMM does not support TMA A/AS/C" in reason
        for reason in analysis.rejection_reasons
    )
    assert any(
        "TMA input-scale loads require M-major" in reason
        for reason in analysis.rejection_reasons
    )


def test_analysis_checks_warp_iterations_and_split_output_constraints():
    analysis = analyze_candidate(
        _problem(),
        _candidate(
            warp_shape=(8, 32, 16),
            use_warp_spec=True,
            use_tma=True,
            use_mbarrier=True,
            use_tma_c=True,
            num_write_splits=2,
        ),
    )

    assert any(
        "at least two warp iterations" in reason
        for reason in analysis.rejection_reasons
    )
    assert any(
        "split output writes require" in reason for reason in analysis.rejection_reasons
    )


def test_decision_rejects_an_illegal_selection():
    problem = _problem()
    candidate = _candidate(block_shape=(8, 64, 128))
    analysis = analyze_candidate(problem, candidate)

    with pytest.raises(ValueError, match="selected candidate must be legal"):
        TuningDecision(
            problem=problem,
            family="indexed_a16",
            selected=candidate,
            considered=(analysis,),
            reason="invalid",
        )
