import dataclasses

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig
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
        sm_version=90,
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


@pytest.mark.parametrize("block_n", [32, 64, 128])
def test_mma_group_scale_packing_limits_block_n(block_n):
    layer = dataclasses.replace(
        _layer(a_dtype=dtypes.float8e4m3, bs_dtype=dtypes.bfloat16, weight_scale_group_size=64),
        sm_version=89,
        use_packed_k_layout=False,
    )
    reasons = get_geometry_rejection_reasons(layer, (64, block_n, 128), (64, 16, 128))
    expected_rejection = block_n < 64 and not layer.should_apply_bs_on_c
    assert ("MMA group scales require block_n >= 64" in reasons) == expected_rejection


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


@pytest.mark.parametrize("warp_k", [64, 128, 256])
def test_packed_k_geometry_requires_k128(warp_k):
    layer = _layer(
        a_dtype=dtypes.float8e4m3,
        bs_dtype=dtypes.bfloat16,
        input_scale_group_size=128,
        weight_scale_group_size=128,
    )
    assert layer.use_packed_k_layout
    reasons = get_geometry_rejection_reasons(layer, (64, 64, 256), (64, 16, warp_k))
    if warp_k == 128:
        assert not reasons
    else:
        assert "use_packed_k_layout requires warp_k=128" in reasons


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


@pytest.mark.parametrize(
    "sm_version,a_dtype,b_dtype,group_size,scale_dtype,packed_k,use_f16_accum,expected",
    (
        (90, "bfloat16", "uint4", 128, "bfloat16", False, False, {"mma", "wgmma"}),
        (90, "float8e4m3", "uint4", 128, "bfloat16", True, False, {"wgmma"}),
        (90, "int4", "int4", 0, "bfloat16", False, False, {"mma"}),
        (103, "bfloat16", "uint4", 128, "bfloat16", False, False, {"mma", "umma"}),
        (103, "float8e4m3", "float8e4m3", 0, "bfloat16", False, False, {"mma", "umma"}),
        (103, "float8e4m3", "float4e2m1", 32, "float8e8m0", False, False, {"umma"}),
        (100, "int8", "uint4", 0, "bfloat16", False, False, {"mma", "umma"}),
        (103, "int8", "uint4", 0, "bfloat16", False, False, {"mma"}),
        (103, "float16", "uint4", 128, "float16", False, True, {"mma"}),
        (120, "float4e2m1", "float4e2m1", 32, "float8e8m0", False, False, {"mxmma"}),
    ),
)
def test_sampled_backends_match_fixed_layout(
    sm_version, a_dtype, b_dtype, group_size, scale_dtype, packed_k, use_f16_accum, expected, monkeypatch
):
    from humming.config import ComputeConfig
    from humming.device import DeviceInfo
    from humming.kernel.humming import HummingKernel
    from humming.testing import tuning

    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: sm_version))
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 132))
    monkeypatch.setattr(DeviceInfo, "is_ppu", property(lambda self: False))
    monkeypatch.setattr(DeviceInfo, "max_registers_per_sm", property(lambda self: 65536))
    monkeypatch.setattr(HummingKernel, "_instances", {})
    monkeypatch.setattr(HummingKernel, "prepare", lambda self: None)
    monkeypatch.setattr(HummingKernel, "register_kernel", lambda self: None)
    monkeypatch.setattr(tuning, "NUM_SAMPLED_TUNING_CONFIGS", 20)
    layer = LayerConfig(
        sm_version=sm_version,
        shape_n=512,
        shape_k=512,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype="float16" if use_f16_accum else "bfloat16",
        bs_dtype=scale_dtype,
        weight_scale_group_size=group_size,
        use_packed_k_layout=packed_k,
    )
    compute = ComputeConfig(gemm_type=GemmType.DENSE, use_f16_accum=use_f16_accum)
    configs = tuning.sample_test_tuning_configs(layer, compute, sample_size=20)
    assert {config["mma_type"] for config in configs} == expected
    assert "mma_type" not in layer.to_dict()
    for config in configs:
        # Exercise the real kernel validators and code generation without compiling.
        tuning_values = tuning.create_tuning_config(config).to_dict()
        kernel = HummingKernel(**(layer.to_dict() | compute.to_dict() | tuning_values))
        assert kernel.mma_type.value == config["mma_type"]
        assert kernel.use_raw_weight == layer.use_raw_weight


def test_sampled_umma_covers_cooperative_and_dequant_options(monkeypatch):
    from humming.config import ComputeConfig
    from humming.device import DeviceInfo
    from humming.testing import tuning

    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: 103))
    layer = LayerConfig(
        sm_version=103,
        shape_n=512,
        shape_k=512,
        a_dtype="bfloat16",
        b_dtype="uint4",
        c_dtype="bfloat16",
        weight_scale_group_size=128,
    )
    compute = ComputeConfig(gemm_type=GemmType.INDEXED, use_batch_invariant=True)
    layer = dataclasses.replace(layer, num_experts=4)
    configs = tuning.sample_test_tuning_configs(layer, compute)
    umma_configs = [config for config in configs if config["mma_type"] == "umma"]
    for name in ("umma_cta_group_size", "umma_num_dequant_warpgroups", "output_chunk_rows"):
        assert {config[name] for config in umma_configs} == set(tuning.SAMPLED_TUNING_VALUES[name])
    assert {config["use_tma_b"] for config in umma_configs} == {False, True}
    for config in configs:
        assert not config["use_stream_k"]
        assert not config["use_tma_a"] and not config["use_tma_c"]
        assert config["block_shape"][2] == config["warp_shape"][2]
    assert configs == tuning.sample_test_tuning_configs(layer, compute)


@pytest.mark.parametrize("output_chunk_rows", (-32, 8, 16, 24, 48, 288))
def test_output_chunk_rows_rejects_invalid_heights(output_chunk_rows):
    from humming.config import TuningConfig

    with pytest.raises(AssertionError, match="multiple of 32"):
        TuningConfig(
            block_shape=(64, 128, 128),
            warp_shape=(32, 32, 64),
            output_chunk_rows=output_chunk_rows,
        )


@pytest.mark.parametrize("sm_version", (100, 103, 107, 110))
@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    (
        (dtypes.bfloat16, dtypes.uint4),
        (dtypes.int8, dtypes.int8),
        (dtypes.float8e4m3, dtypes.float8e4m3),
        (dtypes.float8e4m3, dtypes.float4e2m1),
        (dtypes.float4e2m1, dtypes.float4e2m1),
    ),
)
def test_umma_architecture_selection(sm_version, a_dtype, b_dtype):
    from humming.config import MmaType
    from humming.config.mma import get_default_mma_type

    scale_config = {}
    if a_dtype == dtypes.float4e2m1:
        scale_config = dict(
            as_dtype=dtypes.float8e4m3,
            bs_dtype=dtypes.float8e4m3,
            input_scale_group_size=16,
            weight_scale_group_size=16,
            input_quant_mode="dynamic_group",
        )
    config = LayerConfig(
        sm_version=sm_version,
        shape_n=256,
        shape_k=256,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=dtypes.bfloat16,
        **scale_config,
    )
    expected = MmaType.MMA if a_dtype == dtypes.int8 and sm_version in (103, 107) else MmaType.UMMA
    assert get_default_mma_type(config) == expected


@pytest.mark.parametrize(
    "sm_version,a_dtype,shape_n,use_f16_accum,expected",
    (
        (103, "bfloat16", 256, False, "umma"),
        (103, "float16", 256, False, "umma"),
        (103, "float16", 256, True, "mma"),
        (103, "bfloat16", 192, False, "mma"),
        (90, "bfloat16", 256, False, "wgmma"),
        (80, "bfloat16", 256, False, "mma"),
    ),
)
def test_heuristic_tests_prefer_available_backend(
    sm_version, a_dtype, shape_n, use_f16_accum, expected, monkeypatch
):
    from humming.config import ComputeConfig
    from humming.device import DeviceInfo
    from humming.testing import tuning

    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: sm_version))
    monkeypatch.setattr(DeviceInfo, "is_ppu", property(lambda self: False))
    layer = LayerConfig(
        shape_n=shape_n,
        shape_k=1024,
        sm_version=sm_version,
        a_dtype=a_dtype,
        b_dtype="uint4",
        c_dtype=a_dtype,
        weight_scale_group_size=128,
    )
    compute = ComputeConfig(gemm_type=GemmType.DENSE, use_f16_accum=use_f16_accum)
    configs = tuning.generate_heuristics_configs(layer, compute, (1, 17, 257))
    assert {config["mma_type"] for config in configs} == {expected}


def test_heuristic_test_mode_has_separate_cache_entries(monkeypatch):
    from humming.device import current_device
    from humming.tune import get_heuristics_config

    if current_device.sm_version // 10 not in (10, 11):
        pytest.skip("requires a device with both MMA and UMMA")
    layer = LayerConfig(
        shape_n=256,
        shape_k=1024,
        a_dtype="bfloat16",
        b_dtype="uint4",
        c_dtype="bfloat16",
    )
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("HUMMING_TEST_TUNING_SOURCE", raising=False)
    assert get_heuristics_config(layer, 17)["mma_type"] == "mma"
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "heuristic")
    assert get_heuristics_config(layer, 17)["mma_type"] == "umma"
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "sampled")
    assert get_heuristics_config(layer, 17)["mma_type"] == "mma"


@pytest.mark.parametrize(
    "warp_m,num_ctas,expected",
    (
        (16, 3, True),
        (32, 3, False),
        (32, 2, True),
        (64, 3, False),
        (48, 4, False),
        (64, 4, False),
        (128, 1, False),
        (128, 3, False),
    ),
)
def test_warp_specialization_register_budget(warp_m, num_ctas, expected, monkeypatch):
    from humming.config import ComputeConfig, TuningConfig
    from humming.config.mma import get_register_budget_error
    from humming.device import DeviceInfo
    from humming.kernel.humming import HummingKernel
    from humming.testing import tuning

    monkeypatch.setattr(DeviceInfo, "max_registers_per_sm", property(lambda self: 65536))

    layer = dataclasses.replace(_layer(), sm_version=103)
    compute = ComputeConfig(gemm_type=GemmType.INDEXED)
    config = TuningConfig(
        mma_type="mma",
        block_shape=(warp_m, 128, 256),
        warp_shape=(warp_m, 64, 64),
        num_stages=2,
        num_ctas_per_sm=num_ctas,
        use_warp_spec=True,
        use_tma=True,
        use_tma_a=False,
        use_tma_c=False,
        use_stream_k=False,
        output_chunk_rows=32,
        smem_reuse_mode="last_stage",
    )
    assert (get_register_budget_error(layer, config) is None) == expected
    schedule_fields = {field.name for field in dataclasses.fields(ScheduleCandidate)}
    schedule_config = {name: value for name, value in config.to_dict().items() if name in schedule_fields}
    schedule = ScheduleCandidate.from_config("register-budget", schedule_config)
    analysis = analyze_candidate(_problem(layer_config=layer), schedule)
    has_register_error = any(
        reason.startswith("register budget exceeded:") for reason in analysis.rejection_reasons
    )
    assert has_register_error == (not expected)
    if not expected:
        assert not tuning._fits_device_resources(layer, compute, (config.to_dict(), {}))
        monkeypatch.setattr(HummingKernel, "_instances", {})
        monkeypatch.setattr(
            HummingKernel, "prepare", lambda self: pytest.fail("invalid config reached compilation")
        )
        with pytest.raises(ValueError, match="register budget exceeded"):
            HummingKernel(**(layer.to_dict() | compute.to_dict() | config.to_dict()))


@pytest.mark.parametrize("a_dtype,expected", [("int4", "mma"), ("int8", "wgmma"), ("bfloat16", "wgmma")])
def test_sm90_default_backend_supports_activation(a_dtype, expected):
    from humming.config.mma import get_default_mma_type

    layer = LayerConfig(
        sm_version=90,
        shape_n=256,
        shape_k=256,
        a_dtype=a_dtype,
        b_dtype="uint4",
        c_dtype="bfloat16",
    )
    assert get_default_mma_type(layer).value == expected


@pytest.mark.parametrize(
    "warp_m,warp_n,group_size,num_ctas,expected",
    [(176, 16, 128, 4, False), (80, 64, 128, 3, False), (128, 64, 0, 3, False), (32, 16, 128, 2, True)],
)
def test_sampled_wgmma_accounts_for_all_accumulators(
    warp_m, warp_n, group_size, num_ctas, expected, monkeypatch
):
    from humming.config import ComputeConfig
    from humming.device import DeviceInfo
    from humming.testing import tuning

    monkeypatch.setattr(DeviceInfo, "max_registers_per_sm", property(lambda self: 65536))
    monkeypatch.setattr(DeviceInfo, "max_threads_per_sm", property(lambda self: 2048))
    monkeypatch.setattr(tuning, "fits_device_smem", lambda *args: True)
    layer = LayerConfig(
        sm_version=90,
        shape_n=1024,
        shape_k=1024,
        a_dtype="float8e4m3",
        b_dtype="uint4",
        c_dtype="bfloat16",
        input_scale_group_size=group_size,
        weight_scale_group_size=group_size,
        use_fused_e8m0_scale=False,
    )
    config = dict(
        mma_type="wgmma",
        block_shape=(warp_m, warp_n * 4, 128),
        warp_shape=(warp_m, warp_n, 128),
        num_ctas_per_sm=num_ctas,
        use_warp_spec=False,
        num_stages=2,
        use_tma=False,
    )
    compute = ComputeConfig(gemm_type=GemmType.DENSE)
    assert tuning._fits_device_resources(layer, compute, (config, {})) == expected


@pytest.mark.parametrize(
    "mma_type,raw_weight,group_size,packed,warp_shape,budget,expected",
    [
        # MMA: 192 accumulator + 24 A + 16 B = 232; equality must reject.
        ("mma", False, 0, False, (96, 64, 64), 240, False),
        ("mma", False, 0, False, (96, 64, 64), 248, True),
        # WGMMA excludes A in smem, and excludes B as well for raw-weight SS.
        ("wgmma", False, 0, False, (96, 64, 64), 216, False),
        ("wgmma", False, 0, False, (96, 64, 64), 224, True),
        ("wgmma", True, 0, False, (96, 64, 64), 200, False),
        ("wgmma", True, 0, False, (96, 64, 64), 208, True),
        # FP8 group scale: 128 * 1.25 + 16 A + 16 B = 192, not 288.
        ("mma", False, 128, False, (64, 64, 128), 200, False),
        ("mma", False, 128, False, (64, 64, 128), 208, True),
        ("wgmma", True, 128, False, (64, 64, 128), 168, False),
        ("wgmma", True, 128, False, (64, 64, 128), 176, True),
        # WGMMA packed B covers all K=128 slabs (16 registers for FP8).
        ("wgmma", False, 128, True, (64, 16, 128), 64, False),
        ("wgmma", False, 128, True, (64, 16, 128), 72, True),
    ],
)
def test_sampled_operand_register_budget(
    mma_type, raw_weight, group_size, packed, warp_shape, budget, expected
):
    from humming.config import TuningConfig
    from humming.config.mma import get_register_budget_error

    a_dtype = "float8e4m3" if group_size else "float16"
    layer = LayerConfig(
        sm_version=90,
        shape_n=1024,
        shape_k=1024,
        a_dtype=a_dtype,
        b_dtype=a_dtype if raw_weight else "uint4",
        c_dtype="float16",
        use_packed_k_layout=packed,
        input_scale_group_size=group_size,
        weight_scale_group_size=group_size,
        use_fused_e8m0_scale=False,
    )
    config = TuningConfig(
        mma_type=mma_type,
        warp_shape=warp_shape,
        block_shape=(warp_shape[0], warp_shape[1] * 4, warp_shape[2]),
        use_warp_spec=False,
        num_ctas_per_sm=1,
    )
    error = get_register_budget_error(layer, config, registers_per_sm=budget * 128)
    assert (error is None) == expected


@pytest.mark.parametrize("registers_per_sm,expected", [(65536, False), (131072, True)])
def test_device_register_budget(registers_per_sm, expected, monkeypatch):
    from humming.config import ComputeConfig, TuningConfig
    from humming.config.mma import get_register_budget_error
    from humming.device import DeviceInfo
    from humming.testing import tuning

    monkeypatch.setattr(DeviceInfo, "max_registers_per_sm", property(lambda self: registers_per_sm))
    monkeypatch.setattr(DeviceInfo, "max_threads_per_sm", property(lambda self: 2048))
    monkeypatch.setattr(tuning, "fits_device_smem", lambda *args: True)
    layer = LayerConfig(
        sm_version=80,
        shape_n=1024,
        shape_k=1024,
        a_dtype="int8",
        b_dtype="int8",
        c_dtype="float16",
    )
    config = TuningConfig(
        mma_type="mma",
        block_shape=(64, 128, 128),
        warp_shape=(64, 32, 64),
        num_ctas_per_sm=4,
        use_warp_spec=False,
        use_tma=False,
    )
    assert (get_register_budget_error(layer, config) is None) == expected
    assert tuning._fits_device_resources(layer, ComputeConfig(), (config.to_dict(), {})) == expected
    problem = _problem(layer_config=layer)
    problem = dataclasses.replace(
        problem,
        device=dataclasses.replace(problem.device, max_registers_per_sm=registers_per_sm),
    )
    schedule = _candidate(
        mma_type="mma",
        block_shape=config.block_shape,
        warp_shape=config.warp_shape,
        num_ctas_per_sm=config.num_ctas_per_sm,
    )
    analysis = analyze_candidate(problem, schedule)
    has_register_error = any(
        reason.startswith("register budget exceeded:") for reason in analysis.rejection_reasons
    )
    assert has_register_error == (not expected)


@pytest.mark.parametrize("sm_version", (120, 121))
@pytest.mark.parametrize("compiler_version", ((13, 0), (13, 1)))
@pytest.mark.parametrize(
    "a_dtype,b_dtype,input_group,weight_group,scale_dtype,requires_13_1",
    (
        ("float4e2m1", "float4e2m1", 16, 16, "float8e8m0", True),
        ("float4e0m3", "float4e0m3", 16, 16, "float8e8m0", True),
        ("float4e2m1", "float4e0m3", 16, 16, "float8e8m0", True),
        ("float4e0m3", "float4e2m1", 16, 16, "float8e4m3", False),
        ("float4e2m1", "float4e0m3", 16, 16, "float8e4m3", False),
        ("float4e2m1", "float4e2m1", 32, 32, "float8e8m0", False),
        ("float8e4m3", "float4e2m1", 32, 32, "float8e8m0", False),
        ("float4e0m3", "float4e0m3", 16, 0, "float8e8m0", True),
        ("float4e0m3", "float4e0m3", 0, 16, "float8e4m3", False),
        ("float4e2m1", "float4e2m1", 0, 0, "float8e8m0", False),
    ),
)
def test_mxmma_compiler_version_matches_scale_format(
    sm_version,
    compiler_version,
    a_dtype,
    b_dtype,
    input_group,
    weight_group,
    scale_dtype,
    requires_13_1,
    monkeypatch,
):
    import humming.kernel.humming as kernel_module
    import humming.testing.runner as runner_module
    from humming.config import ComputeConfig, TuningConfig
    from humming.device import DeviceInfo
    from humming.kernel.humming import HummingKernel
    from humming.testing import KernelTestCase, KernelTestRunner, skip_if_unsupported

    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: sm_version))
    monkeypatch.setattr(kernel_module, "_cuda_compiler_version", lambda _: compiler_version)
    monkeypatch.setattr(runner_module, "_cuda_compiler_version", lambda _: compiler_version)
    monkeypatch.setattr(HummingKernel, "_instances", {})
    prepared = []
    monkeypatch.setattr(HummingKernel, "prepare", lambda self: prepared.append(self))
    monkeypatch.setattr(HummingKernel, "register_kernel", lambda self: None)
    layer = LayerConfig(
        sm_version=sm_version,
        shape_n=256,
        shape_k=256,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype="bfloat16",
        as_dtype=scale_dtype if input_group else None,
        bs_dtype=scale_dtype if weight_group else "bfloat16",
        input_scale_group_size=input_group,
        weight_scale_group_size=weight_group,
    )
    compute = ComputeConfig(gemm_type=GemmType.DENSE)
    tuning = TuningConfig(
        mma_type="mxmma",
        block_shape=(32, 64, 128),
        warp_shape=(32, 32, 128),
        num_stages=2,
        use_stream_k=False,
    )
    kernel_args = layer.to_dict() | compute.to_dict() | tuning.to_dict()
    should_reject = requires_13_1 and compiler_version < (13, 1)
    message = "scale_vec::4X and UE8M0 scales requires CUDA 13.1"
    if should_reject:
        with pytest.raises(AssertionError, match=message):
            HummingKernel(**kernel_args)
        assert not prepared
    else:
        kernel = HummingKernel(**kernel_args)
        assert prepared == [kernel]
        if requires_13_1:
            assert "kind::mxf4nvf4.block_scale.scale_vec::4X" in kernel.code
            assert ".f32.e2m1.e2m1.f32.ue8m0" in kernel.code

    # Hardware-only checks must not reject every E0M3 configuration on SM121.
    skip_if_unsupported(a_dtype=layer.a_dtype, mma_type="mxmma")
    monkeypatch.setattr(KernelTestRunner, "prepare_weight", lambda self: None)
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "sampled")

    def stop_before_sampling(*args):
        raise RuntimeError("reached tuning generation")

    monkeypatch.setattr(runner_module, "sample_test_tuning_configs", stop_before_sampling)
    case = KernelTestCase(name="mxmma-version", layer_config=layer, compute_config=compute)
    runner = KernelTestRunner(case)
    if should_reject:
        with pytest.raises(pytest.skip.Exception, match=message):
            runner.prepare_kernels((1,))
    else:
        with pytest.raises(RuntimeError, match="reached tuning generation"):
            runner.prepare_kernels((1,))
