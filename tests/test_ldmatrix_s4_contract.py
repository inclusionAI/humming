"""Pure eligibility checks; intentionally runnable without CUDA package imports."""

import runpy
from pathlib import Path

import pytest

contract = runpy.run_path(str(Path(__file__).parents[1] / "humming/config/ldmatrix_s4.py"))
layer_reasons = contract["layer_rejection_reasons"]
specialization_reasons = contract["specialization_rejection_reasons"]

BASE = dict(
    sm_version=90,
    mma_type="wgmma",
    num_experts=0,
    a_dtype="int8",
    b_dtype="uint4",
    use_packed_k_layout=True,
    has_zero_point=False,
    is_fp_zero_point=False,
    use_int_weight_scale=False,
    use_fused_e8m0_scale=False,
    weight_scale_type="group",
    weight_scale_group_size=128,
    weight_scale_group_size_n=0,
    weight_scale_2_type="none",
    input_quant_mode="dynamic_token",
    input_scale_group_size=0,
    as_dtype="float32",
    bs_dtype="bfloat16",
    c_dtype="bfloat16",
    has_bias=False,
    shape_n=128,
    shape_k=128,
    pad_shape_n=0,
    pad_shape_k=0,
)
TUNING = dict(
    gemm_type="dense",
    block_shape=(16, 128, 64),
    warp_shape=(16, 32, 64),
    use_tma=True,
    use_tma_b=True,
    use_warp_spec=True,
    use_mbarrier=True,
    num_stages=3,
    use_stream_k=False,
)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"sm_version": 100}, "target"),
        ({"mma_type": "mma"}, "mma"),
        ({"num_experts": 1}, "moe"),
        ({"num_experts": 8}, "moe"),
        ({"a_dtype": "bfloat16"}, "physical_format"),
        *[
            ({"b_dtype": dtype}, "physical_format")
            for dtype in ("int4", "uint8", "uint3", "float4e2m1", "float8e4m3", "float8e8m0")
        ],
        ({"use_packed_k_layout": False}, "packed_k"),
        ({"has_zero_point": True}, "zero_point"),
        ({"is_fp_zero_point": True}, "zero_point"),
        ({"use_int_weight_scale": True}, "preprocess"),
        ({"use_fused_e8m0_scale": True}, "preprocess"),
        ({"weight_scale_2_type": "tensor"}, "weight_scale"),
        ({"weight_scale_group_size": 64}, "weight_scale"),
        ({"weight_scale_type": "block"}, "weight_scale"),
        ({"input_scale_group_size": 128}, "input_scale"),
        ({"input_quant_mode": "static_tensor"}, "input_scale"),
        ({"c_dtype": "float16"}, "scale_output_dtype"),
        ({"has_bias": True}, "bias"),
        ({"pad_shape_n": 8}, "padding"),
        ({"pad_shape_k": 8}, "padding"),
    ],
)
def test_physical_contract_exclusions(changes, reason):
    assert not layer_reasons(BASE)
    assert reason in layer_reasons(BASE | changes)


@pytest.mark.parametrize("axis", ["shape_n", "shape_k"])
@pytest.mark.parametrize("size,eligible", [(64, False), (127, False), (128, True), (129, False), (256, True)])
def test_panel_boundary(axis, size, eligible):
    assert (not layer_reasons(BASE | {axis: size})) == eligible


@pytest.mark.parametrize(
    "compiler,version,eligible",
    [
        ("NVCCCompiler", (13, 4), True),
        ("NVCCCompiler", (13, 3), False),
        ("NVRTCCompiler", (13, 4), False),
        ("NVCCCompiler", (12, 9), False),
    ],
)
def test_toolchain(compiler, version, eligible):
    assert (not contract["toolchain_rejection_reasons"](compiler, version)) == eligible


@pytest.mark.parametrize(
    "changes,reason",
    [
        *[
            ({"gemm_type": mode}, "execution_mode")
            for mode in ("indexed", "grouped_contiguous", "grouped_masked")
        ],
        ({"block_shape": (16, 128, 128)}, "geometry_k"),
        ({"warp_shape": (16, 32, 128)}, "geometry_k"),
        ({"block_shape": (16, 64, 64)}, "geometry_n"),
        ({"warp_shape": (8, 32, 64)}, "geometry_m"),
        ({"use_tma_b": False}, "staging"),
        ({"use_tma": False}, "staging"),
        ({"use_warp_spec": False}, "staging"),
        ({"use_mbarrier": False}, "staging"),
        ({"num_stages": 2}, "stages"),
        ({"multi_cast_size_a": 2}, "multicast"),
        ({"use_stream_k": True}, "schedule"),
        ({"use_pdl": True}, "schedule"),
        ({"use_f16_accum": True}, "schedule"),
    ],
)
def test_specialization_exclusions(changes, reason):
    assert not specialization_reasons(TUNING, 128)
    assert reason in specialization_reasons(TUNING | changes, 128)


def test_layout_compatibility():
    for name in ("NEW_LAYOUT", "LEGACY_LAYOUT"):
        contract["assert_layout_compatible"](contract[name], contract[name])
    for prepared, required in (("NEW_LAYOUT", "LEGACY_LAYOUT"), ("LEGACY_LAYOUT", "NEW_LAYOUT")):
        with pytest.raises(ValueError, match="does not match"):
            contract["assert_layout_compatible"](contract[prepared], contract[required])


@pytest.mark.parametrize("forced", [False, True])
def test_selected_loader_resolves_actual_tuning(forced):
    layer = BASE | {"use_ldmatrix_s4": not forced, "test_force_packed_k_legacy": forced}
    loader, reasons = contract["resolve_specialization_loader"](layer, TUNING, layer_reasons(layer))
    assert loader == ("packed_k_legacy" if forced else "ldmatrix_s8_s4")
    assert reasons == (("test_force_legacy",) if forced else ())


@pytest.mark.parametrize(
    "changes",
    [
        {"block_shape": (16, 128, 128)},  # KWARPS=2 even with WarpShapeK=64
        {"warp_shape": (16, 32, 128)},
        {"use_tma_b": False},
        *[{"gemm_type": mode} for mode in ("indexed", "grouped_contiguous", "grouped_masked")],
    ],
)
def test_early_capability_cannot_override_chosen_tuning(changes):
    assert not layer_reasons(BASE)
    with pytest.raises(ValueError, match="cannot use specialization"):
        contract["resolve_specialization_loader"](BASE | {"use_ldmatrix_s4": True}, TUNING | changes, ())


@pytest.mark.parametrize("projection", ["w13_gated", "w13_non_gated", "w2", "ep", "deepep"])
@pytest.mark.parametrize("mode", ["indexed", "grouped_contiguous", "grouped_masked"])
def test_expert_contracts_never_select_new(projection, mode):
    # Projection/transport labels do not bypass their common expert contract.
    layer = BASE | {"num_experts": 4, "use_ldmatrix_s4": False}
    loader, reasons = contract["resolve_specialization_loader"](
        layer, TUNING | {"gemm_type": mode}, layer_reasons(layer)
    )
    assert loader == "packed_k_legacy" and "moe" in reasons


def test_auto_dense_eligibility_has_satisfiable_specialization():
    resolve = contract["resolve_specialization_loader"]
    forced_tuning = contract["forced_tuning"]
    new_loader = contract["NEW_LOADER"]
    legacy_loader = contract["LEGACY_LOADER"]

    def selected_tuning(shape_n, block_m_seed=128):
        return {"gemm_type": "dense"} | forced_tuning(block_m_seed, shape_n)

    # Eligible dense layers resolve to the specialized loader.
    for shape_n, shape_k in ((128, 128), (256, 256), (384, 128), (4096, 4096)):
        layer = BASE | {"shape_n": shape_n, "shape_k": shape_k, "use_ldmatrix_s4": True}
        assert layer_reasons(layer) == ()
        for block_m_seed in (16, 96, 128, 200):
            tuning = selected_tuning(shape_n, block_m_seed)
            assert specialization_reasons(tuning, shape_n) == ()
            assert resolve(layer, tuning, ()) == (new_loader, ())

    # Reject incompatible selected tuning.
    eligible = BASE | {"use_ldmatrix_s4": True}
    incompatible = selected_tuning(128) | {"num_stages": 4}
    assert specialization_reasons(incompatible, 128)  # non-empty
    with pytest.raises(ValueError, match="cannot use specialization"):
        resolve(eligible, incompatible, ())

    # Force-legacy does not change layer eligibility.
    forced = BASE | {"use_ldmatrix_s4": False, "test_force_packed_k_legacy": True}
    loader, reasons = resolve(forced, selected_tuning(128), ())
    assert loader == legacy_loader
    assert "test_force_legacy" in reasons
    assert layer_reasons(BASE) == ()  # capability predicate unaffected by force-legacy
