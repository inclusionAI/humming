import itertools

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.tune.sm90_h20 import Sm90H20Heuristics
from humming.tune.sm90_h20_families import (
    fused_e8m0_moe_in_scope,
    make_h20_device_profile,
    select_fused_e8m0_moe,
)
from humming.tune.candidate import TuningProblem

_NUM_SMS = 78


@pytest.fixture(autouse=True)
def _h20_num_sms(monkeypatch):
    monkeypatch.setattr(
        Sm90H20Heuristics, "get_num_sms", classmethod(lambda cls: _NUM_SMS)
    )


def _fused_moe_layer(
    shape_n: int,
    shape_k: int,
    num_experts: int,
    input_scale_group_size: int,
    weight_scale_group_size: int = 32,
) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        weight_scale_type="group",
        mma_type=MmaType.WGMMA,
    )


_SHAPES = (
    (6144, 7168, 48),  # DSV4-Pro w13
    (7168, 3072, 48),  # DSV4-Pro w2
    (3072, 7168, 16),
    (1536, 2048, 64),
    (4096, 1536, 32),
    (1344, 3072, 48),  # 1344 % 128 != 0: forces the block_n fit loop
    (6144, 1088, 48),  # 1088 % 128 != 0: block_k halving (isg=0 only; isg=128
    # makes the problem itself illegal and must route to legacy)
    (1536, 7168, 8),  # 96 output tiles: grid fill stops at CTA2
    (128, 7168, 8),  # 8 output tiles: CTA1, num_warps==4 reshape, num_sms trim
)
_GEMM_TYPES = (
    GemmType.INDEXED,
    GemmType.GROUPED_CONTIGUOUS,
    GemmType.GROUPED_MASKED,
)
_SHAPE_M_VALUES = (48, 96, 384, 768, 1536, 3072, 4098, 6144, 12288, 24576, 49152)


def _grid():
    for (shape_n, shape_k, num_experts), gemm_type, isg, shape_m, f16 in (
        itertools.product(
            _SHAPES, _GEMM_TYPES, (0, 128), _SHAPE_M_VALUES, (False, True)
        )
    ):
        yield _fused_moe_layer(shape_n, shape_k, num_experts, isg), gemm_type, shape_m, f16


def _problem(layer, gemm_type, shape_m, use_f16_accum):
    return TuningProblem(
        layer_config=layer,
        shape_m=shape_m,
        gemm_type=gemm_type,
        device=make_h20_device_profile(_NUM_SMS),
        use_f16_accum=use_f16_accum,
    )


def test_family_matches_legacy_across_grid():
    """The migrated family must reproduce the legacy config for every in-scope
    point; out-of-scope points must keep routing to the legacy path."""
    covered = 0
    for layer, gemm_type, shape_m, f16 in _grid():
        legacy = Sm90H20Heuristics._get_config_legacy(
            layer, shape_m, use_f16_accum=f16, gemm_type=gemm_type
        )
        dispatched = Sm90H20Heuristics.get_config(
            layer, shape_m, use_f16_accum=f16, gemm_type=gemm_type
        )
        if not fused_e8m0_moe_in_scope(layer, shape_m, gemm_type, False):
            assert dispatched == legacy
            continue
        covered += 1
        decision = select_fused_e8m0_moe(_problem(layer, gemm_type, shape_m, f16))
        assert decision.to_config() == legacy, (
            f"family diverges from legacy for n={layer.shape_n} k={layer.shape_k} "
            f"E={layer.num_experts} isg={layer.input_scale_group_size} "
            f"gemm={gemm_type.value} m={shape_m} f16={f16}:\n"
            f"family: {decision.to_config()}\nlegacy: {legacy}"
        )
        assert dispatched == decision.to_config()
    assert covered >= 100  # the family must actually be exercised by the grid


def test_register_budget_audit_trail():
    """Large-M grouped-input-scale decisions must show the pressured CTA3
    alternative in the audit trail and select an unpressured candidate."""
    layer = _fused_moe_layer(6144, 7168, 48, 128)
    decision = select_fused_e8m0_moe(
        _problem(layer, GemmType.INDEXED, 12288, False)
    )
    assert decision.selected.num_ctas_per_sm == 2
    assert not decision.selected_analysis.register_pressured
    assert "register-budget" in decision.reason
    by_id = {
        analysis.candidate.candidate_id: analysis
        for analysis in decision.considered
    }
    cta3 = by_id["fused_e8m0_moe_cta3"]
    assert cta3.legal
    assert cta3.register_pressured
    assert cta3.register_budget_per_thread == 168  # 65536 // (128 * 3), 8-aligned


def test_masked_and_small_m_keep_occupancy_choice():
    layer = _fused_moe_layer(6144, 7168, 48, 128)
    masked = select_fused_e8m0_moe(
        _problem(layer, GemmType.GROUPED_MASKED, 12288, False)
    )
    assert masked.selected.num_ctas_per_sm == 3
    assert "occupancy" in masked.reason

    small_m = select_fused_e8m0_moe(
        _problem(layer, GemmType.INDEXED, 4098, False)
    )
    assert small_m.selected.num_ctas_per_sm == 3


def test_per_token_scale_keeps_grid_fill_choice():
    layer = _fused_moe_layer(6144, 7168, 48, 0)
    decision = select_fused_e8m0_moe(
        _problem(layer, GemmType.INDEXED, 12288, False)
    )
    assert decision.selected.num_ctas_per_sm == 3
    assert not decision.selected_analysis.register_pressured


def test_grid_fill_cta2_not_demoted_by_register_gate():
    """The register preference replicates the legacy clamp exactly: it only
    demotes from 3 CTAs/SM. A grid-fill CTA2 choice must survive large-M
    grouped-input-scale untouched."""
    layer = _fused_moe_layer(1536, 7168, 8, 128)
    decision = select_fused_e8m0_moe(
        _problem(layer, GemmType.INDEXED, 12288, False)
    )
    assert decision.selected.num_ctas_per_sm == 2
    assert "occupancy" in decision.reason


def test_low_grid_keeps_legacy_construction():
    """8 output tiles: CTA1, the num_warps==4 warp_k reshape, and the num_sms
    trim must all match legacy field by field."""
    layer = _fused_moe_layer(128, 7168, 8, 128)
    config = select_fused_e8m0_moe(
        _problem(layer, GemmType.INDEXED, 12288, False)
    ).to_config()
    legacy = Sm90H20Heuristics._get_config_legacy(
        layer, 12288, use_f16_accum=False, gemm_type=GemmType.INDEXED
    )
    assert config == legacy
    assert config["num_ctas_per_sm"] == 1
    assert config["num_sms"] < _NUM_SMS  # tile-count trim engaged
    assert config["warp_shape"][2] == 64  # num_warps==4 reshape halves warp_k


def test_k_divisibility_halves_block_k():
    """shape_k % 128 != 0 is only a legal problem with per-token input scale
    (a grouped scale must divide shape_k), so the halving loop is exercised
    with isg=0."""
    layer = _fused_moe_layer(6144, 1088, 48, 0)
    config = select_fused_e8m0_moe(
        _problem(layer, GemmType.INDEXED, 12288, False)
    ).to_config()
    assert config["block_shape"][2] == 64
    assert config["warp_shape"][2] == 64


def test_grouped_scale_indivisible_k_routes_to_legacy():
    layer = _fused_moe_layer(6144, 1088, 48, 128)
    assert not fused_e8m0_moe_in_scope(layer, 12288, GemmType.INDEXED, False)


def test_weight_scale_indivisible_k_keeps_parity():
    """A weight scale group that does not divide shape_k is not rejected at
    the problem level; family and legacy must produce the same config."""
    layer = _fused_moe_layer(6144, 1088, 48, 0, weight_scale_group_size=128)
    config = select_fused_e8m0_moe(
        _problem(layer, GemmType.INDEXED, 12288, False)
    ).to_config()
    legacy = Sm90H20Heuristics._get_config_legacy(
        layer, 12288, use_f16_accum=False, gemm_type=GemmType.INDEXED
    )
    assert config == legacy


def test_out_of_contract_k_fails_on_both_paths():
    """shape_k % 64 != 0 leaves no block_k that both divides shape_k and
    nests warp_k=64: legacy trips its assert and the family raises with the
    rejection audit. Neither path returns a config."""
    layer = _fused_moe_layer(6144, 1040, 48, 0)
    with pytest.raises(AssertionError):
        Sm90H20Heuristics._get_config_legacy(
            layer, 12288, gemm_type=GemmType.INDEXED
        )
    with pytest.raises(AssertionError, match="no legal fused-E8M0"):
        select_fused_e8m0_moe(_problem(layer, GemmType.INDEXED, 12288, False))


def test_small_tile_family_covers_decode_shapes():
    """block_m <= 32 (decode-sized m per expert) must dispatch through the
    small-tile selector with the residency alternatives in the audit trail."""
    layer = _fused_moe_layer(6144, 7168, 48, 128)
    legacy = Sm90H20Heuristics._get_config_legacy(
        layer, 96, gemm_type=GemmType.GROUPED_MASKED
    )
    decision = select_fused_e8m0_moe(
        _problem(layer, GemmType.GROUPED_MASKED, 96, False)
    )
    assert decision.family == "fused_e8m0_moe_small_tile"
    assert decision.to_config() == legacy
    assert len(decision.considered) >= 2
    assert decision.reason


def _fused_dense_layer(shape_n, shape_k, input_scale_group_size):
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=32,
        weight_scale_type="group",
        mma_type=MmaType.WGMMA,
    )


_DENSE_SHAPES = (
    (6144, 7168),
    (4096, 4096),
    (1536, 2048),
    (896, 1024),  # short K: no stream-K, deep-pipeline arm reachable
    (1344, 3072),  # 1344 % 128 != 0: block_n fit loop
    (4096, 512),  # short K, wide N
)
_DENSE_M_VALUES = (8, 16, 24, 48, 64, 96, 128, 256, 1024, 4096, 12288, 49152)


def test_dense_family_matches_legacy_across_grid():
    from humming.tune.sm90_h20_families import (
        fused_e8m0_dense_in_scope,
        select_fused_e8m0_dense,
    )

    covered = 0
    arms = set()
    for (shape_n, shape_k), isg, shape_m, f16 in itertools.product(
        _DENSE_SHAPES, (0, 128), _DENSE_M_VALUES, (False, True)
    ):
        layer = _fused_dense_layer(shape_n, shape_k, isg)
        legacy = Sm90H20Heuristics._get_config_legacy(
            layer, shape_m, use_f16_accum=f16, gemm_type=GemmType.DENSE
        )
        dispatched = Sm90H20Heuristics.get_config(
            layer, shape_m, use_f16_accum=f16, gemm_type=GemmType.DENSE
        )
        if not fused_e8m0_dense_in_scope(layer, shape_m, GemmType.DENSE, False):
            assert dispatched == legacy
            continue
        covered += 1
        decision = select_fused_e8m0_dense(
            _problem(layer, GemmType.DENSE, shape_m, f16)
        )
        arms.add(decision.selected.candidate_id)
        assert decision.to_config() == legacy, (
            f"dense family diverges for n={shape_n} k={shape_k} isg={isg} "
            f"m={shape_m} f16={f16}:\n"
            f"family: {decision.to_config()}\nlegacy: {legacy}"
        )
        assert dispatched == decision.to_config()
    assert covered >= 100
    # the grid must exercise all three selection arms
    assert arms == {"dense_plain", "dense_tma_warp_spec", "dense_deep_pipeline"}


def test_dense_scope_negative_boundaries():
    """Constructions the dense family must NOT take (each makes a legacy-only
    branch reachable or the problem illegal); dispatch must equal legacy."""
    from humming.tune.sm90_h20_families import fused_e8m0_dense_in_scope

    def _layer(**overrides):
        fields = dict(
            shape_n=6144,
            shape_k=7168,
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float4e2m1,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.float8e8m0,
            input_scale_group_size=128,
            weight_scale_group_size=32,
            weight_scale_type="group",
            mma_type=MmaType.WGMMA,
        )
        fields.update(overrides)
        return LayerConfig(**fields)

    out_of_scope = (
        # integer B <= 4 bits: _get_small_m_dense_override becomes reachable
        _layer(b_dtype=dtypes.int4),
        # integer activation: legacy int8 block_m alignment special cases
        _layer(a_dtype=dtypes.int8, bs_dtype=dtypes.bfloat16),
        # (packed-K is rejected by LayerConfig itself for fused-E8M0, so the
        # scope check for it cannot be exercised with a real layer)
        # grouped input scale does not divide shape_k
        _layer(shape_k=1088),
    )
    for layer in out_of_scope:
        assert not fused_e8m0_dense_in_scope(layer, 64, GemmType.DENSE, False), (
            layer
        )
        legacy = Sm90H20Heuristics._get_config_legacy(
            layer, 64, gemm_type=GemmType.DENSE
        )
        assert (
            Sm90H20Heuristics.get_config(layer, 64, gemm_type=GemmType.DENSE)
            == legacy
        )

    # fitted warp_n < 16 is outside the legacy contract too (its warp_n
    # assert trips); the family must not claim it, and dispatch fails the
    # same way legacy does.
    tiny_n = _layer(shape_n=32)
    assert not fused_e8m0_dense_in_scope(tiny_n, 64, GemmType.DENSE, False)
    with pytest.raises(AssertionError):
        Sm90H20Heuristics.get_config(tiny_n, 64, gemm_type=GemmType.DENSE)
