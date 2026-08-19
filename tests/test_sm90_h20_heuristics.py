import dataclasses

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.tune.sm90_h20 import Sm90H20Heuristics


@pytest.fixture(autouse=True)
def _h20_num_sms(monkeypatch):
    monkeypatch.setattr(Sm90H20Heuristics, "get_num_sms", classmethod(lambda cls: 78))


def _layer(shape_n: int, shape_k: int, num_experts: int = 0) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.int4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=0,
        mma_type=MmaType.WGMMA,
    )


def _shared_layer(shape_n: int, shape_k: int, num_experts: int) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=0,
        weight_scale_group_size=32,
        mma_type=MmaType.WGMMA,
        use_shared_e8m0_scale_storage=True,
    )


def test_short_k_dense_uses_small_m_tiles_without_tma():
    config = Sm90H20Heuristics.get_config(_layer(512, 512), 64)

    assert config["block_shape"][0] == 8
    assert not config.get("use_tma", False)


@pytest.mark.parametrize(
    ("shape_k", "use_tma"),
    [(1024, False), (1536, True)],
)
def test_dense_tma_requires_enough_pipeline_turns(shape_k, use_tma):
    config = Sm90H20Heuristics.get_config(_layer(8192, shape_k), 256)

    assert config.get("use_tma", False) is use_tma


def test_moderate_stream_k_keeps_four_stages():
    config = Sm90H20Heuristics.get_config(_layer(2048, 2048), 8)

    assert config["use_stream_k"]
    assert config["num_stages"] == 4


def test_long_k_balances_m_reuse_and_stream_k_grid():
    config = Sm90H20Heuristics.get_config(_layer(512, 8192), 32)

    assert config["block_shape"][0] == 16
    assert config["num_stages"] == 4
    assert config["num_sms"] >= 64


@pytest.mark.parametrize(
    ("shape_n", "block_n"),
    [(512, 256), (1024, 512)],
)
def test_sparse_long_k_moe_keeps_two_n_tiles_per_expert(shape_n, block_n):
    config = Sm90H20Heuristics.get_config(
        _layer(shape_n, 7168, num_experts=256),
        8,
        gemm_type=GemmType.INDEXED,
    )

    assert config["block_shape"] == (8, block_n, 64)
    assert config["num_stages"] == 3


def test_shared_e8m0_large_m_long_k_disables_stream_k():
    shared = Sm90H20Heuristics.get_config(
        _shared_layer(768, 3584, num_experts=896),
        131072,
        gemm_type=GemmType.INDEXED,
    )
    native = Sm90H20Heuristics.get_config(
        dataclasses.replace(
            _shared_layer(768, 3584, num_experts=896),
            use_shared_e8m0_scale_storage=False,
        ),
        131072,
        gemm_type=GemmType.INDEXED,
    )

    assert shared["block_shape"][0] >= 48
    assert not shared["use_stream_k"]
    assert native["use_stream_k"]


def test_shared_e8m0_large_m_short_k_uses_cta4_schedule():
    shared = Sm90H20Heuristics.get_config(
        _shared_layer(3584, 384, num_experts=896),
        131072,
        gemm_type=GemmType.INDEXED,
    )

    block_m = shared["block_shape"][0]
    assert block_m >= 48
    assert shared["block_shape"][1] == 128
    assert shared["warp_shape"] == (block_m, 32, shared["block_shape"][2])
    assert shared["num_stages"] == 3
    assert shared["num_ctas_per_sm"] == 4
    assert not shared["use_stream_k"]
    assert shared["use_tma"]
    assert shared["use_mbarrier"]
    assert not shared["use_tma_a"]
    assert not shared["use_tma_c"]
    assert shared["num_sms"] >= 3072


def test_shared_e8m0_very_short_k_keeps_native_schedule_with_grid_bump():
    # DSV4-Flash W2 (K=256) regresses under the CTA4 schedule; only the grid
    # floor applies below the measured K window.
    shared = Sm90H20Heuristics.get_config(
        _shared_layer(4096, 256, num_experts=256),
        49152,
        gemm_type=GemmType.INDEXED,
    )
    native = Sm90H20Heuristics.get_config(
        dataclasses.replace(
            _shared_layer(4096, 256, num_experts=256),
            use_shared_e8m0_scale_storage=False,
        ),
        49152,
        gemm_type=GemmType.INDEXED,
    )

    assert shared["block_shape"][0] >= 48
    assert shared["num_sms"] >= 3072
    assert shared["block_shape"] == native["block_shape"]
    assert shared["num_stages"] == native["num_stages"]
    assert shared["num_ctas_per_sm"] == native["num_ctas_per_sm"]


def test_shared_e8m0_small_m_keeps_existing_schedule():
    config = Sm90H20Heuristics.get_config(
        _shared_layer(768, 3584, num_experts=896),
        2048,
        gemm_type=GemmType.INDEXED,
    )

    assert config["block_shape"][0] == 8
    assert config["use_stream_k"]


def test_shared_e8m0_explicit_path_keeps_native_large_m_schedule():
    shared_explicit = Sm90H20Heuristics.get_config(
        dataclasses.replace(
            _shared_layer(768, 3584, num_experts=896),
            use_fused_e8m0_scale=False,
        ),
        131072,
        gemm_type=GemmType.INDEXED,
    )
    native_explicit = Sm90H20Heuristics.get_config(
        dataclasses.replace(
            _shared_layer(768, 3584, num_experts=896),
            use_fused_e8m0_scale=False,
            use_shared_e8m0_scale_storage=False,
        ),
        131072,
        gemm_type=GemmType.INDEXED,
    )

    assert shared_explicit == native_explicit
    assert shared_explicit["use_stream_k"]


def _auto_single(layer, shape_m):
    from humming.tune import _shared_auto_single_config

    bands = Sm90H20Heuristics.shared_e8m0_auto_fused_bands[(layer.shape_n, layer.shape_k, layer.num_experts)]
    _, config = _shared_auto_single_config(
        Sm90H20Heuristics,
        layer,
        shape_m=shape_m,
        use_f16_accum=False,
        use_batch_invariant=False,
        gemm_type=GemmType.INDEXED,
        fused_bands=bands,
    )
    return config


def test_shared_e8m0_auto_picks_explicit_for_decode_m():
    config = _auto_single(_shared_layer(3584, 384, num_experts=896), 2048)

    assert config["fuse_e8m0_scale"] is False
    assert config["block_shape"][0] < 48


def test_shared_e8m0_auto_picks_fused_cta4_for_large_m():
    config = _auto_single(_shared_layer(3584, 384, num_experts=896), 131072)

    assert config["fuse_e8m0_scale"] is True
    assert config["num_ctas_per_sm"] == 4
    assert config["num_stages"] == 3
    assert not config["use_stream_k"]


def _auto_table(layer):
    from humming.tune import _shared_auto_config_table

    return _shared_auto_config_table(
        Sm90H20Heuristics,
        layer,
        use_f16_accum=False,
        use_batch_invariant=False,
        gemm_type=GemmType.INDEXED,
        fused_bands=Sm90H20Heuristics.shared_e8m0_auto_fused_bands[
            (layer.shape_n, layer.shape_k, layer.num_experts)
        ],
    )


def _table_path(table, routed_m):
    for lo, hi, config in table:
        if lo < routed_m <= hi:
            return config["fuse_e8m0_scale"]
    raise AssertionError(routed_m)


def test_shared_e8m0_auto_w2_table_flips_at_measured_band():
    table = _auto_table(_shared_layer(3584, 384, num_experts=896))

    for (_, prev_hi, _), (lo, _, _) in zip(table, table[1:], strict=False):
        assert prev_hi == lo
    assert all("fuse_e8m0_scale" in config for _, _, config in table)
    assert _table_path(table, 2048) is False
    assert _table_path(table, 25728) is False
    assert _table_path(table, 25729) is True
    assert _table_path(table, 131072) is True


def test_shared_e8m0_auto_w13_table_has_fused_pocket():
    table = _auto_table(_shared_layer(768, 3584, num_experts=896))

    assert _table_path(table, 2048) is False
    assert _table_path(table, 11456) is False
    assert _table_path(table, 11457) is True
    assert _table_path(table, 14336) is True
    assert _table_path(table, 14337) is False
    assert _table_path(table, 25728) is False
    assert _table_path(table, 25729) is True
    assert _table_path(table, 131072) is True


def test_shared_e8m0_auto_only_engages_for_measured_shapes(monkeypatch):
    import humming.tune as tune

    monkeypatch.setattr(tune, "get_heuristics_class", lambda *a, **k: Sm90H20Heuristics)
    tune.get_heuristics_config.cache_clear()
    try:
        layer = _shared_layer(3584, 384, num_experts=896)

        auto = tune.get_heuristics_config(layer, 131072, gemm_type=GemmType.INDEXED)
        assert auto["fuse_e8m0_scale"] is True

        forced = tune.get_heuristics_config(layer, 131072, fuse_e8m0_scale=False, gemm_type=GemmType.INDEXED)
        assert "fuse_e8m0_scale" not in forced

        table = tune.get_heuristics_config(layer, gemm_type=GemmType.INDEXED)
        assert all("fuse_e8m0_scale" in config for _, _, config in table)

        # Unmeasured shared shape: AUTO stays off, plain table, no flags.
        other = tune.get_heuristics_config(
            _shared_layer(4096, 256, num_experts=256), gemm_type=GemmType.INDEXED
        )
        assert all("fuse_e8m0_scale" not in config for _, _, config in other)

        # Unmeasured gemm type on a measured shape: AUTO stays off too.
        dense_like = tune.get_heuristics_config(layer, gemm_type=GemmType.GROUPED_CONTIGUOUS)
        assert all("fuse_e8m0_scale" not in config for _, _, config in dense_like)
    finally:
        tune.get_heuristics_config.cache_clear()
