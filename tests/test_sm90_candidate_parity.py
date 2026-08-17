import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.tune.sm90 import Sm90Heuristics

_SHAPE_M_CANDIDATES = (1, 2, 4, *range(8, 257, 8))


@pytest.fixture(autouse=True)
def _mock_h200_sm_count(monkeypatch):
    monkeypatch.setattr(
        Sm90Heuristics,
        "get_num_sms",
        classmethod(lambda cls: 132),
    )


def _layer(
    shape_n: int,
    shape_k: int,
    *,
    num_experts: int,
    a_dtype=dtypes.float8e4m3,
    as_dtype=dtypes.float32,
    bs_dtype=dtypes.float8e8m0,
    input_scale_group_size: int = 128,
    weight_scale_group_size: int = 32,
) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=a_dtype,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        as_dtype=as_dtype,
        bs_dtype=bs_dtype,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        mma_type=MmaType.WGMMA,
    )


def test_offline_grouped_and_legacy_selection_do_not_query_grid_size(
    monkeypatch,
):
    def fail_grid_query(cls):
        raise AssertionError("unexpected live device query")

    monkeypatch.setattr(
        Sm90Heuristics,
        "get_num_sms",
        classmethod(fail_grid_query),
    )
    grouped = _layer(6144, 3584, num_experts=0)
    a16 = _layer(
        5760,
        2880,
        num_experts=0,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        input_scale_group_size=0,
    )

    grouped_config = Sm90Heuristics.get_config(
        grouped,
        shape_m=32,
        gemm_type=GemmType.DENSE,
    )
    a16_config = Sm90Heuristics.get_config(
        a16,
        shape_m=32,
        gemm_type=GemmType.DENSE,
    )

    assert grouped_config["block_shape"] == (32, 128, 256)
    assert a16_config["block_shape"][1:] == (128, 64)


def _grouped_a8_config(
    block_m: int,
    block_n: int,
    block_k: int,
    warp_n: int,
    *,
    warp_k: int = 128,
    dense: bool = False,
    multicast: bool = False,
) -> dict:
    config = {
        "block_shape": (block_m, block_n, block_k),
        "warp_shape": (block_m, warp_n, warp_k),
        "use_stream_k": True,
        "use_f16_accum": False,
        "num_stages": 4,
    }
    if dense:
        config |= {
            "use_warp_spec": True,
            "use_tma": True,
            "use_mbarrier": True,
        }
    if multicast:
        config["multi_cast_size_a"] = 2
    return config


def _indexed_a16_config(
    block_shape: tuple[int, int, int],
    warp_shape: tuple[int, int, int],
    num_ctas_per_sm: int,
    *,
    use_stream_k: bool,
) -> dict:
    return {
        "block_shape": block_shape,
        "warp_shape": warp_shape,
        "use_stream_k": use_stream_k,
        "use_f16_accum": False,
        "num_stages": 4,
        "num_ctas_per_sm": num_ctas_per_sm,
    }


def _assert_interval_agreement(
    layer: LayerConfig,
    gemm_type: GemmType,
    intervals: list[list[int | dict]],
) -> None:
    for shape_m in _SHAPE_M_CANDIDATES:
        interval_config = next(
            config for lower, upper, config in intervals if lower < shape_m <= upper
        )
        assert interval_config == Sm90Heuristics.get_config(
            layer,
            shape_m=shape_m,
            gemm_type=gemm_type,
        )


def _grouped_intervals(
    rows,
    *,
    block_n: int,
    warp_n: int,
    warp_k: int = 128,
    dense: bool = False,
) -> list[list[int | dict]]:
    return [
        [
            lower,
            upper,
            _grouped_a8_config(
                block_m,
                block_n,
                block_k,
                warp_n,
                warp_k=warp_k,
                dense=dense,
                multicast=multicast,
            ),
        ]
        for lower, upper, block_m, block_k, multicast in rows
    ]


def _indexed_intervals(rows) -> list[list[int | dict]]:
    return [
        [
            lower,
            upper,
            _indexed_a16_config(
                block_shape,
                warp_shape,
                num_ctas_per_sm,
                use_stream_k=use_stream_k,
            ),
        ]
        for (
            lower,
            upper,
            block_shape,
            warp_shape,
            num_ctas_per_sm,
            use_stream_k,
        ) in rows
    ]


_SCHEDULE_CASES = [
    (
        "grouped-a8-n128",
        _layer(6144, 3584, num_experts=12),
        GemmType.INDEXED,
        _grouped_intervals(
            [
                (0, 32, 8, 256, False),
                (32, 96, 16, 256, False),
                (96, 152, 24, 256, False),
                (152, 216, 32, 256, False),
                (216, 272, 40, 128, False),
            ],
            block_n=128,
            warp_n=32,
        ),
    ),
    (
        "grouped-a8-n64-k64",
        _layer(
            2880,
            2880,
            num_experts=12,
            as_dtype=dtypes.float32,
            input_scale_group_size=32,
        ),
        GemmType.DENSE,
        _grouped_intervals(
            [
                (0, 32, 8, 64, False),
                (32, 96, 16, 64, False),
                (96, 152, 24, 64, False),
                (152, 216, 32, 64, False),
                (216, 272, 40, 64, False),
            ],
            block_n=64,
            warp_n=16,
            warp_k=64,
            dense=True,
        ),
    ),
    (
        "mxfp4-a16-indexed",
        _layer(
            5760,
            2880,
            num_experts=32,
            a_dtype=dtypes.bfloat16,
            as_dtype=None,
            input_scale_group_size=0,
        ),
        GemmType.INDEXED,
        _indexed_intervals(
            [
                (0, 2, (8, 128, 64), (8, 32, 64), 1, True),
                (2, 4, (8, 128, 64), (8, 32, 64), 2, False),
                (4, 256, (8, 128, 64), (8, 32, 64), 3, False),
            ]
        ),
    ),
    (
        "nvfp4-a16-indexed",
        _layer(
            5376,
            2688,
            num_experts=128,
            a_dtype=dtypes.bfloat16,
            as_dtype=None,
            bs_dtype=dtypes.float8e4m3,
            input_scale_group_size=0,
            weight_scale_group_size=16,
        ),
        GemmType.INDEXED,
        _indexed_intervals(
            [
                (0, 4, (8, 256, 128), (8, 32, 64), 1, True),
                (4, 8, (8, 128, 128), (8, 32, 64), 3, False),
                (8, 1024, (8, 256, 128), (8, 64, 64), 2, False),
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    ("layer", "gemm_type", "expected"),
    [
        pytest.param(layer, gemm_type, expected, id=name)
        for name, layer, gemm_type, expected in _SCHEDULE_CASES
    ],
)
def test_sm90_interval_and_lookup_parity_through_m256(
    layer,
    gemm_type,
    expected,
):
    intervals = Sm90Heuristics.get_configs(layer, gemm_type=gemm_type)

    assert [entry for entry in intervals if entry[0] < 256] == expected
    _assert_interval_agreement(layer, gemm_type, intervals)


@pytest.mark.parametrize(
    ("layer", "gemm_type"),
    [
        pytest.param(layer, gemm_type, id=name)
        for name, layer, gemm_type, _ in _SCHEDULE_CASES
    ],
)
def test_sm90_selected_tiles_are_nested_and_aligned(layer, gemm_type):
    for shape_m in _SHAPE_M_CANDIDATES:
        config = Sm90Heuristics.get_config(
            layer,
            shape_m=shape_m,
            gemm_type=gemm_type,
        )
        block_shape = config["block_shape"]
        warp_shape = config["warp_shape"]

        assert layer.shape_n % block_shape[1] == 0
        assert layer.shape_k % block_shape[2] == 0
        assert all(
            block % warp == 0
            for block, warp in zip(block_shape, warp_shape, strict=True)
        )
        assert (block_shape[1] // warp_shape[1]) % 4 == 0


def test_grouped_scale_decision_explains_legality_fallbacks():
    decision = Sm90Heuristics.get_tuning_decision(
        _layer(
            2880,
            2880,
            num_experts=12,
            as_dtype=dtypes.float32,
            input_scale_group_size=32,
        ),
        shape_m=32,
        gemm_type=GemmType.DENSE,
    )

    assert decision.family == "grouped_scale"
    assert decision.selected.candidate_id == "grouped_scale_n64_k64_direct"
    assert decision.selected_analysis.legal
    multicast = next(
        analysis
        for analysis in decision.considered
        if analysis.candidate.candidate_id == "grouped_scale_n64_k64_multicast"
    )
    assert any(
        "block_n * multi_cast_size_a" in reason
        for reason in multicast.rejection_reasons
    )


def test_grouped_scale_selects_multicast_only_at_the_dense_threshold():
    layer = _layer(6144, 3584, num_experts=0)

    direct = Sm90Heuristics.get_tuning_decision(
        layer,
        shape_m=504,
        gemm_type=GemmType.DENSE,
    )
    multicast = Sm90Heuristics.get_tuning_decision(
        layer,
        shape_m=512,
        gemm_type=GemmType.DENSE,
    )

    assert direct.selected.candidate_id == "grouped_scale_n128_k128_direct"
    assert multicast.selected.candidate_id == ("grouped_scale_n128_k128_multicast")
    assert "multi_cast_size_a" not in direct.to_config()
    assert multicast.to_config()["multi_cast_size_a"] == 2


def test_indexed_a16_decision_records_residency_transforms():
    layer = _layer(
        5376,
        2688,
        num_experts=128,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        bs_dtype=dtypes.float8e4m3,
        input_scale_group_size=0,
        weight_scale_group_size=16,
    )
    decision = Sm90Heuristics.get_tuning_decision(
        layer,
        shape_m=8,
        gemm_type=GemmType.INDEXED,
    )

    assert decision.family == "indexed_a16"
    assert decision.selected.candidate_id == "indexed_a16_split_n_widen_k"
    assert [analysis.candidate.candidate_id for analysis in decision.considered] == [
        "indexed_a16_base",
        "indexed_a16_half_k",
        "indexed_a16_split_n_widen_k",
    ]
    assert "increased CTA residency" in decision.reason
    assert "without adding a grid wave" in decision.reason
