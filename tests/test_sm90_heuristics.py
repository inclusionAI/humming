import math

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.device import DeviceInfo
from humming.tune.sm90 import Sm90Heuristics


@pytest.fixture(autouse=True)
def _mock_h200_sm_count(monkeypatch):
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 132))


def _layer(
    shape_n: int,
    shape_k: int,
    *,
    num_experts: int = 12,
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


@pytest.mark.parametrize(
    ("layer", "gemm_type"),
    [
        (
            _layer(
                2880,
                2880,
                input_scale_group_size=32,
                as_dtype=dtypes.float32,
            ),
            GemmType.DENSE,
        ),
        (
            _layer(
                5760,
                2880,
                num_experts=32,
                a_dtype=dtypes.bfloat16,
                as_dtype=None,
                input_scale_group_size=0,
            ),
            GemmType.INDEXED,
        ),
    ],
)
def test_migrated_policies_own_proposal_generation(
    monkeypatch,
    layer,
    gemm_type,
):
    def fail_legacy_helper(cls, *args, **kwargs):
        raise AssertionError("migrated policy called a legacy proposal helper")

    monkeypatch.setattr(
        Sm90Heuristics,
        "get_config1",
        classmethod(fail_legacy_helper),
    )
    monkeypatch.setattr(
        Sm90Heuristics,
        "calc_num_block_list",
        classmethod(fail_legacy_helper),
    )

    decision = Sm90Heuristics.get_tuning_decision(
        layer,
        shape_m=32,
        gemm_type=gemm_type,
    )

    assert decision.selected_analysis.legal


def test_grouped_and_legacy_selection_work_without_a_live_device(monkeypatch):
    def fail_device_query(self):
        raise AssertionError("unexpected live device query")

    monkeypatch.setattr(DeviceInfo, "sm_count", property(fail_device_query))

    grouped = Sm90Heuristics.get_config(
        _layer(6144, 3584, num_experts=0),
        shape_m=32,
        gemm_type=GemmType.DENSE,
    )
    legacy_a16 = Sm90Heuristics.get_config(
        _layer(
            5760,
            2880,
            num_experts=0,
            a_dtype=dtypes.bfloat16,
            as_dtype=None,
            input_scale_group_size=0,
        ),
        shape_m=32,
        gemm_type=GemmType.DENSE,
    )

    assert grouped["block_shape"] == (32, 128, 256)
    assert legacy_a16["block_shape"][1:] == (128, 64)


@pytest.mark.parametrize(
    ("input_scale_group_size", "as_dtype"),
    [(128, dtypes.float32), (32, dtypes.float32)],
)
def test_grouped_fp8_moe_avoids_512_thread_tiles(
    input_scale_group_size,
    as_dtype,
):
    config = Sm90Heuristics.get_config(
        _layer(
            6144,
            3584,
            input_scale_group_size=input_scale_group_size,
            as_dtype=as_dtype,
        ),
        shape_m=64,
        gemm_type=GemmType.INDEXED,
    )

    assert config["warp_shape"][1] == 32
    block_volume = math.prod(config["block_shape"])
    warp_volume = math.prod(config["warp_shape"])
    num_threads = block_volume // warp_volume * 32
    assert num_threads <= 256


def test_grouped_fp8_uses_legal_n_tile_for_non_128_multiple():
    decision = Sm90Heuristics.get_tuning_decision(
        _layer(
            2880,
            2880,
            input_scale_group_size=32,
            as_dtype=dtypes.float32,
        ),
        shape_m=32,
        gemm_type=GemmType.DENSE,
    )
    config = decision.to_config()

    assert decision.selected_analysis.legal
    assert "multi_cast_size_a" not in config
    assert config["block_shape"][1] == 64
    assert config["warp_shape"][1] == 16
    assert config["block_shape"][2] == 64
    assert config["warp_shape"][2] == 64


def test_grouped_fp8_enables_multicast_at_the_dense_threshold():
    layer = _layer(6144, 3584, num_experts=0)

    below_threshold = Sm90Heuristics.get_config(
        layer,
        shape_m=504,
        gemm_type=GemmType.DENSE,
    )
    at_threshold = Sm90Heuristics.get_config(
        layer,
        shape_m=512,
        gemm_type=GemmType.DENSE,
    )

    assert "multi_cast_size_a" not in below_threshold
    assert at_threshold["multi_cast_size_a"] == 2


def test_dense_a16_uses_legal_n_tile_for_non_256_multiple():
    config = Sm90Heuristics.get_config(
        _layer(
            5760,
            2880,
            a_dtype=dtypes.bfloat16,
            as_dtype=None,
            input_scale_group_size=0,
        ),
        shape_m=16,
        gemm_type=GemmType.DENSE,
    )

    assert config["block_shape"][1] == 128
    assert config["warp_shape"][1] == 32


def test_dense_a16_requires_padding_when_n_has_no_wgmma_tile():
    unpadded = _layer(
        2880,
        2880,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        input_scale_group_size=0,
    )
    padded = _layer(
        3072,
        3072,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        input_scale_group_size=0,
    )

    with pytest.raises(AssertionError):
        Sm90Heuristics.get_config(
            unpadded,
            shape_m=16,
            gemm_type=GemmType.DENSE,
        )

    config = Sm90Heuristics.get_config(
        padded,
        shape_m=16,
        gemm_type=GemmType.DENSE,
    )
    assert config["block_shape"][1:] == (128, 64)
    assert config["warp_shape"][1:] == (32, 64)


def test_short_k_does_not_leave_warp_k_larger_than_block_k():
    config = Sm90Heuristics.get_config(
        LayerConfig(
            shape_n=128,
            shape_k=64,
            a_dtype=dtypes.int8,
            b_dtype=dtypes.uint7,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.bfloat16,
            mma_type=MmaType.WGMMA,
        ),
        shape_m=64,
        gemm_type=GemmType.DENSE,
    )

    assert config["block_shape"][2] == 64
    assert config["warp_shape"][2] == 64


@pytest.mark.parametrize(
    ("routed_m", "expected_ctas", "expected_stream_k"),
    [(2, 1, True), (4, 2, False), (8, 3, False), (256, 3, False)],
)
def test_mxfp4_a16_indexed_preserves_warp_k_and_fits_grid(
    routed_m,
    expected_ctas,
    expected_stream_k,
):
    config = Sm90Heuristics.get_config(
        _layer(
            5760,
            2880,
            num_experts=32,
            a_dtype=dtypes.bfloat16,
            as_dtype=None,
            input_scale_group_size=0,
        ),
        shape_m=routed_m,
        gemm_type=GemmType.INDEXED,
    )

    assert config["block_shape"] == (8, 128, 64)
    assert config["warp_shape"] == (8, 32, 64)
    assert config["num_ctas_per_sm"] == expected_ctas
    assert config["use_stream_k"] is expected_stream_k


@pytest.mark.parametrize("shape_k", [512, 256])
def test_mxfp4_a16_short_k_limits_wide_n_tile_before_block_m48(shape_k):
    layer = _layer(
        6144,
        shape_k,
        num_experts=256,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        input_scale_group_size=0,
    )

    small = Sm90Heuristics.get_config(
        layer,
        shape_m=6144,
        gemm_type=GemmType.INDEXED,
    )
    large = Sm90Heuristics.get_config(
        layer,
        shape_m=8192,
        gemm_type=GemmType.INDEXED,
    )

    assert small["block_shape"] == (32, 512, 64)
    assert small["num_ctas_per_sm"] == 2
    assert large["block_shape"] == (48, 256, 64)
    assert large["num_ctas_per_sm"] == 2


def test_nvfp4_a16_uses_narrow_first_wave_only():
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
    below_first_wave = Sm90Heuristics.get_config(
        layer,
        shape_m=4,
        gemm_type=GemmType.INDEXED,
    )
    first_wave = Sm90Heuristics.get_config(
        layer,
        shape_m=8,
        gemm_type=GemmType.INDEXED,
    )
    next_wave = Sm90Heuristics.get_config(
        layer,
        shape_m=16,
        gemm_type=GemmType.INDEXED,
    )

    assert below_first_wave["block_shape"] == (8, 256, 128)
    assert below_first_wave["warp_shape"] == (8, 32, 64)
    assert below_first_wave["num_ctas_per_sm"] == 1
    assert below_first_wave["use_stream_k"] is True
    assert first_wave["block_shape"] == (8, 128, 128)
    assert first_wave["warp_shape"] == (8, 32, 64)
    assert first_wave["num_ctas_per_sm"] == 3
    assert next_wave["block_shape"] == (8, 256, 128)
    assert next_wave["warp_shape"] == (8, 64, 64)


def test_fp4_a16_dense_uses_small_tile_only_through_m128():
    layer = _layer(
        5376,
        2688,
        num_experts=0,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        bs_dtype=dtypes.float8e4m3,
        input_scale_group_size=0,
        weight_scale_group_size=16,
    )
    small = Sm90Heuristics.get_config(layer, shape_m=128, gemm_type=GemmType.DENSE)
    large = Sm90Heuristics.get_config(layer, shape_m=256, gemm_type=GemmType.DENSE)

    assert small["block_shape"][1:] == (128, 64)
    assert small["warp_shape"][1:] == (32, 64)
    assert small["num_ctas_per_sm"] == 2
    assert large["block_shape"][1:] != (128, 64)
