import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import DeviceInfo
from humming.tune.sm90_h20 import Sm90H20Heuristics


@pytest.fixture(autouse=True)
def _h20_num_sms(monkeypatch):
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 78))


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
        sm_version=90,
    )


def _mxfp4_layer(shape_n: int, shape_k: int) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=96,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        weight_scale_group_size=32,
        sm_version=90,
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


@pytest.mark.parametrize("shape_m", [3072, 12288, 24576])
def test_large_mxfp4_grouped_contiguous_uses_three_resident_ctas(shape_m):
    config = Sm90H20Heuristics.get_config(
        _mxfp4_layer(4608, 5120),
        shape_m,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )

    assert config["block_shape"] == (48, 128, 64)
    assert config["warp_shape"] == (48, 32, 64)
    assert config["num_ctas_per_sm"] == 3
    assert not config["use_stream_k"]


def test_small_mxfp4_grouped_contiguous_keeps_small_expert_tiles():
    config = Sm90H20Heuristics.get_config(
        _mxfp4_layer(4608, 5120),
        2048,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )

    assert config["block_shape"][0] == 32
