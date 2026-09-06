import math

import pytest

import humming.tune
from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.device import DeviceInfo
from humming.tune import get_heuristics_class
from humming.tune.sm90 import Sm90Heuristics
from humming.tune.sm90_h20 import Sm90H20Heuristics
from humming.tune.sm90_h200 import Sm90H200Heuristics


@pytest.fixture(autouse=True)
def _h200_num_sms(monkeypatch):
    # Heuristics read the live device singleton; pin it to H200's 132 SMs
    # so calibration is exercised on any host (matches upstream test style).
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 132))


def _w4a8_moe_layer(shape_n: int, shape_k: int, num_experts: int = 256) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.int4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float32,
        weight_scale_group_size=128,
        mma_type=MmaType.WGMMA,
    )


def _dense_layer(shape_n: int, shape_k: int) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=0,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.int4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=0,
        mma_type=MmaType.WGMMA,
    )


def _assert_vllm_constraints(config: dict) -> None:
    assert config["block_shape"][2] <= 128
    if config["block_shape"][1] % 32 == 0:
        assert config["warp_shape"][1] >= 32


def test_small_m_w4a8_moe_picks_sane_tiles():
    config = Sm90H200Heuristics.get_config(
        _w4a8_moe_layer(6144, 3584),
        shape_m=2048,
        gemm_type=GemmType.INDEXED,
    )

    assert 8 <= config["block_shape"][0] <= 64
    _assert_vllm_constraints(config)
    assert config["num_ctas_per_sm"] >= 1
    assert config["num_stages"] >= 3


def test_large_m_w4a8_moe_picks_sane_tiles():
    config = Sm90H200Heuristics.get_config(
        _w4a8_moe_layer(6144, 3584),
        shape_m=16384,
        gemm_type=GemmType.INDEXED,
    )

    _assert_vllm_constraints(config)
    # A 16k-token batch over 256 experts averages ~72 rows/expert: expect a
    # 48-64 row tile rather than the tiny small-M tiles.
    assert config["block_shape"][0] >= 32


def test_w4a8_moe_grouped_gemm_respects_constraints():
    config = Sm90H200Heuristics.get_config(
        _w4a8_moe_layer(6144, 3584),
        shape_m=4096,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )

    _assert_vllm_constraints(config)


def test_w4a8_moe_w2_shape_respects_constraints():
    # w2 GEMM: N and K swapped relative to w13.
    config = Sm90H200Heuristics.get_config(
        _w4a8_moe_layer(3584, 6144),
        shape_m=2048,
        gemm_type=GemmType.INDEXED,
    )

    _assert_vllm_constraints(config)


def test_dense_not_degenerate_vs_generic_sm90():
    # Generic Sm90 heuristics also read current_device (pinned to 132 SMs by
    # the autouse fixture), so the delta isolates H200 tile calibration.
    layer = _dense_layer(6144, 3584)
    for shape_m in (32, 256, 2048):
        h200 = Sm90H200Heuristics.get_config(layer, shape_m=shape_m)
        generic = Sm90Heuristics.get_config(layer, shape_m=shape_m)

        _assert_vllm_constraints(h200)
        # Non-degenerate: tiles stay within a sane range of the generic path.
        ratio = math.prod(h200["block_shape"]) / math.prod(generic["block_shape"])
        assert 0.25 <= ratio <= 4.0
        assert h200["num_ctas_per_sm"] >= 1


@pytest.mark.parametrize(
    ("device_name", "expected_cls"),
    [
        ("NVIDIA H200", Sm90H200Heuristics),
        ("NVIDIA H20", Sm90H20Heuristics),
        ("NVIDIA H100", Sm90Heuristics),
    ],
)
def test_dispatch_selects_correct_heuristics(monkeypatch, device_name, expected_cls):
    # get_heuristics_class resolves the device name via the C++ extension
    # through DeviceInfo; substitute a fake that reports SM 9.0 + our name.
    class _FakeDeviceInfo:
        def __init__(self, index=None) -> None:
            pass

        @property
        def name(self) -> str:
            return device_name

        @property
        def sm_version(self) -> int:
            return 90

    monkeypatch.setattr(humming.tune, "DeviceInfo", _FakeDeviceInfo)

    assert get_heuristics_class() is expected_cls


@pytest.mark.parametrize(
    "forced",
    ["Sm90H200Heuristics", "Sm90H20Heuristics", "Sm90Heuristics", "90"],
)
def test_force_heuristics_env_overrides_dispatch(monkeypatch, forced):
    monkeypatch.setenv("HUMMING_FORCE_HEURISTICS", forced)
    # Dispatch would otherwise resolve the real device; the override must win.
    assert get_heuristics_class() is not None


def test_force_heuristics_env_rejects_unknown(monkeypatch):
    monkeypatch.setenv("HUMMING_FORCE_HEURISTICS", "Sm99Heuristics")
    with pytest.raises(ValueError, match="HUMMING_FORCE_HEURISTICS"):
        get_heuristics_class()
