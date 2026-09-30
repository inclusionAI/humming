import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import DeviceInfo
from humming.tune.sm8x import Sm86Heuristics, Sm89Heuristics
from humming.utils.smem import estimate_smem_size_layer


@pytest.fixture(autouse=True)
def _mock_rtx3080_device(monkeypatch):
    tensorcore_tops = {"float16": 61.1, "bfloat16": 61.1, "int8": 244.4, "int4": 488.8}
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 68))
    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: 86))
    monkeypatch.setattr(DeviceInfo, "memory_bandwidth_gbps", property(lambda self: 760.0))
    monkeypatch.setattr(DeviceInfo, "tensorcore_tops", property(lambda self: tensorcore_tops))


@pytest.mark.parametrize("heuristics_cls", [Sm86Heuristics, Sm89Heuristics])
@pytest.mark.parametrize("a_dtype", [dtypes.float16, dtypes.bfloat16])
@pytest.mark.parametrize("b_dtype", [dtypes.int8, dtypes.float8e4m3, dtypes.int4])
@pytest.mark.parametrize("weight_scale_group_size", [0, 32, 128])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("shape_m", [1, 1024, 16384])
def test_a16_config_fits_in_smem(
    heuristics_cls, a_dtype, b_dtype, weight_scale_group_size, has_bias, shape_m
):
    layer_config = LayerConfig(
        shape_n=4096,
        shape_k=4096,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=a_dtype,
        bs_dtype=a_dtype,
        weight_scale_group_size=weight_scale_group_size,
        has_bias=has_bias,
    )

    config = heuristics_cls.get_config(layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE)

    smem_size = estimate_smem_size_layer(
        layer_config,
        config["block_shape"],
        GemmType.DENSE,
        config["num_stages"],
    )
    assert smem_size * config["num_ctas_per_sm"] <= heuristics_cls.max_smem_size
