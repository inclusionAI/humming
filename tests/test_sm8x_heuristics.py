import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import DeviceInfo
from humming.tune.sm8x import Sm80Heuristics, Sm86Heuristics, Sm89Heuristics
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


def _make_nvfp4_layer_config(shape_n, shape_k, num_experts):
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e4m3,
        weight_scale_group_size=16,
    )


@pytest.mark.parametrize("shape_n, shape_k", [(1280, 2560), (2560, 640), (1536, 2048)])
@pytest.mark.parametrize("shape_m", [80, 160, 1280, 5120])
def test_sm80_memory_bound_moe_uses_more_ctas(shape_n, shape_k, shape_m):
    layer_config = _make_nvfp4_layer_config(shape_n, shape_k, num_experts=512)

    config = Sm80Heuristics.get_config(layer_config, shape_m=shape_m, gemm_type=GemmType.INDEXED)

    smem_size = estimate_smem_size_layer(
        layer_config,
        config["block_shape"],
        GemmType.INDEXED,
        config["num_stages"],
    )
    assert config["num_ctas_per_sm"] > 1
    assert config["num_stages"] >= 3
    assert shape_k % config["block_shape"][2] == 0
    assert smem_size * config["num_ctas_per_sm"] <= Sm80Heuristics.max_smem_size


@pytest.mark.parametrize(
    "num_experts, shape_m, gemm_type",
    [
        (512, 20480, GemmType.INDEXED),
        (512, 32768, GemmType.INDEXED),
        (0, 16, GemmType.DENSE),
        (0, 1024, GemmType.DENSE),
    ],
)
def test_sm80_moe_occupancy_keeps_other_configs(monkeypatch, num_experts, shape_m, gemm_type):
    layer_config = _make_nvfp4_layer_config(1280, 2560, num_experts=num_experts)

    config = Sm80Heuristics.get_config(layer_config, shape_m=shape_m, gemm_type=gemm_type)
    monkeypatch.setattr(Sm80Heuristics, "moe_occupancy_warps_per_sm", 0)
    baseline_config = Sm80Heuristics.get_config(layer_config, shape_m=shape_m, gemm_type=gemm_type)

    assert config == baseline_config
