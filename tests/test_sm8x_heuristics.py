import math

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import DeviceInfo
from humming.tune.sm8x import Sm80Heuristics, Sm86Heuristics, Sm87Heuristics, Sm89Heuristics
from humming.tune.sm100 import Sm100Heuristics
from humming.utils.smem import estimate_smem_size_layer


@pytest.fixture
def _mock_rtx3080_device(monkeypatch):
    tensorcore_tops = {"float16": 61.1, "bfloat16": 61.1, "int8": 244.4, "int4": 488.8}
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 68))
    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: 86))
    monkeypatch.setattr(DeviceInfo, "memory_bandwidth_gbps", property(lambda self: 760.0))
    monkeypatch.setattr(DeviceInfo, "tensorcore_tops", property(lambda self: tensorcore_tops))


@pytest.mark.usefixtures("_mock_rtx3080_device")
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


SM80_DEVICES = {
    # name: (sm_count, memory bandwidth GB/s, FP16 tensor core TFLOPS)
    "sm80-108sm": (108, 2039.0, 311.9),
    "sm80-56sm": (56, 933.0, 165.1),
}


@pytest.fixture(params=list(SM80_DEVICES))
def sm80_device(request, monkeypatch):
    sm_count, memory_bandwidth, fp16_tops = SM80_DEVICES[request.param]
    tensorcore_tops = {
        "float16": fp16_tops,
        "bfloat16": fp16_tops,
        "int8": 2 * fp16_tops,
        "int4": 4 * fp16_tops,
    }
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: sm_count))
    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: 80))
    monkeypatch.setattr(DeviceInfo, "memory_bandwidth_gbps", property(lambda self: memory_bandwidth))
    monkeypatch.setattr(DeviceInfo, "tensorcore_tops", property(lambda self: tensorcore_tops))
    monkeypatch.setattr(DeviceInfo, "max_registers_per_sm", property(lambda self: 65536))
    return sm_count


WEIGHT_FORMATS = {
    "nvfp4": dict(b_dtype=dtypes.float4e2m1, bs_dtype=dtypes.float8e4m3, weight_scale_group_size=16),
    "mxfp4": dict(b_dtype=dtypes.float4e2m1, bs_dtype=dtypes.float8e8m0, weight_scale_group_size=32),
    "uint4-g128": dict(b_dtype=dtypes.uint4, bs_dtype=dtypes.bfloat16, weight_scale_group_size=128),
    "uint8-channel": dict(b_dtype=dtypes.uint8, bs_dtype=dtypes.bfloat16),
    "uint2-g128": dict(b_dtype=dtypes.uint2, bs_dtype=dtypes.bfloat16, weight_scale_group_size=128),
}


def _make_moe_layer_config(shape_n, shape_k, num_experts, weight_format="nvfp4", a_dtype=dtypes.bfloat16):
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=a_dtype,
        c_dtype=dtypes.bfloat16,
        **WEIGHT_FORMATS[weight_format],
    )


def _get_config_without_rule(monkeypatch, heuristics_cls, layer_config, shape_m, **kwargs):
    with monkeypatch.context() as patch:
        patch.setattr(heuristics_cls, "moe_occupancy_warps_per_sm", 0)
        return heuristics_cls.get_config(layer_config, shape_m=shape_m, **kwargs)


def _get_warps_per_cta(config):
    block_shape, warp_shape = config["block_shape"], config["warp_shape"]
    return math.prod(block // warp for block, warp in zip(block_shape, warp_shape, strict=True))


@pytest.mark.parametrize("weight_format", ["nvfp4", "mxfp4", "uint4-g128", "uint8-channel"])
@pytest.mark.parametrize("gemm_type", [GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS])
@pytest.mark.parametrize(
    "shape_n, shape_k, num_experts, shape_m",
    [
        (1280, 2560, 512, 160),
        (1280, 2560, 512, 2560),
        (1280, 2560, 512, 10240),
        (2560, 640, 512, 160),
        (1536, 2048, 128, 128),
        (640, 1024, 128, 256),
    ],
)
def test_sm80_memory_bound_moe_uses_more_ctas(
    sm80_device, monkeypatch, weight_format, gemm_type, shape_n, shape_k, num_experts, shape_m
):
    layer_config = _make_moe_layer_config(shape_n, shape_k, num_experts, weight_format)

    config = Sm80Heuristics.get_config(layer_config, shape_m=shape_m, gemm_type=gemm_type)
    baseline_config = _get_config_without_rule(
        monkeypatch, Sm80Heuristics, layer_config, shape_m, gemm_type=gemm_type
    )

    block_shape, warp_shape = config["block_shape"], config["warp_shape"]
    num_k_warps = block_shape[2] // warp_shape[2]
    warps_per_cta = _get_warps_per_cta(config)
    baseline_warps_per_sm = _get_warps_per_cta(baseline_config) * baseline_config["num_ctas_per_sm"]
    smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, config["num_stages"])
    # Merged M warps dequantize each weight tile once, which pays off without more warps.
    merges_m_warps = block_shape[0] == warp_shape[0] > 16
    assert config["num_ctas_per_sm"] > 1
    assert merges_m_warps or warps_per_cta * config["num_ctas_per_sm"] > baseline_warps_per_sm
    assert config["num_stages"] >= 3
    assert not config["use_stream_k"]
    assert warps_per_cta * config["num_ctas_per_sm"] <= Sm80Heuristics.moe_occupancy_warps_per_sm
    assert warps_per_cta <= 4 or num_k_warps == 1
    assert shape_k % block_shape[2] == 0
    assert block_shape[2] % warp_shape[2] == 0
    assert smem_size * config["num_ctas_per_sm"] <= Sm80Heuristics.max_smem_size


@pytest.mark.parametrize(
    "excluded, control",
    [
        # Wide enough that a 16-row dense GEMM would have the tiles to take more CTAs.
        pytest.param(dict(num_experts=0, shape_m=16, shape_n=65536), dict(shape_n=65536), id="dense"),
        pytest.param(dict(use_batch_invariant=True), dict(), id="batch-invariant"),
        pytest.param(
            dict(a_dtype=dtypes.float16, use_f16_accum=True), dict(a_dtype=dtypes.float16), id="f16-accum"
        ),
        pytest.param(
            dict(a_dtype=dtypes.int8, weight_format="uint4-g128"), dict(weight_format="uint4-g128"), id="a8"
        ),
        # 2-bit weights put the compute-bound threshold below 28 tokens per expert, where
        # expert blocks are still 32 rows, so only the threshold keeps this config.
        pytest.param(
            dict(weight_format="uint2-g128", shape_m=512 * 28),
            dict(weight_format="uint2-g128", shape_m=512 * 4),
            id="tokens-per-expert-threshold",
        ),
        pytest.param(dict(shape_m=15360), dict(shape_m=10240), id="48-row-expert-blocks"),
        pytest.param(dict(shape_m=10), dict(shape_m=160), id="too-few-tiles"),
        pytest.param(dict(heuristics_cls=Sm87Heuristics), dict(), id="sm87"),
        # Sm100Heuristics inherits from Sm80Heuristics and falls back to its MMA configs.
        pytest.param(dict(heuristics_cls=Sm100Heuristics), dict(), id="sm100"),
        pytest.param(dict(gemm_type=GemmType.GROUPED_MASKED), dict(), id="grouped-masked"),
    ],
)
def test_sm80_moe_occupancy_exclusions(sm80_device, monkeypatch, excluded, control):
    def get_configs(values):
        defaults = dict(
            shape_m=160, shape_n=1280, num_experts=512, weight_format="nvfp4", a_dtype=dtypes.bfloat16
        )
        values = defaults | values
        heuristics_cls = values.pop("heuristics_cls", Sm80Heuristics)
        use_batch_invariant = values.pop("use_batch_invariant", False)
        use_f16_accum = values.pop("use_f16_accum", False)
        shape_m = values.pop("shape_m")
        shape_n = values.pop("shape_n")
        num_experts = values.pop("num_experts")
        gemm_type = values.pop("gemm_type", GemmType.INDEXED if num_experts else GemmType.DENSE)
        layer_config = _make_moe_layer_config(shape_n, 2560, num_experts, **values)
        kwargs = dict(
            gemm_type=gemm_type, use_batch_invariant=use_batch_invariant, use_f16_accum=use_f16_accum
        )
        config = heuristics_cls.get_config(layer_config, shape_m=shape_m, **kwargs)
        baseline_config = _get_config_without_rule(
            monkeypatch, heuristics_cls, layer_config, shape_m, **kwargs
        )
        return config, baseline_config

    excluded_config, excluded_baseline = get_configs(excluded)
    control_config, control_baseline = get_configs(control)

    assert control_config != control_baseline
    assert excluded_config == excluded_baseline


def test_sm80_moe_occupancy_expert_block_boundary(sm80_device):
    layer_config = _make_moe_layer_config(1280, 2560, 512)

    config_32_rows = Sm80Heuristics.get_config(layer_config, shape_m=10240, gemm_type=GemmType.INDEXED)
    config_48_rows = Sm80Heuristics.get_config(layer_config, shape_m=15360, gemm_type=GemmType.INDEXED)

    assert config_32_rows["block_shape"][0] == 32
    assert config_32_rows["num_ctas_per_sm"] > 1
    assert config_48_rows["block_shape"][0] == 48
    assert config_48_rows["num_ctas_per_sm"] == 1


def test_sm80_moe_occupancy_merges_m_warps_within_register_file(sm80_device, monkeypatch):
    layer_config = _make_moe_layer_config(1280, 2560, 512)

    # 32-row expert blocks: 8 warps of 16x64 become 4 warps of 32x64, whose
    # accumulators leave room for three CTAs in the register file.
    config = Sm80Heuristics.get_config(layer_config, shape_m=10240, gemm_type=GemmType.INDEXED)
    assert config["block_shape"][0] == config["warp_shape"][0] == 32
    assert _get_warps_per_cta(config) == 4
    assert config["num_ctas_per_sm"] == 3

    monkeypatch.setattr(DeviceInfo, "max_registers_per_sm", property(lambda self: 2 * 65536))
    config = Sm80Heuristics.get_config(layer_config, shape_m=10240, gemm_type=GemmType.INDEXED)
    assert config["num_ctas_per_sm"] == 4


def test_sm80_moe_occupancy_splits_n_when_tile_starved(sm80_device):
    # 32 expert blocks x 5 N tiles is fewer than three tiles per SM.
    layer_config = _make_moe_layer_config(1280, 2560, 512)
    starved_config = Sm80Heuristics.get_config(layer_config, shape_m=32, gemm_type=GemmType.INDEXED)
    config = Sm80Heuristics.get_config(layer_config, shape_m=160, gemm_type=GemmType.INDEXED)

    assert starved_config["block_shape"][1] == config["block_shape"][1] // 2
    assert starved_config["warp_shape"][1] == config["warp_shape"][1] // 2
    assert _get_warps_per_cta(starved_config) == _get_warps_per_cta(config) == 4
    assert starved_config["num_ctas_per_sm"] > 2
    assert not starved_config["use_stream_k"]


def test_sm80_moe_occupancy_halves_k_tile_for_8bit_weights(sm80_device):
    # 4-warp CTAs with a 64-wide K tile of 8-bit weights fit only twice per SM;
    # a 32-wide K tile halves the shared memory per stage and fits four.
    layer_config = _make_moe_layer_config(2560, 640, 512, "uint8-channel")
    config = Sm80Heuristics.get_config(layer_config, shape_m=40, gemm_type=GemmType.INDEXED)

    assert config["block_shape"][2] == config["warp_shape"][2] == 32
    assert _get_warps_per_cta(config) == 4
    assert config["num_ctas_per_sm"] == 4


def test_sm80_moe_occupancy_stages_follow_k_iterations(sm80_device):
    # Both GEMMs are tile-starved and split N; only the long K loop takes more stages.
    short_k_config = Sm80Heuristics.get_config(
        _make_moe_layer_config(2560, 640, 512), shape_m=16, gemm_type=GemmType.INDEXED
    )
    long_k_config = Sm80Heuristics.get_config(
        _make_moe_layer_config(1280, 2560, 512), shape_m=32, gemm_type=GemmType.INDEXED
    )

    assert short_k_config["block_shape"][1] == long_k_config["block_shape"][1] == 128
    assert short_k_config["num_stages"] == 3
    assert long_k_config["num_stages"] > 3


def test_sm80_moe_occupancy_shared_memory_fallback(sm80_device, monkeypatch):
    layer_config = _make_moe_layer_config(1280, 2560, 512)
    narrow_cta_smem_size = estimate_smem_size_layer(layer_config, (16, 256, 32), GemmType.INDEXED, 3)

    # Room for three 3-stage 4-warp CTAs of the narrowest K tile: 12 resident warps instead of 8.
    monkeypatch.setattr(Sm80Heuristics, "max_smem_size", 3 * narrow_cta_smem_size)
    three_cta_config = Sm80Heuristics.get_config(layer_config, shape_m=160, gemm_type=GemmType.INDEXED)
    assert three_cta_config["block_shape"][2] == three_cta_config["warp_shape"][2] == 32
    assert three_cta_config["num_ctas_per_sm"] == 3
    assert three_cta_config["num_stages"] == 3

    # Without room for two CTAs even of the narrowest K tile, the previous config is kept.
    monkeypatch.setattr(Sm80Heuristics, "max_smem_size", 2 * narrow_cta_smem_size - 1)
    config = Sm80Heuristics.get_config(layer_config, shape_m=160, gemm_type=GemmType.INDEXED)
    baseline_config = _get_config_without_rule(
        monkeypatch, Sm80Heuristics, layer_config, 160, gemm_type=GemmType.INDEXED
    )
    assert config == baseline_config
