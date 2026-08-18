import pytest

from humming import dtypes
from humming.config import GemmType, MmaType
from humming.config import LayerConfig
from humming.tune.space import Sm90H20SearchSpace, get_search_space
from humming.utils.smem import estimate_smem_size_layer

H20_NUM_SMS = 78


def _w2_meta(num_experts: int = 0, shape_k: int = 2048, a_dtype: str = "float8e4m3"):
    return LayerConfig(
        shape_n=7168,
        shape_k=shape_k,
        a_dtype=dtypes.DataType.from_str(a_dtype),
        b_dtype=dtypes.DataType.from_str("int4"),
        c_dtype=dtypes.DataType.from_str("bfloat16"),
        bs_dtype=dtypes.DataType.from_str("bfloat16"),
        weight_scale_group_size=128,
        num_experts=num_experts,
        mma_type=MmaType.WGMMA,
    )


def test_enumerate_deterministic():
    meta = _w2_meta()

    candidates = Sm90H20SearchSpace.enumerate(meta, GemmType.DENSE, H20_NUM_SMS)
    candidates_again = Sm90H20SearchSpace.enumerate(meta, GemmType.DENSE, H20_NUM_SMS)

    assert candidates == candidates_again


@pytest.mark.parametrize(
    "gemm_type,num_experts",
    [
        ("dense", 0),
        ("grouped_masked", 32),
    ],
)
def test_enumerated_all_valid(gemm_type, num_experts):
    gemm_type = GemmType(gemm_type)
    meta = _w2_meta(num_experts=num_experts)

    candidates = Sm90H20SearchSpace.enumerate(meta, gemm_type, H20_NUM_SMS)

    assert candidates
    for config in candidates:
        assert Sm90H20SearchSpace.is_valid(meta, gemm_type, config), f"{config=}"
        assert Sm90H20SearchSpace.broken_reason(meta, gemm_type, config) is None, f"{config=}"


def _base_config(block_shape, warp_shape, use_warp_spec=False):
    config = {
        "block_shape": block_shape,
        "warp_shape": warp_shape,
        "use_stream_k": False,
        "use_f16_accum": False,
        "num_sms": H20_NUM_SMS,
        "num_stages": 3,
        "num_ctas_per_sm": 1,
    }
    if use_warp_spec:
        config["use_warp_spec"] = True
    return config


def test_broken_families():
    meta = _w2_meta(num_experts=32)

    hit_cases = [
        (
            "moe-warp-spec",
            GemmType.GROUPED_MASKED,
            _base_config((64, 128, 128), (64, 32, 64), use_warp_spec=True),
        ),
        (
            "bn64-bk64-small-bm",
            GemmType.DENSE,
            _base_config((48, 64, 64), (48, 16, 64)),
        ),
        (
            "bn64-small-bm",
            GemmType.DENSE,
            _base_config((32, 64, 128), (32, 16, 64)),
        ),
        (
            "wn64-8bit",
            GemmType.DENSE,
            _base_config((64, 128, 128), (64, 64, 64)),
        ),
    ]
    miss_cases = [
        (GemmType.DENSE, _base_config((64, 128, 128), (64, 32, 64), use_warp_spec=True)),
        (GemmType.DENSE, _base_config((64, 64, 64), (64, 16, 32))),
        (GemmType.DENSE, _base_config((48, 64, 128), (48, 16, 64))),
        (GemmType.DENSE, _base_config((64, 128, 128), (64, 32, 64))),
    ]

    for name, gemm_type, config in hit_cases:
        reason = Sm90H20SearchSpace.broken_reason(meta, gemm_type, config)
        assert reason is not None and name in reason, f"{name=} {reason=}"
    for gemm_type, config in miss_cases:
        assert Sm90H20SearchSpace.broken_reason(meta, gemm_type, config) is None, f"{config=}"



def test_pipeline_gating():
    dense_meta = _w2_meta()
    masked_meta = _w2_meta(num_experts=32)

    dense_candidates = Sm90H20SearchSpace.enumerate(dense_meta, GemmType.DENSE, H20_NUM_SMS)
    masked_candidates = Sm90H20SearchSpace.enumerate(
        masked_meta,
        GemmType.GROUPED_MASKED,
        H20_NUM_SMS,
    )

    assert not any(config.get("use_tma") for config in masked_candidates)
    assert not any(config.get("use_warp_spec") for config in masked_candidates)
    assert any(config.get("use_tma") for config in dense_candidates)
    assert all(
        config["block_shape"][0] >= 48
        for config in dense_candidates
        if config.get("use_tma")
    )


def test_stream_k_gating():
    small_k_meta = _w2_meta(shape_k=1024)
    masked_meta = _w2_meta(num_experts=32)

    dense_candidates = Sm90H20SearchSpace.enumerate(small_k_meta, GemmType.DENSE, H20_NUM_SMS)
    masked_candidates = Sm90H20SearchSpace.enumerate(
        masked_meta,
        GemmType.GROUPED_MASKED,
        H20_NUM_SMS,
    )

    assert not any(config["use_stream_k"] for config in dense_candidates)
    assert not any(config["use_stream_k"] for config in masked_candidates)


@pytest.mark.parametrize(
    "gemm_type,num_experts",
    [
        ("dense", 0),
        ("grouped_masked", 32),
    ],
)
def test_smem_respected(gemm_type, num_experts):
    gemm_type = GemmType(gemm_type)
    meta = _w2_meta(num_experts=num_experts)

    candidates = Sm90H20SearchSpace.enumerate(meta, gemm_type, H20_NUM_SMS)

    assert candidates
    for config in candidates:
        smem_size = estimate_smem_size_layer(
            meta,
            config["block_shape"],
            gemm_type,
            config["num_stages"],
        )
        total_smem = smem_size * config["num_ctas_per_sm"]
        assert total_smem <= Sm90H20SearchSpace.max_smem_size, f"{total_smem=}"


def test_golden_counts():
    dense_meta = _w2_meta()
    masked_meta = _w2_meta(num_experts=32)

    dense_candidates = Sm90H20SearchSpace.enumerate(dense_meta, GemmType.DENSE, H20_NUM_SMS)
    masked_candidates = Sm90H20SearchSpace.enumerate(
        masked_meta,
        GemmType.GROUPED_MASKED,
        H20_NUM_SMS,
    )

    # Update these golden counts when the search space intentionally changes.
    assert len(dense_candidates) == 3548
    assert len(masked_candidates) == 808


def test_get_search_space():
    assert get_search_space(90, "NVIDIA H20") is Sm90H20SearchSpace

    with pytest.raises(NotImplementedError, match="register a new DeviceSearchSpace"):
        get_search_space(80, "NVIDIA A100")
