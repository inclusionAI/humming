import pytest
import torch

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.tune import explain_tuning_config
from humming.tune import cache as tune_cache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "H20" not in torch.cuda.get_device_name()
    or "H200" in torch.cuda.get_device_name(),
    reason="explain_tuning_config audit trail is implemented for H20 only",
)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMMING_CACHE_DIR", str(tmp_path))
    tune_cache._get_default_tune_cache_dir.cache_clear()
    yield str(tmp_path)
    tune_cache._get_default_tune_cache_dir.cache_clear()


def _fused_moe_layer(input_scale_group_size=128):
    return LayerConfig(
        shape_n=6144,
        shape_k=7168,
        num_experts=48,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=32,
        weight_scale_type="group",
        mma_type=MmaType.WGMMA,
    )


_FLAGS = {
    "use_f16_accum": False,
    "use_batch_invariant": False,
    "use_m_major_input_scale": False,
}


def test_measured_hit_returns_audited_decision(cache_dir):
    layer = _fused_moe_layer()
    config = {
        "block_shape": (16, 128, 128),
        "warp_shape": (16, 32, 128),
        "use_stream_k": True,
        "use_f16_accum": False,
        "num_sms": 78,
        "num_stages": 5,
        "num_ctas_per_sm": 2,
    }
    tune_cache.save_table(
        layer,
        GemmType.INDEXED,
        _FLAGS,
        tune_cache.current_fingerprint(),
        [[0, 1 << 30, config]],
        cache_dir=cache_dir,
    )

    decision = explain_tuning_config(layer, 4096, gemm_type="indexed")
    assert decision is not None
    assert decision.family == "measured"
    assert decision.to_config() == config
    assert "measured winner from tune cache" in decision.reason
    assert len(decision.considered) == 1


def test_miss_in_scope_returns_family_decision(cache_dir):
    decision = explain_tuning_config(_fused_moe_layer(), 12288, gemm_type="indexed")
    assert decision is not None
    assert decision.family == "fused_e8m0_moe"
    assert decision.selected.num_ctas_per_sm == 2  # register-budget demotion
    assert "register-budget" in decision.reason


def test_legacy_path_has_no_decision(cache_dir):
    layer = LayerConfig(
        shape_n=4096,
        shape_k=4096,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.int4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
        weight_scale_type="group",
    )
    assert explain_tuning_config(layer, 4096, gemm_type="dense") is None


def test_measured_reason_carries_audit_metadata(cache_dir):
    layer = _fused_moe_layer()
    config = {
        "block_shape": (16, 128, 128),
        "warp_shape": (16, 32, 128),
        "use_stream_k": True,
        "use_f16_accum": False,
        "num_sms": 78,
        "num_stages": 5,
        "num_ctas_per_sm": 2,
    }
    tune_cache.save_table(
        layer,
        GemmType.INDEXED,
        _FLAGS,
        tune_cache.current_fingerprint(),
        [[64, 4096, config]],
        cache_dir=cache_dir,
    )

    decision = explain_tuning_config(layer, 2048, gemm_type="indexed")
    key = tune_cache.make_cache_key(layer, GemmType.INDEXED, _FLAGS)
    assert key in decision.reason
    assert "(64, 4096]" in decision.reason
    assert "created 20" in decision.reason  # ISO timestamp persisted and shown


def test_fingerprint_mismatch_falls_back_to_family_decision(cache_dir):
    layer = _fused_moe_layer()
    stale = dict(tune_cache.current_fingerprint(), humming_version="0.0.stale")
    tune_cache.save_table(
        layer,
        GemmType.INDEXED,
        _FLAGS,
        stale,
        [[0, 1 << 30, {"num_stages": 3}]],
        cache_dir=cache_dir,
    )

    decision = explain_tuning_config(layer, 12288, gemm_type="indexed")
    assert decision is not None
    assert decision.family == "fused_e8m0_moe"
