"""SM90 W4A8 indexed-MoE heuristic coverage.

These tests intentionally validate the production selector directly.  They do
not benchmark kernels and do not depend on the removed experimental env var.
"""

import pytest
import torch

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.tune.candidate import DeviceProfile, TuningProblem
from humming.tune.sm90 import Sm90Heuristics
from humming.tune.sm90_policies import (
    _get_w4a8_moe_bm_candidates,
    _use_w4a8_moe_bm_heuristic_v1,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 CUDA device required",
)


def _layer(shape_n: int = 4096, shape_k: int = 6144) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=256,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=32,
        mma_type=MmaType.WGMMA,
    )


def _problem(layer, gemm_type, sm_version=90):
    return TuningProblem(
        layer_config=layer,
        shape_m=4096,
        gemm_type=gemm_type,
        device=DeviceProfile(
            name=f"sm{sm_version}",
            sm_version=sm_version,
            num_sms=132,
            max_smem_size=227 * 1024,
        ),
    )


def test_w4a8_indexed_selector_scope():
    layer = _layer()
    assert _use_w4a8_moe_bm_heuristic_v1(_problem(layer, GemmType.INDEXED))
    assert not _use_w4a8_moe_bm_heuristic_v1(_problem(layer, GemmType.DENSE))
    assert not _use_w4a8_moe_bm_heuristic_v1(
        _problem(layer, GemmType.GROUPED_CONTIGUOUS)
    )
    assert not _use_w4a8_moe_bm_heuristic_v1(
        _problem(layer, GemmType.GROUPED_MASKED)
    )
    assert not _use_w4a8_moe_bm_heuristic_v1(
        _problem(layer, GemmType.INDEXED, sm_version=80)
    )


def test_w4a8_candidates_are_dynamic_and_legal():
    candidates = _get_w4a8_moe_bm_candidates(_layer(), max_block_m=192)
    assert candidates == list(range(8, 193, 8))
    assert candidates
    assert all(8 <= bm <= 192 and bm % 8 == 0 for bm in candidates)


@pytest.mark.parametrize("shape_n,shape_k", [(4096, 6144), (6144, 2048)])
@pytest.mark.parametrize(
    ("routed_m", "expected_bm", "expected_bk"),
    [
        (4096, 16, 256),
        (8192, 32, 128),
        (16384, 64, 128),
    ],
)
def test_w4a8_indexed_config_changes_bm_only(
    monkeypatch,
    shape_n,
    shape_k,
    routed_m,
    expected_bm,
    expected_bk,
):
    monkeypatch.setattr(
        Sm90Heuristics,
        "get_num_sms",
        classmethod(lambda cls: 132),
    )
    config = Sm90Heuristics.get_config(
        _layer(shape_n, shape_k),
        routed_m,
        gemm_type=GemmType.INDEXED,
    )

    assert config["block_shape"] == (expected_bm, 128, expected_bk)
    assert config["warp_shape"] == (expected_bm, 32, 128)
    assert config["num_stages"] == 4
    assert config["use_stream_k"] is True
