"""Numerical correctness for the production SM90 W4A8 indexed-MoE path."""

import pytest
import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType
from humming.testing import KernelTestCase, KernelTestRunner, skip_if_unsupported


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 CUDA device required",
)


def test_sm90_w4a8_indexed_group32_shapes():
    """Cover three BM/BK regimes without retaining a multi-GB reference."""
    test_case = KernelTestCase(
        name="sm90-w4a8-indexed-group32",
        layer_config=LayerConfig(
            shape_n=1024,
            shape_k=1024,
            num_experts=8,
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float4e2m1,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=32,
            mma_type=MmaType.WGMMA,
        ),
        compute_config=ComputeConfig(gemm_type=GemmType.INDEXED),
        top_k=2,
        seed=2026,
        rtol=0.02,
        atol=2.0,
    )
    config = test_case.layer_config
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)

    results = KernelTestRunner(test_case).run((64, 128, 256))
    assert [result.tuning_config.block_shape for result in results] == [
        (16, 128, 256),
        (32, 128, 128),
        (64, 128, 128),
    ]
