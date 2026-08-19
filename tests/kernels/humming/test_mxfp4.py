"""MXFP4 W4A8 coverage with grouped FP8 inputs."""

import pytest

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)

SHAPE_N = 1024
SHAPE_K = 1024
INPUT_GROUP_SIZE = 128
WEIGHT_GROUP_SIZE = 32
NUM_EXPERTS = 8


def _layer_config(
    *,
    shape_n: int = SHAPE_N,
    shape_k: int = SHAPE_K,
    num_experts: int = 0,
    use_fused_e8m0_scale: bool | None = None,
    a_dtype=dtypes.float8e4m3,
    as_dtype=None,
    input_scale_group_size: int = INPUT_GROUP_SIZE,
) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=a_dtype,
        as_dtype=as_dtype,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=WEIGHT_GROUP_SIZE,
        mma_type=MmaType.WGMMA,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
    )


def _case(
    name: str,
    *,
    shape_n: int = SHAPE_N,
    shape_k: int = SHAPE_K,
    gemm_type: GemmType = GemmType.DENSE,
    use_fused_e8m0_scale: bool | None = None,
    a_dtype=dtypes.float8e4m3,
    as_dtype=None,
    input_scale_group_size: int = INPUT_GROUP_SIZE,
) -> KernelTestCase:
    is_dense = gemm_type == GemmType.DENSE
    return KernelTestCase(
        name=name,
        layer_config=_layer_config(
            shape_n=shape_n,
            shape_k=shape_k,
            num_experts=0 if is_dense else NUM_EXPERTS,
            use_fused_e8m0_scale=use_fused_e8m0_scale,
            a_dtype=a_dtype,
            as_dtype=as_dtype,
            input_scale_group_size=input_scale_group_size,
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type),
        top_k=1 if is_dense else 2,
        seed=2026,
    )


MXFP4_CASES = (
    (
        False,
        _case(
            "mxfp4-a16-dense",
            a_dtype=dtypes.bfloat16,
            input_scale_group_size=0,
        ),
    ),
    (
        False,
        _case(
            "mxfp4-a16-indexed",
            gemm_type=GemmType.INDEXED,
            a_dtype=dtypes.bfloat16,
            input_scale_group_size=0,
        ),
    ),
    (True, _case("mxfp4-grouped-fp8-dense-auto")),
    (
        True,
        _case(
            "mxfp4-grouped-fp8-g32-dense-n64-k64",
            shape_n=2880,
            shape_k=2880,
            as_dtype=dtypes.float32,
            input_scale_group_size=32,
        ),
    ),
    (False, _case("mxfp4-grouped-fp8-dense-nonfused", use_fused_e8m0_scale=False)),
    (True, _case("mxfp4-grouped-fp8-indexed-auto", gemm_type=GemmType.INDEXED)),
    (
        True,
        _case(
            "mxfp4-grouped-fp8-grouped-contiguous-auto",
            gemm_type=GemmType.GROUPED_CONTIGUOUS,
        ),
    ),
    (
        True,
        _case(
            "mxfp4-grouped-fp8-grouped-masked-auto",
            gemm_type=GemmType.GROUPED_MASKED,
        ),
    ),
)


@pytest.mark.parametrize(
    "expected_fused,test_case",
    MXFP4_CASES,
    ids=[case.name for _, case in MXFP4_CASES],
)
def test_mxfp4(expected_fused, test_case):
    config = test_case.layer_config
    assert config.use_fused_e8m0_scale is expected_fused
    assert config.is_group_weight_scale
    assert config.is_tensor_weight_scale_2 is expected_fused

    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_mxfp4_case_coverage():
    assert {expected_fused for expected_fused, _ in MXFP4_CASES} == {False, True}
    assert {case.compute_config.gemm_type for _, case in MXFP4_CASES} == {
        GemmType.DENSE,
        GemmType.INDEXED,
        GemmType.GROUPED_CONTIGUOUS,
        GemmType.GROUPED_MASKED,
    }
    assert {case.layer_config.a_dtype for _, case in MXFP4_CASES} == {
        dtypes.bfloat16,
        dtypes.float8e4m3,
    }
