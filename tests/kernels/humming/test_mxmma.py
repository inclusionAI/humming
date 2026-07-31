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


def _case(
    name: str,
    *,
    a_dtype,
    b_dtype,
    bs_dtype,
    group_size: int,
    c_dtype=dtypes.bfloat16,
    has_zero_point: bool = False,
) -> KernelTestCase:
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=SHAPE_N,
            shape_k=SHAPE_K,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            bs_dtype=bs_dtype,
            input_scale_group_size=group_size,
            weight_scale_group_size=group_size,
            has_zero_point=has_zero_point,
            mma_type=MmaType.MXMMA,
        ),
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        seed=2026,
    )


MXMMA_FORMAT_CASES = (
    _case(
        "e3m4-fp4-e8m0-g32",
        a_dtype=dtypes.float8e3m4,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e4m3-fp4-e8m0-g32-native",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e5m2-fp4-e8m0-g32-native",
        a_dtype=dtypes.float8e5m2,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e4m3-f6e2m3-e8m0-g32-native",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float6e2m3,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e5m2-f6e3m2-e8m0-g32-native",
        a_dtype=dtypes.float8e5m2,
        b_dtype=dtypes.float6e3m2,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e4m3-e4m3-e8m0-g32",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float8e4m3,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e2m1-e2m1-e8m0-g32",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e2m1-e2m1-e4m3-g16",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e4m3,
        group_size=16,
    ),
    _case(
        "e0m3-e0m3-e8m0-g16",
        a_dtype=dtypes.float4e0m3,
        b_dtype=dtypes.float4e0m3,
        bs_dtype=dtypes.float8e8m0,
        group_size=16,
    ),
    _case(
        "e0m3-uint3-e4m3-g16",
        a_dtype=dtypes.float4e0m3,
        b_dtype=dtypes.uint3,
        bs_dtype=dtypes.float8e4m3,
        group_size=16,
    ),
)

MXMMA_ZERO_POINT_CASES = (
    _case(
        "e3m4-uint5-e8m0-g32-zp",
        a_dtype=dtypes.float8e3m4,
        b_dtype=dtypes.uint5,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        has_zero_point=True,
    ),
    _case(
        "e4m3-uint4-e8m0-g32-zp-fp16-output",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.uint4,
        c_dtype=dtypes.float16,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        has_zero_point=True,
    ),
    _case(
        "e5m2-uint3-e8m0-g32-zp",
        a_dtype=dtypes.float8e5m2,
        b_dtype=dtypes.uint3,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        has_zero_point=True,
    ),
    _case(
        "e2m1-uint2-e4m3-g16-zp",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.uint2,
        bs_dtype=dtypes.float8e4m3,
        group_size=16,
        has_zero_point=True,
    ),
    _case(
        "e2m1-uint2-e8m0-g32-zp",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.uint2,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        has_zero_point=True,
    ),
    _case(
        "e0m3-uint3-e4m3-g16-zp",
        a_dtype=dtypes.float4e0m3,
        b_dtype=dtypes.uint3,
        bs_dtype=dtypes.float8e4m3,
        group_size=16,
        has_zero_point=True,
    ),
)

MXMMA_CASES = MXMMA_FORMAT_CASES + MXMMA_ZERO_POINT_CASES


@pytest.mark.parametrize("test_case", MXMMA_CASES, ids=str)
def test_mxmma(test_case):
    config = test_case.layer_config
    assert config.mma_type == MmaType.MXMMA
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_mxmma_case_coverage():
    assert all(case.layer_config.mma_type == MmaType.MXMMA for case in MXMMA_CASES)
    assert {case.layer_config.a_dtype for case in MXMMA_CASES} == {
        dtypes.float4e0m3,
        dtypes.float4e2m1,
        dtypes.float8e3m4,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    }
    assert {case.layer_config.bs_dtype for case in MXMMA_CASES} == {
        dtypes.float8e4m3,
        dtypes.float8e8m0,
    }
    assert {case.layer_config.weight_scale_group_size for case in MXMMA_CASES} == {16, 32}
    assert any(case.layer_config.mxmma_native_mixed for case in MXMMA_CASES)

    assert len(MXMMA_ZERO_POINT_CASES) == 6
    assert all(case.layer_config.has_zero_point for case in MXMMA_ZERO_POINT_CASES)
    assert {case.layer_config.a_dtype for case in MXMMA_ZERO_POINT_CASES} == {
        dtypes.float4e0m3,
        dtypes.float4e2m1,
        dtypes.float8e3m4,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    }
