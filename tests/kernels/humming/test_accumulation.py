import pytest

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)


def _case(
    name: str,
    *,
    a_dtype,
    b_dtype,
    shape_k: int = 1024,
    input_scale_group_size: int = 0,
    weight_scale_group_size: int = 0,
    bs_dtype=dtypes.float16,
    has_bias: bool = False,
    pad_shape_n: int = 0,
    pad_shape_k: int = 0,
    gemm_type: GemmType = GemmType.DENSE,
    num_experts: int = 0,
) -> KernelTestCase:
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=1024,
            shape_k=shape_k,
            pad_shape_n=pad_shape_n,
            pad_shape_k=pad_shape_k,
            num_experts=num_experts,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=dtypes.float16,
            bs_dtype=bs_dtype,
            input_scale_group_size=input_scale_group_size,
            weight_scale_group_size=weight_scale_group_size,
            has_bias=has_bias,
        ),
        compute_config=ComputeConfig(
            gemm_type=gemm_type,
            use_f16_accum=True,
        ),
        top_k=2 if gemm_type != GemmType.DENSE else 1,
        seed=2026,
    )


F16_ACCUM_CASES = (
    _case(
        "fp16-channel-scale",
        a_dtype=dtypes.float16,
        b_dtype=dtypes.uint4,
    ),
    _case(
        "fp16-group-scale",
        a_dtype=dtypes.float16,
        b_dtype=dtypes.uint4,
        weight_scale_group_size=64,
    ),
    _case(
        "fp16-bias-pad-nk",
        a_dtype=dtypes.float16,
        b_dtype=dtypes.uint4,
        has_bias=True,
        pad_shape_n=24,
        pad_shape_k=32,
    ),
    _case(
        "fp8-integer-weight",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.uint4,
    ),
    _case(
        "fp8-fp4-weight",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
    ),
    _case(
        "fp8-grouped-input-and-weight",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.uint4,
        input_scale_group_size=64,
        weight_scale_group_size=64,
    ),
    _case(
        "fp8-f4-grouped-input-and-weight-32",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        input_scale_group_size=32,
        weight_scale_group_size=32,
    ),
    _case(
        "fp16-indexed-bias",
        a_dtype=dtypes.float16,
        b_dtype=dtypes.uint4,
        has_bias=True,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
        num_experts=8,
    ),
)


@pytest.mark.parametrize("test_case", F16_ACCUM_CASES, ids=str)
def test_f16_accumulation(test_case):
    config = test_case.layer_config
    assert test_case.compute_config.use_f16_accum
    assert config.c_dtype == dtypes.float16
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)

    runner = KernelTestRunner(test_case)
    results = runner.run()
    assert_kernel_test_shape_coverage(results)
    assert all(result.tuning_values["use_f16_accum"] for result in results)


def test_f16_accumulation_case_coverage():
    assert {case.layer_config.a_dtype for case in F16_ACCUM_CASES} == {dtypes.float16, dtypes.float8e4m3}
    assert any(case.layer_config.has_bias for case in F16_ACCUM_CASES)
    assert any(case.layer_config.pad_shape_k for case in F16_ACCUM_CASES)
    assert any(case.layer_config.weight_scale_group_size for case in F16_ACCUM_CASES)
    assert any(case.layer_config.input_scale_group_size for case in F16_ACCUM_CASES)
    assert not any(case.layer_config.use_fused_e8m0_scale for case in F16_ACCUM_CASES)
    assert any(case.compute_config.gemm_type == GemmType.GROUPED_CONTIGUOUS for case in F16_ACCUM_CASES)
