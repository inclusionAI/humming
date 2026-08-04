import pytest

from humming import dtypes
from humming.config import (
    ComputeConfig,
    GemmType,
    LayerConfig,
    WeightScale2Type,
    WeightScaleType,
)
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)


def _case(name: str, **layer_values) -> KernelTestCase:
    c_dtype = layer_values.pop("c_dtype", dtypes.bfloat16)
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=1024,
            shape_k=1024,
            c_dtype=c_dtype,
            **layer_values,
        ),
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        seed=2026,
    )


GROUP_SCALE_DTYPES = (
    dtypes.float16,
    dtypes.bfloat16,
    dtypes.float8e4m3,
    dtypes.float8e5m2,
    dtypes.float8e8m0,
)

ACTIVATION_DTYPES = (
    dtypes.float16,
    dtypes.bfloat16,
    dtypes.float8e4m3,
    dtypes.float8e5m2,
    dtypes.float8e3m4,
    dtypes.int8,
    dtypes.int4,
)

SECONDARY_SCALE_TYPES = (
    WeightScale2Type.CHANNEL,
    WeightScale2Type.TENSOR,
)


def _output_dtype(a_dtype, bs_dtype):
    if a_dtype.num_bits == 16:
        if bs_dtype.num_bits == 16 and bs_dtype != a_dtype:
            return None
        if a_dtype == dtypes.float16 and bs_dtype == dtypes.float8e8m0:
            return None
        return a_dtype
    if bs_dtype in (dtypes.float16, dtypes.bfloat16):
        if a_dtype == dtypes.float8e5m2 and bs_dtype == dtypes.float16:
            return None
        return bs_dtype
    return dtypes.bfloat16


def _bs2_case(bs_dtype, scale_2_type):
    param_dtype = bs_dtype if bs_dtype.num_bits == 16 else dtypes.bfloat16
    return _case(
        f"group64-{bs_dtype}-secondary-{scale_2_type.value}",
        a_dtype=param_dtype,
        b_dtype=dtypes.uint4,
        c_dtype=param_dtype,
        bs_dtype=bs_dtype,
        weight_scale_group_size=64,
        weight_scale_2_type=scale_2_type,
    )


BS_DTYPE_CASES = tuple(
    _case(
        f"group64-{a_dtype}-bs-{bs_dtype}",
        a_dtype=a_dtype,
        b_dtype=dtypes.uint3,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        input_scale_group_size=0 if a_dtype.num_bits == 16 else 64,
        weight_scale_group_size=64,
        use_int_weight_scale=False,
        use_fused_e8m0_scale=False,
    )
    for a_dtype in ACTIVATION_DTYPES
    for bs_dtype in GROUP_SCALE_DTYPES
    if (c_dtype := _output_dtype(a_dtype, bs_dtype)) is not None
)

BS2_CASES = tuple(
    _bs2_case(bs_dtype, scale_2_type)
    for bs_dtype in GROUP_SCALE_DTYPES
    for scale_2_type in SECONDARY_SCALE_TYPES
)

SCALE_CASES = (
    _case(
        "channel-bfloat16",
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
    ),
    *BS_DTYPE_CASES,
    _case(
        "tensor-float32",
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.float32,
        weight_scale_type=WeightScaleType.TENSOR,
    ),
    _case(
        "block64x64-float32",
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.float32,
        weight_scale_group_size=64,
        weight_scale_group_size_n=64,
        weight_scale_type=WeightScaleType.BLOCK,
    ),
    *BS2_CASES,
    _case(
        "fp8-input-group64-weight-group64",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
        input_scale_group_size=64,
        weight_scale_group_size=64,
    ),
)


@pytest.mark.parametrize("test_case", SCALE_CASES, ids=str)
def test_scale_config(test_case):
    config = test_case.layer_config
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_scale_config_case_coverage():
    configs = [case.layer_config for case in SCALE_CASES]
    assert {config.weight_scale_type for config in configs} == set(WeightScaleType)
    assert {config.bs_dtype for config in configs} == {*GROUP_SCALE_DTYPES, dtypes.float32}
    assert any(config.input_scale_group_size for config in configs)

    bs_configs = [case.layer_config for case in BS_DTYPE_CASES]
    output_dtypes = {(config.a_dtype, config.bs_dtype): config.c_dtype for config in bs_configs}
    assert len(output_dtypes) == len(BS_DTYPE_CASES)
    for a_dtype in ACTIVATION_DTYPES:
        for bs_dtype in GROUP_SCALE_DTYPES:
            expected = _output_dtype(a_dtype, bs_dtype)
            if expected is None:
                assert (a_dtype, bs_dtype) not in output_dtypes
            else:
                assert output_dtypes[a_dtype, bs_dtype] == expected

    bs2_configs = [case.layer_config for case in BS2_CASES]
    secondary_scale_pairs = {(config.bs_dtype, config.weight_scale_2_type) for config in bs2_configs}
    assert len(secondary_scale_pairs) == len(BS2_CASES)
    for bs_dtype in GROUP_SCALE_DTYPES:
        for scale_type in SECONDARY_SCALE_TYPES:
            assert (bs_dtype, scale_type) in secondary_scale_pairs
