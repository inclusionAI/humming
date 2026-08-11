import pytest

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)

SHAPE_N = 2048
SHAPE_K = 2048
A_DTYPES = (
    "float16",
    "bfloat16",
    "float8e4m3",
    "float8e5m2",
    "float8e3m4",
    "int8",
    "int4",
)

B_DTYPES = (
    "uint1",
    "uint2",
    "uint3",
    "uint4",
    "uint5",
    "uint6",
    "uint7",
    "uint8",
    "int4",
    "int8",
    "float2e0m1",
    "float3e0m2",
    "float3e1m1",
    "float3e2m0",
    "float4e0m3",
    "float4e2m1",
    "float4e3m0",
    "float5e2m2",
    "float5e4m0",
    "float6e2m3",
    "float6e4m1",
    "float7e0m6",
    "float7e2m4",
    "float7e4m2",
    "float7e6m0",
    "float8e1m6",
    "float8e4m3",
    "float8e5m2",
)

C_DTYPES = ("float16", "bfloat16")


def _is_compatible(a_dtype, b_dtype) -> bool:
    if b_dtype.num_bits > a_dtype.num_bits:
        return False

    if b_dtype.is_integer_type and a_dtype.is_integer_type:
        if a_dtype.num_bits == b_dtype.num_bits:
            return a_dtype == b_dtype
        return not b_dtype.is_signed

    if b_dtype.is_integer_type and a_dtype.is_floating_point_type:
        return not b_dtype.is_signed and b_dtype.num_bits <= a_dtype.mantissa_bits + 2

    if b_dtype.is_floating_point_type and a_dtype.is_floating_point_type:
        return (
            b_dtype.is_signed
            and b_dtype.exponent_bits <= a_dtype.exponent_bits
            and b_dtype.mantissa_bits <= a_dtype.mantissa_bits
            and (a_dtype.exponent_bits == 0 or b_dtype.exponent_bits >= 1)
        )

    return False


def _make_cases() -> list[KernelTestCase]:
    cases = []
    signatures = set()
    compute_config = ComputeConfig(gemm_type=GemmType.DENSE)

    for a_dtype_str in A_DTYPES:
        a_dtype = dtypes.DataType.from_str(a_dtype_str)
        for b_dtype_str in B_DTYPES:
            for c_dtype_str in C_DTYPES:
                c_dtype = dtypes.DataType.from_str(c_dtype_str)
                if a_dtype.num_bits == 16 and a_dtype != c_dtype:
                    continue
                if a_dtype == dtypes.float8e5m2 and c_dtype == dtypes.float16:
                    continue

                layer_config = LayerConfig(
                    shape_n=SHAPE_N,
                    shape_k=SHAPE_K,
                    a_dtype=a_dtype,
                    b_dtype=dtypes.DataType.from_str(b_dtype_str),
                    c_dtype=c_dtype,
                    bs_dtype=c_dtype,
                    input_scale_group_size=0,
                    weight_scale_group_size=0,
                )
                if not _is_compatible(layer_config.a_dtype, layer_config.b_dtype):
                    continue

                signature = (
                    str(layer_config.a_dtype),
                    str(layer_config.b_dtype),
                    str(layer_config.c_dtype),
                )
                if signature in signatures:
                    continue
                signatures.add(signature)

                name = "-".join(signature)
                case = KernelTestCase(
                    name=name,
                    layer_config=layer_config,
                    compute_config=compute_config,
                    seed=2026,
                    input_std_scale=0.05 if layer_config.b_dtype == dtypes.float8e5m2 else 1.0,
                )
                cases.append(case)

    return cases


DATATYPE_CASES = _make_cases()


@pytest.mark.parametrize("test_case", DATATYPE_CASES, ids=str)
def test_datatype(test_case):
    skip_if_unsupported(
        a_dtype=test_case.layer_config.a_dtype,
        mma_type=test_case.layer_config.mma_type.value,
    )
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_datatype_case_coverage():
    signatures = {
        (
            str(case.layer_config.a_dtype),
            str(case.layer_config.b_dtype),
            str(case.layer_config.c_dtype),
        )
        for case in DATATYPE_CASES
    }

    for a_dtype in A_DTYPES:
        assert any(signature[0] == a_dtype for signature in signatures)
    for bit_width in range(1, 9):
        assert any(case.layer_config.b_dtype.num_bits == bit_width for case in DATATYPE_CASES)
    assert {signature[2] for signature in signatures} == set(C_DTYPES)
