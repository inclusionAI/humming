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
WEIGHT_SCALE_GROUP_SIZE = 64

A_DTYPES = (
    "float16",
    "bfloat16",
    "float8e4m3",
    "float8e5m2",
    "float8e3m4",
    "int8",
    "int4",
)
B_DTYPES = tuple(f"uint{num_bits}" for num_bits in range(1, 9))
C_DTYPES = ("float16", "bfloat16")


def _output_dtypes(a_dtype):
    if a_dtype == dtypes.float16:
        return (dtypes.float16,)
    if a_dtype == dtypes.bfloat16 or a_dtype == dtypes.float8e5m2:
        return (dtypes.bfloat16,)
    return tuple(dtypes.DataType.from_str(dtype) for dtype in C_DTYPES)


def _supports_integer_zero_point(a_dtype, b_dtype) -> bool:
    if b_dtype.num_bits > a_dtype.num_bits:
        return False
    if a_dtype.is_integer_type:
        return b_dtype.num_bits < a_dtype.num_bits
    return b_dtype.num_bits <= a_dtype.mantissa_bits + 1


def _make_cases() -> list[KernelTestCase]:
    cases = []
    compute_config = ComputeConfig(gemm_type=GemmType.DENSE)

    for is_fp_zero_point in (False, True):
        for a_dtype_str in A_DTYPES:
            a_dtype = dtypes.DataType.from_str(a_dtype_str)
            if is_fp_zero_point and a_dtype.num_bits != 16:
                continue

            for b_dtype_str in B_DTYPES:
                b_dtype = dtypes.DataType.from_str(b_dtype_str)
                if not _supports_integer_zero_point(a_dtype, b_dtype):
                    continue

                for c_dtype in _output_dtypes(a_dtype):
                    zero_point_kind = "fp-zp" if is_fp_zero_point else "int-zp"
                    layer_config = LayerConfig(
                        shape_n=SHAPE_N,
                        shape_k=SHAPE_K,
                        a_dtype=a_dtype,
                        b_dtype=b_dtype,
                        c_dtype=c_dtype,
                        bs_dtype=c_dtype,
                        input_scale_group_size=0,
                        weight_scale_group_size=64,
                        use_int_weight_scale=False,
                        has_zero_point=True,
                        is_fp_zero_point=is_fp_zero_point,
                    )
                    case = KernelTestCase(
                        name=(f"{zero_point_kind}-{a_dtype}-{layer_config.b_dtype}-{c_dtype}"),
                        layer_config=layer_config,
                        compute_config=compute_config,
                        seed=2026,
                    )
                    cases.append(case)

    return cases


ZERO_POINT_CASES = _make_cases()


@pytest.mark.parametrize("test_case", ZERO_POINT_CASES, ids=str)
def test_zero_point(test_case):
    skip_if_unsupported(
        a_dtype=test_case.layer_config.a_dtype,
        mma_type=test_case.layer_config.mma_type.value,
    )
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_zero_point_case_coverage():
    integer_cases = [case for case in ZERO_POINT_CASES if not case.layer_config.is_fp_zero_point]
    floating_cases = [case for case in ZERO_POINT_CASES if case.layer_config.is_fp_zero_point]

    assert {str(case.layer_config.a_dtype) for case in integer_cases} == set(A_DTYPES)
    assert {case.layer_config.b_dtype.num_bits for case in integer_cases} == set(range(1, 9))
    assert {str(case.layer_config.c_dtype) for case in ZERO_POINT_CASES} == set(C_DTYPES)
    assert {str(case.layer_config.a_dtype) for case in floating_cases} == {"float16", "bfloat16"}
    assert {case.layer_config.b_dtype.num_bits for case in floating_cases} == set(range(1, 9))
    assert all(case.layer_config.has_zero_point for case in ZERO_POINT_CASES)
    assert all(case.layer_config.weight_scale_group_size == 64 for case in ZERO_POINT_CASES)
