import pytest

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)

MIN_SHAPE_N = 64
MIN_SHAPE_K = 32


def _case(
    name: str,
    *,
    shape_n: int,
    shape_k: int,
    pad_shape_n: int = 0,
    pad_shape_k: int = 0,
    a_dtype=dtypes.bfloat16,
    b_dtype=dtypes.uint4,
) -> KernelTestCase:
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=shape_n,
            shape_k=shape_k,
            pad_shape_n=pad_shape_n,
            pad_shape_k=pad_shape_k,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.bfloat16,
        ),
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        seed=2026,
    )


PROBLEM_SHAPE_CASES = (
    _case(
        "minimum",
        shape_n=MIN_SHAPE_N,
        shape_k=MIN_SHAPE_K,
    ),
    _case(
        "small-tile-boundary",
        shape_n=128,
        shape_k=64,
        a_dtype=dtypes.int8,
        b_dtype=dtypes.uint7,
    ),
    _case(
        "large-n-small-k",
        shape_n=8192,
        shape_k=MIN_SHAPE_K,
    ),
    _case(
        "small-n-large-k",
        shape_n=MIN_SHAPE_N,
        shape_k=8192,
        a_dtype=dtypes.int4,
        b_dtype=dtypes.uint3,
    ),
    _case(
        "large-rectangular",
        shape_n=4096,
        shape_k=8192,
    ),
)


PAD_SHAPE_CASES = (
    _case(
        "minimum-pad-n",
        shape_n=128,
        shape_k=128,
        pad_shape_n=8,
    ),
    _case(
        "minimum-packable-pad-k",
        shape_n=128,
        shape_k=128,
        pad_shape_k=32,
    ),
    _case(
        "one-third-pad-n",
        shape_n=1024,
        shape_k=1024,
        pad_shape_n=344,
    ),
    _case(
        "one-third-pad-k",
        shape_n=1024,
        shape_k=1024,
        pad_shape_k=352,
    ),
    _case(
        "one-third-pad-nk",
        shape_n=1024,
        shape_k=1024,
        pad_shape_n=344,
        pad_shape_k=352,
        a_dtype=dtypes.int4,
        b_dtype=dtypes.uint3,
    ),
    _case(
        "maximum-pad-n",
        shape_n=1024,
        shape_k=1024,
        pad_shape_n=1024 - MIN_SHAPE_N,
    ),
    _case(
        "maximum-pad-k",
        shape_n=1024,
        shape_k=1024,
        pad_shape_k=1024 - MIN_SHAPE_K,
    ),
    _case(
        "maximum-pad-nk",
        shape_n=1024,
        shape_k=1024,
        pad_shape_n=1024 - MIN_SHAPE_N,
        pad_shape_k=1024 - MIN_SHAPE_K,
        a_dtype=dtypes.int8,
        b_dtype=dtypes.uint7,
    ),
)


SHAPE_CASES = PROBLEM_SHAPE_CASES + PAD_SHAPE_CASES


@pytest.mark.parametrize("test_case", SHAPE_CASES, ids=str)
def test_shape(test_case):
    config = test_case.layer_config
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_shape_case_coverage():
    problem_configs = {case.name: case.layer_config for case in PROBLEM_SHAPE_CASES}
    minimum = problem_configs["minimum"]
    tile_boundary = problem_configs["small-tile-boundary"]
    large_n = problem_configs["large-n-small-k"]
    large_k = problem_configs["small-n-large-k"]
    large_rectangular = problem_configs["large-rectangular"]

    assert (minimum.shape_n, minimum.shape_k) == (MIN_SHAPE_N, MIN_SHAPE_K)
    assert large_n.shape_n >= 8192 and large_n.shape_k == MIN_SHAPE_K
    assert large_k.shape_n == MIN_SHAPE_N and large_k.shape_k >= 8192
    assert large_rectangular.shape_n >= 4096 and large_rectangular.shape_k >= 4096

    assert (minimum.a_dtype.num_bits, minimum.b_dtype.num_bits) == (16, 4)
    assert (tile_boundary.a_dtype.num_bits, tile_boundary.b_dtype.num_bits) == (8, 7)
    assert (large_k.a_dtype.num_bits, large_k.b_dtype.num_bits) == (4, 3)

    pad_configs = {case.name: case.layer_config for case in PAD_SHAPE_CASES}
    pad_modes = {(bool(config.pad_shape_n), bool(config.pad_shape_k)) for config in pad_configs.values()}
    assert pad_modes == {(True, False), (False, True), (True, True)}

    one_third = pad_configs["one-third-pad-nk"]
    assert one_third.pad_shape_n / one_third.shape_n == pytest.approx(1 / 3, abs=0.01)
    assert one_third.pad_shape_k / one_third.shape_k == pytest.approx(1 / 3, abs=0.02)
    assert (one_third.a_dtype.num_bits, one_third.b_dtype.num_bits) == (4, 3)

    maximum = pad_configs["maximum-pad-nk"]
    assert maximum.shape_n - maximum.pad_shape_n == MIN_SHAPE_N
    assert maximum.shape_k - maximum.pad_shape_k == MIN_SHAPE_K
    assert (maximum.a_dtype.num_bits, maximum.b_dtype.num_bits) == (8, 7)
