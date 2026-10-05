import pytest

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType
from humming.config.mma import get_default_mma_type
from humming.device import current_device
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)

SHAPE_N = 1024
SHAPE_K = 1024
NUM_EXPERTS = 8
TOP_K = 2


def _case(
    name: str,
    *,
    a_dtype,
    b_dtype,
    bs_dtype,
    group_size: int,
    c_dtype=dtypes.bfloat16,
    has_zero_point: bool = False,
    input_group_size: int | None = None,
    weight_group_size: int | None = None,
    gemm_type: GemmType = GemmType.DENSE,
    use_m_major_input_scale: bool = False,
    **layer_values,
) -> KernelTestCase:
    input_group_size = group_size if input_group_size is None else input_group_size
    weight_group_size = group_size if weight_group_size is None else weight_group_size
    if input_group_size and not weight_group_size:
        scale_dtype = dtypes.float8e4m3 if input_group_size == 16 else dtypes.float8e8m0
        layer_values.setdefault("as_dtype", scale_dtype)
    is_dense = gemm_type == GemmType.DENSE
    sm_version = current_device.sm_version
    if sm_version // 10 not in (10, 11, 12):
        sm_version = 120
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=SHAPE_N,
            shape_k=SHAPE_K,
            num_experts=0 if is_dense else NUM_EXPERTS,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            bs_dtype=bs_dtype,
            input_scale_group_size=input_group_size,
            weight_scale_group_size=weight_group_size,
            has_zero_point=has_zero_point,
            sm_version=sm_version,
            **layer_values,
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type, use_m_major_input_scale=use_m_major_input_scale),
        top_k=1 if is_dense else TOP_K,
        seed=2026,
    )


FORMAT_CASES = (
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
        "e4m3-e2m1-channel-input",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        input_group_size=0,
    ),
    _case(
        "e2m1-e2m1-channel-weight",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.bfloat16,
        group_size=32,
        weight_group_size=0,
    ),
    _case(
        "e2m1-e2m1-channel-input-channel-weight",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.bfloat16,
        group_size=32,
        input_group_size=0,
        weight_group_size=0,
    ),
    _case(
        "e4m3-e2m1-channel-input-indexed",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        input_group_size=0,
        gemm_type=GemmType.INDEXED,
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

ZERO_POINT_CASES = (
    _case(
        "e4m3-uint4-channel-zp",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
        group_size=0,
        has_zero_point=True,
        input_quant_mode="static_tensor",
    ),
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
    _case(
        "e2m1-uint2-e8m0-g16-zp-indexed",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.uint2,
        bs_dtype=dtypes.float8e8m0,
        group_size=16,
        has_zero_point=True,
        gemm_type=GemmType.INDEXED,
    ),
)

NATIVE_QUANTIZATION_CASES = tuple(
    _case(
        f"{a_dtype}-{b_dtype}-{scale_dtype}-g{group_size}-{quant_mode}-{gemm_type.value}",
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        bs_dtype=scale_dtype,
        as_dtype=scale_dtype,
        group_size=group_size,
        input_quant_mode=quant_mode,
        weight_scale_2_type="tensor",
        has_bias=True,
        gemm_type=gemm_type,
        use_m_major_input_scale=gemm_type != GemmType.INDEXED,
    )
    for a_dtype, b_dtype, group_size, scale_dtype, quant_mode, gemm_type in (
        ("float4e2m1", "float4e2m1", 32, "float8e8m0", "dynamic_group", GemmType.DENSE),
        ("float4e2m1", "float4e2m1", 16, "float8e4m3", "dynamic_group_token", GemmType.INDEXED),
        ("float4e2m1", "float4e2m1", 16, "float8e8m0", "static_tensor_dynamic_group", GemmType.DENSE),
        ("float4e0m3", "float4e0m3", 16, "float8e4m3", "dynamic_group_token", GemmType.DENSE),
        ("float4e0m3", "float4e0m3", 16, "float8e4m3", "dynamic_group_token", GemmType.GROUPED_MASKED),
        ("float4e0m3", "float4e0m3", 16, "float8e8m0", "static_tensor_dynamic_group", GemmType.DENSE),
        ("float4e0m3", "float4e2m1", 16, "float8e4m3", "dynamic_group_token", GemmType.INDEXED),
        ("float4e0m3", "float4e2m1", 16, "float8e8m0", "dynamic_group", GemmType.DENSE),
        ("float4e2m1", "float4e0m3", 16, "float8e8m0", "dynamic_group", GemmType.GROUPED_CONTIGUOUS),
        ("float4e2m1", "float4e0m3", 16, "float8e4m3", "dynamic_group_token", GemmType.INDEXED),
        ("float8e4m3", "float4e2m1", 32, "float8e8m0", "dynamic_group", GemmType.DENSE),
        ("float8e5m2", "float6e3m2", 32, "float8e8m0", "dynamic_group", GemmType.INDEXED),
        ("float8e3m4", "float8e3m4", 32, "float8e8m0", "dynamic_group", GemmType.DENSE),
        ("float8e4m3", "float6e2m3", 32, "float8e8m0", "dynamic_group", GemmType.GROUPED_MASKED),
    )
)

OPTIONAL_SCALE_CASES = tuple(
    _case(
        f"{a_dtype}-{b_dtype}-as{input_group}-bs{weight_group}-{quant_mode}",
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        bs_dtype=dtypes.float8e8m0 if weight_group else dtypes.bfloat16,
        as_dtype=dtypes.float8e8m0 if input_group else None,
        group_size=32,
        input_group_size=input_group,
        weight_group_size=weight_group,
        input_quant_mode=quant_mode,
    )
    for a_dtype, b_dtype, input_group, weight_group, quant_mode in (
        ("float8e4m3", "float8e4m3", 0, 32, "dynamic_token"),
        ("float8e4m3", "float4e2m1", 32, 0, "dynamic_group"),
        ("float4e2m1", "float4e2m1", 0, 0, "dynamic_token"),
        ("float8e4m3", "float8e4m3", 0, 0, "static_tensor"),
        ("float8e4m3", "float4e2m1", 0, 32, "dynamic_token"),
        ("float4e2m1", "float4e2m1", 32, 0, "dynamic_group"),
    )
)

BLOCK_SCALED_CASES = FORMAT_CASES + ZERO_POINT_CASES + NATIVE_QUANTIZATION_CASES + OPTIONAL_SCALE_CASES


@pytest.mark.parametrize("test_case", BLOCK_SCALED_CASES, ids=str)
def test_block_scaled(test_case):
    config = test_case.layer_config
    assert get_default_mma_type(config) in (MmaType.UMMA, MmaType.MXMMA)
    mma_type = get_default_mma_type(config).value
    skip_if_unsupported(a_dtype=config.a_dtype, b_dtype=config.b_dtype, mma_type=mma_type)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_block_scaled_case_coverage():
    assert all(
        get_default_mma_type(case.layer_config) in (MmaType.UMMA, MmaType.MXMMA)
        for case in BLOCK_SCALED_CASES
    )
    assert {case.layer_config.a_dtype for case in BLOCK_SCALED_CASES} == {
        dtypes.float4e0m3,
        dtypes.float4e2m1,
        dtypes.float8e3m4,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    }
    assert {case.layer_config.bs_dtype for case in BLOCK_SCALED_CASES} == {
        dtypes.bfloat16,
        dtypes.float8e4m3,
        dtypes.float8e8m0,
    }
    assert {case.layer_config.weight_scale_group_size for case in BLOCK_SCALED_CASES} == {
        0,
        16,
        32,
    }
    assert {case.compute_config.gemm_type for case in BLOCK_SCALED_CASES} == {
        GemmType.DENSE,
        GemmType.INDEXED,
        GemmType.GROUPED_CONTIGUOUS,
        GemmType.GROUPED_MASKED,
    }
    assert any(case.layer_config.has_bias for case in BLOCK_SCALED_CASES)
    assert any(case.compute_config.use_m_major_input_scale for case in BLOCK_SCALED_CASES)
    assert any(
        case.layer_config.a_dtype in (dtypes.float8e4m3, dtypes.float8e5m2)
        and case.layer_config.b_dtype in (dtypes.float4e2m1, dtypes.float6e3m2, dtypes.float6e2m3)
        for case in BLOCK_SCALED_CASES
    )

    assert len(ZERO_POINT_CASES) == 8
    assert all(case.layer_config.has_zero_point for case in ZERO_POINT_CASES)
    assert {case.layer_config.a_dtype for case in ZERO_POINT_CASES} == {
        dtypes.float4e0m3,
        dtypes.float4e2m1,
        dtypes.float8e3m4,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    }
