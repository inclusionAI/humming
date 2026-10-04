"""MXFP4 W4A8 coverage with grouped FP8 inputs."""

import pytest
import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.config.mma import get_default_mma_type
from humming.device import current_device
from humming.schema.compressed_tensors import CompressedTensorsInputSchema
from humming.schema.humming import HummingWeightSchema
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
        sm_version=90,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
    )


def _case(
    name: str,
    *,
    shape_n: int = SHAPE_N,
    shape_k: int = SHAPE_K,
    gemm_type: GemmType = GemmType.DENSE,
    num_experts: int = NUM_EXPERTS,
    top_k: int = 2,
    use_m_major_input_scale: bool = False,
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
            num_experts=0 if is_dense else num_experts,
            use_fused_e8m0_scale=use_fused_e8m0_scale,
            a_dtype=a_dtype,
            as_dtype=as_dtype,
            input_scale_group_size=input_scale_group_size,
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type, use_m_major_input_scale=use_m_major_input_scale),
        top_k=1 if is_dense else top_k,
        seed=2026,
        atol=0.5 if use_m_major_input_scale else 0.05,
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
    *(
        (
            True,
            _case(
                f"mxfp4-m-major-e{experts}-n{shape_n}-k{shape_k}",
                shape_n=shape_n,
                shape_k=shape_k,
                gemm_type=GemmType.GROUPED_CONTIGUOUS,
                num_experts=experts,
                top_k=8,
                use_m_major_input_scale=True,
            ),
        )
        for experts, shape_n, shape_k in (
            (8, 1024, 1024),
            (8, 1088, 1024),
            (33, 1024, 1024),
            (128, 1024, 1024),
            (32, 4096, 6144),
            (32, 6144, 2048),
        )
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
    if test_case.uses_m_major_input_scale:
        assert config.use_packed_k_layout

    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=get_default_mma_type(config).value)
    results = KernelTestRunner(test_case).run()
    if test_case.uses_m_major_input_scale:
        assert all(
            result.tuning_values.get("use_packed_k_layout", config.use_packed_k_layout) for result in results
        )
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


@pytest.mark.parametrize("checkpoint_format", ["mxfp4-pack-quantized", "float-quantized"])
@pytest.mark.parametrize("group_size", [64, 128])
def test_mxfp4_input_schema_compatibility(checkpoint_format, group_size):
    skip_if_unsupported(a_dtype=dtypes.float8e4m3, mma_type="wgmma")
    weight = HummingWeightSchema(
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        weight_scale_group_size=WEIGHT_GROUP_SIZE,
    )
    inputs = CompressedTensorsInputSchema(
        format=checkpoint_format,
        type="float",
        num_bits=8,
        dynamic=True,
        group_size=group_size,
    ).to_humming_schema(torch.bfloat16)
    assert inputs.input_scale_dtype is None
    assert inputs.is_compatible_with(weight, torch.bfloat16) == (group_size == INPUT_GROUP_SIZE)


@pytest.mark.parametrize(
    "a_dtype,b_dtype,group_size,scale_dtype,quant_mode,gemm_type",
    (
        ("float4e2m1", "float4e2m1", 32, "float8e8m0", "dynamic_group", GemmType.DENSE),
        ("float4e2m1", "float4e2m1", 16, "float8e4m3", "dynamic_group_token", GemmType.INDEXED),
        ("float4e2m1", "float4e2m1", 16, "float8e8m0", "static_tensor_dynamic_group", GemmType.DENSE),
        ("float4e0m3", "float4e0m3", 16, "float8e4m3", "dynamic_group_token", GemmType.DENSE),
        ("float4e0m3", "float4e2m1", 16, "float8e4m3", "dynamic_group_token", GemmType.INDEXED),
        ("float4e2m1", "float4e0m3", 16, "float8e8m0", "dynamic_group", GemmType.GROUPED_CONTIGUOUS),
        ("float8e4m3", "float4e2m1", 32, "float8e8m0", "dynamic_group", GemmType.DENSE),
        ("float8e5m2", "float6e3m2", 32, "float8e8m0", "dynamic_group", GemmType.INDEXED),
        ("float8e3m4", "float8e3m4", 32, "float8e8m0", "dynamic_group", GemmType.DENSE),
        ("float8e4m3", "float6e2m3", 32, "float8e8m0", "dynamic_group", GemmType.GROUPED_MASKED),
    ),
)
def test_native_block_scaled(a_dtype, b_dtype, group_size, scale_dtype, quant_mode, gemm_type):
    """Numerical contracts shared by native block-scaled backends and sampled tuning."""
    if current_device.sm_version < 100:
        pytest.skip("native block-scaled MMA requires SM100 or newer")
    layer = LayerConfig(
        shape_n=512,
        shape_k=1024,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=dtypes.bfloat16,
        as_dtype=scale_dtype,
        bs_dtype=scale_dtype,
        input_scale_group_size=group_size,
        weight_scale_group_size=group_size,
        input_quant_mode=quant_mode,
        weight_scale_2_type="tensor",
        has_bias=True,
        num_experts=0 if gemm_type == GemmType.DENSE else 4,
    )
    backend = get_default_mma_type(layer)
    skip_if_unsupported(a_dtype=layer.a_dtype, mma_type=backend.value)
    if not layer.use_block_scaled_mma:
        pytest.skip("the device does not support this native block-scaled format")
    case = KernelTestCase(
        name="native-block-scaled",
        layer_config=layer,
        compute_config=ComputeConfig(
            gemm_type=gemm_type,
            use_m_major_input_scale=gemm_type != GemmType.INDEXED,
        ),
        seed=2026,
    )
    results = KernelTestRunner(case).run((1, 17, 129))
    assert_kernel_test_shape_coverage(results, (1, 17, 129))
