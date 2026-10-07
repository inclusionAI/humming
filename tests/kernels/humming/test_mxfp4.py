"""MXFP4 W4A8 coverage with grouped FP8 inputs."""

from types import SimpleNamespace

import pytest
import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.config.mma import get_default_mma_type
from humming.schema.compressed_tensors import CompressedTensorsInputSchema
from humming.schema.humming import HummingInputSchema, HummingWeightSchema
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
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_mxfp4_input_schema_compatibility(checkpoint_format, group_size, monkeypatch):
    monkeypatch.setattr("humming.schema.humming.current_device", SimpleNamespace(sm_version=90))
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
    assert inputs.is_compatible_with(weight, torch.bfloat16) == (group_size in (32, INPUT_GROUP_SIZE))


@pytest.mark.parametrize("sm_version", [89, 90, 100, 103, 107, 110, 120, 121])
@pytest.mark.parametrize("a_dtype", ["float16", "bfloat16", "float8e4m3", "float8e5m2"])
@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
@pytest.mark.parametrize("bs_dtype", [None, "bfloat16", "float8e4m3", "float8e8m0"])
@pytest.mark.parametrize("use_group_input_scale", [False, True])
def test_fp4_weight_schema_compatibility(
    sm_version, a_dtype, group_size, bs_dtype, use_group_input_scale, monkeypatch
):
    monkeypatch.setattr("humming.schema.humming.current_device", SimpleNamespace(sm_version=sm_version))
    activation_dtype = dtypes.DataType.from_str(a_dtype)
    has_fp8_activation = activation_dtype.num_bits == 8
    input_group_size = group_size if has_fp8_activation and use_group_input_scale else 0
    inputs = HummingInputSchema(a_dtype=activation_dtype, input_scale_group_size=input_group_size)
    weight = HummingWeightSchema(
        b_dtype=dtypes.float4e2m1,
        bs_dtype=bs_dtype,
        weight_scale_group_size=group_size,
    )
    assert inputs.is_compatible_with(weight, torch.bfloat16) == (not has_fp8_activation or group_size >= 32)


@pytest.mark.parametrize("sm_version", [89, 90, 100, 103, 107, 110, 120, 121])
@pytest.mark.parametrize("bs_dtype", ["bfloat16", "float8e4m3", "float8e8m0"])
@pytest.mark.parametrize("as_dtype", ["float32", "float8e4m3", "float8e8m0"])
def test_nvfp4_input_schema_compatibility(sm_version, bs_dtype, as_dtype, monkeypatch):
    monkeypatch.setattr("humming.schema.humming.current_device", SimpleNamespace(sm_version=sm_version))
    inputs = HummingInputSchema(
        a_dtype=dtypes.float4e2m1,
        input_scale_group_size=16,
        input_scale_dtype=as_dtype,
    )
    weight = HummingWeightSchema(
        b_dtype=dtypes.float4e2m1,
        bs_dtype=bs_dtype,
        weight_scale_group_size=16,
    )
    is_supported = sm_version >= 100 and bs_dtype in ("float8e4m3", "float8e8m0") and as_dtype == bs_dtype
    assert inputs.is_compatible_with(weight, torch.bfloat16) == is_supported
