"""MXFP4 W4A8 coverage with grouped FP8 inputs."""

import pytest
import torch

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
    num_experts: int = 0,
    use_fused_e8m0_scale: bool | None = None,
) -> LayerConfig:
    return LayerConfig(
        shape_n=SHAPE_N,
        shape_k=SHAPE_K,
        num_experts=num_experts,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=INPUT_GROUP_SIZE,
        weight_scale_group_size=WEIGHT_GROUP_SIZE,
        mma_type=MmaType.WGMMA,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
    )


def _case(
    name: str,
    *,
    gemm_type: GemmType = GemmType.DENSE,
    use_fused_e8m0_scale: bool | None = None,
) -> KernelTestCase:
    is_dense = gemm_type == GemmType.DENSE
    return KernelTestCase(
        name=name,
        layer_config=_layer_config(
            num_experts=0 if is_dense else NUM_EXPERTS,
            use_fused_e8m0_scale=use_fused_e8m0_scale,
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type),
        top_k=1 if is_dense else 2,
        seed=2026,
    )


MXFP4_CASES = (
    (True, _case("mxfp4-grouped-fp8-dense-auto")),
    (False, _case("mxfp4-grouped-fp8-dense-nonfused", use_fused_e8m0_scale=False)),
    (True, _case("mxfp4-grouped-fp8-indexed-auto", gemm_type=GemmType.INDEXED)),
    (True, _case("mxfp4-grouped-fp8-grouped-masked-auto", gemm_type=GemmType.GROUPED_MASKED)),
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
        GemmType.GROUPED_MASKED,
    }
    assert all(case.layer_config.input_scale_group_size > 0 for _, case in MXFP4_CASES)


def _shared_storage_case(*, fuse_e8m0_scale: bool) -> KernelTestCase:
    return KernelTestCase(
        name=f"mxfp4-shared-storage-{'fused' if fuse_e8m0_scale else 'explicit'}",
        layer_config=LayerConfig(
            shape_n=256,
            shape_k=256,
            num_experts=NUM_EXPERTS,
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float4e2m1,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.float8e8m0,
            input_scale_group_size=0,
            weight_scale_group_size=WEIGHT_GROUP_SIZE,
            mma_type=MmaType.WGMMA,
            use_shared_e8m0_scale_storage=True,
        ),
        compute_config=ComputeConfig(
            gemm_type=GemmType.INDEXED,
            fuse_e8m0_scale=fuse_e8m0_scale,
        ),
        top_k=2,
        seed=2027,
    )


@pytest.mark.parametrize("fuse_e8m0_scale", [False, True], ids=["explicit", "fused"])
def test_mxfp4_shared_storage_runtime_scale_mode(fuse_e8m0_scale):
    test_case = _shared_storage_case(fuse_e8m0_scale=fuse_e8m0_scale)
    config = test_case.layer_config
    assert config.use_shared_e8m0_scale_storage
    assert config.is_tensor_weight_scale_2
    assert not config.use_packed_k_layout

    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
    results = KernelTestRunner(test_case).run((17,))
    assert len(results) == 1
    torch.testing.assert_close(
        results[0].outputs,
        results[0].outputs_ref,
        rtol=test_case.rtol,
        atol=test_case.atol,
    )


def test_mxfp4_shared_storage_reuses_resident_tensors():
    explicit_case = _shared_storage_case(fuse_e8m0_scale=False)
    fused_case = _shared_storage_case(fuse_e8m0_scale=True)
    skip_if_unsupported(
        a_dtype=explicit_case.layer_config.a_dtype,
        mma_type=explicit_case.layer_config.mma_type.value,
    )

    explicit_runner = KernelTestRunner(explicit_case)
    fused_runner = KernelTestRunner(fused_case)
    fused_runner.kernel_tensors = explicit_runner.kernel_tensors
    fused_runner.weight_ref = explicit_runner.weight_ref

    for name in ("weight", "weight_scale", "weight_scale_2"):
        assert explicit_runner.kernel_tensors[name].data_ptr() == fused_runner.kernel_tensors[name].data_ptr()
    stored_scale = explicit_runner.kernel_tensors["weight_scale"].view(torch.uint8)
    assert int(stored_scale.min().item()) >= 128
    assert int(stored_scale.max().item()) <= 139

    explicit = explicit_runner.run((17,))[0]
    fused = fused_runner.run((17,))[0]
    for result, test_case in ((explicit, explicit_case), (fused, fused_case)):
        torch.testing.assert_close(
            result.outputs,
            result.outputs_ref,
            rtol=test_case.rtol,
            atol=test_case.atol,
        )
