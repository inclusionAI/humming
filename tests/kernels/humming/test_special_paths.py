import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType, WeightScale2Type
from humming.forward import humming_forward
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)
from humming.testing.data import generate_random_tensor

SHAPE_N = 1024
SHAPE_K = 1024
GROUPED_INPUT_SIZE = 128

SPECIAL_FEATURES = {
    "use_int_weight_scale",
    "use_fused_e8m0_scale",
    "use_packed_k_layout",
}


def _layer_config(**kwargs) -> LayerConfig:
    return LayerConfig(
        shape_n=SHAPE_N,
        shape_k=SHAPE_K,
        c_dtype=dtypes.bfloat16,
        **kwargs,
    )


def _kernel_case(
    required_features: tuple[str, ...],
    name: str,
    layer_config: LayerConfig,
    gemm_type: GemmType = GemmType.DENSE,
) -> tuple[tuple[str, ...], KernelTestCase]:
    is_dense = gemm_type == GemmType.DENSE
    test_case = KernelTestCase(
        name=name,
        layer_config=layer_config,
        compute_config=ComputeConfig(gemm_type=gemm_type),
        top_k=1 if is_dense else 2,
        seed=2026,
    )
    return required_features, test_case


def _ldmatrix_s4_case(
    name: str, shape_n: int, shape_k: int, force_legacy: bool = False
) -> tuple[tuple[str, ...], KernelTestCase]:
    # force_legacy runs the same shape/seed on packed_k_legacy; both arms compare to the
    # runner's reference, so matching it proves automatic-S4 == forced-legacy (A/B parity).
    features = ("use_packed_k_layout",) if force_legacy else ("use_packed_k_layout", "use_ldmatrix_s4")
    return _kernel_case(
        required_features=features,
        name=name,
        layer_config=LayerConfig(
            shape_n=shape_n,
            shape_k=shape_k,
            a_dtype=dtypes.int8,
            b_dtype=dtypes.uint4,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=128,
            has_zero_point=False,
            mma_type=MmaType.WGMMA,
            test_force_packed_k_legacy=force_legacy,
        ),
    )


LDMATRIX_S4_SHAPE_MS = {
    "packed-k-ldmatrix-s4-minimum": (1, 127, 128),
    "packed-k-ldmatrix-s4-forced-legacy": (1, 127, 128),
    "packed-k-ldmatrix-s4-model-shape": (256,),
}


SPECIAL_WEIGHT_CASES = (
    _kernel_case(
        required_features=(),
        name="nvfp4-a16-dense",
        layer_config=_layer_config(
            a_dtype=dtypes.bfloat16,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e4m3,
            weight_scale_group_size=16,
            weight_scale_2_type=WeightScale2Type.TENSOR,
        ),
    ),
    _kernel_case(
        required_features=(),
        name="nvfp4-a16-indexed",
        layer_config=_layer_config(
            a_dtype=dtypes.bfloat16,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e4m3,
            weight_scale_group_size=16,
            weight_scale_2_type=WeightScale2Type.TENSOR,
            num_experts=8,
        ),
        gemm_type=GemmType.INDEXED,
    ),
    _kernel_case(
        required_features=("use_int_weight_scale",),
        name="int-weight-scale-int8",
        layer_config=_layer_config(
            a_dtype=dtypes.int8,
            b_dtype=dtypes.uint4,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=64,
            weight_scale_group_size_n=1,
        ),
    ),
    _kernel_case(
        required_features=("use_int_weight_scale",),
        name="int-weight-scale-int4",
        layer_config=_layer_config(
            a_dtype=dtypes.int4,
            b_dtype=dtypes.uint3,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=64,
            weight_scale_group_size_n=1,
        ),
    ),
    _kernel_case(
        required_features=("use_int_weight_scale",),
        name="odd-bit-packed-k-fallback",
        layer_config=_layer_config(
            a_dtype=dtypes.int8,
            b_dtype=dtypes.uint5,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=128,
            weight_scale_group_size_n=1,
            mma_type=MmaType.WGMMA,
        ),
    ),
    _kernel_case(
        required_features=("use_fused_e8m0_scale",),
        name="fused-e8m0-tensor-secondary-grouped-input",
        layer_config=_layer_config(
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e8m0,
            input_scale_group_size=GROUPED_INPUT_SIZE,
            weight_scale_group_size=64,
        ),
    ),
    _kernel_case(
        required_features=("use_fused_e8m0_scale",),
        name="fused-e8m0-channel-secondary",
        layer_config=_layer_config(
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e8m0,
            weight_scale_group_size=64,
            weight_scale_2_type=WeightScale2Type.CHANNEL,
        ),
    ),
    _kernel_case(
        required_features=("use_packed_k_layout",),
        name="packed-k-fp8-grouped-input",
        layer_config=_layer_config(
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.uint4,
            bs_dtype=dtypes.bfloat16,
            input_scale_group_size=GROUPED_INPUT_SIZE,
            weight_scale_group_size=128,
            mma_type=MmaType.WGMMA,
        ),
    ),
    _kernel_case(
        required_features=("use_int_weight_scale", "use_packed_k_layout"),
        name="packed-k-int8-with-int-scale",
        layer_config=_layer_config(
            a_dtype=dtypes.int8,
            b_dtype=dtypes.uint4,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=128,
            weight_scale_group_size_n=1,
            mma_type=MmaType.WGMMA,
        ),
    ),
    _kernel_case(
        required_features=("use_packed_k_layout",),
        name="packed-k-zero-point",
        layer_config=_layer_config(
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.uint4,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=128,
            has_zero_point=True,
            mma_type=MmaType.WGMMA,
        ),
    ),
<<<<<<< ours
<<<<<<< ours
=======
    _kernel_case(
        required_features=("use_packed_k_layout", "use_ldmatrix_s4"),
        name="packed-k-ldmatrix-s4",
        layer_config=_layer_config(
            a_dtype=dtypes.int8,
            b_dtype=dtypes.uint4,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=128,
            has_zero_point=False,
            mma_type=MmaType.WGMMA,
            # The test re-resolves selection after its environment checks.
        ),
    ),
>>>>>>> theirs
=======
    _ldmatrix_s4_case("packed-k-ldmatrix-s4-minimum", 128, 128),
    _ldmatrix_s4_case("packed-k-ldmatrix-s4-forced-legacy", 128, 128, force_legacy=True),
    _ldmatrix_s4_case("packed-k-ldmatrix-s4", SHAPE_N, SHAPE_K),
    _ldmatrix_s4_case("packed-k-ldmatrix-s4-model-shape", 4096, 4096),
>>>>>>> theirs
)


def test_forward_fullgraph():
    """Catch graph breaks in the forward path and reuse the graph across token counts."""
    torch._dynamo.reset()
    skip_if_unsupported(a_dtype=dtypes.bfloat16)
    config = _layer_config(
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
    )
    runner = KernelTestRunner(
        KernelTestCase(
            name="fullgraph",
            layer_config=config,
            compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        )
    )
    compute_config = runner.compute_config.to_str()
    locks = torch.zeros(1024, device="cuda", dtype=torch.int32)

    def forward(inputs):
        return humming_forward(
            config,
            inputs,
            **runner.kernel_tensors,
            locks=locks,
            compute_config=compute_config,
            tuning_config={},
        )

    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(forward, backend=counter, fullgraph=True, dynamic=True)
    for shape_m in (17, 257):
        torch.manual_seed(runner.test_case.seed + shape_m)
        # Match the numerical range used by KernelTestRunner: Stream-K stores
        # partial sums in the output dtype, so absolute error scales with inputs.
        inputs = generate_random_tensor((shape_m, SHAPE_K), dtype=torch.bfloat16, device="cuda")
        expected = (inputs.float() @ runner.weight_ref.T).to(torch.bfloat16)
        torch.testing.assert_close(forward(inputs), expected, rtol=0.01, atol=0.05)
        torch.testing.assert_close(compiled(inputs), expected, rtol=0.01, atol=0.05)
    assert counter.frame_count == 1


@pytest.mark.parametrize(
    "required_features,test_case",
    SPECIAL_WEIGHT_CASES,
    ids=[case.name for _, case in SPECIAL_WEIGHT_CASES],
)
def test_special_weight_path(required_features, test_case):
    config = test_case.layer_config
    if "use_fused_e8m0_scale" in required_features and config.mma_type == MmaType.MXMMA:
        pytest.skip("fused E8M0 scale is not supported by MXMMA")

<<<<<<< ours
<<<<<<< ours
=======
    min_cuda_version = (13, 4) if "use_ldmatrix_s4" in required_features else None
=======
>>>>>>> theirs
    skip_if_unsupported(
        a_dtype=config.a_dtype,
        mma_type=config.mma_type.value,
    )

    if "use_ldmatrix_s4" in required_features:
        config = dataclasses.replace(config, use_ldmatrix_s4=None)
        if not config.can_use_ldmatrix_s4:
            pytest.skip(str(config.ldmatrix_s4_rejection_reasons))
        test_case = dataclasses.replace(test_case, layer_config=config)

>>>>>>> theirs
    for feature in required_features:
        assert getattr(config, feature) is True
    if "use_int_weight_scale" in required_features or "use_fused_e8m0_scale" in required_features:
        assert config.weight_scale_2_type != WeightScale2Type.NONE

<<<<<<< ours
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)
=======
    shape_ms = LDMATRIX_S4_SHAPE_MS.get(test_case.name)
    results = KernelTestRunner(test_case).run(shape_ms)
    assert_kernel_test_shape_coverage(results, shape_ms)
    if "use_ldmatrix_s4" in required_features:
        assert all(result.tuning_values["block_shape"][2] == 64 for result in results)
        assert all(result.tuning_values["warp_shape"][2] == 64 for result in results)
>>>>>>> theirs


def test_special_weight_path_coverage():
    assert {feature for features, _ in SPECIAL_WEIGHT_CASES for feature in features} == SPECIAL_FEATURES

    configs_by_feature = {feature: [] for feature in SPECIAL_FEATURES}
    for features, case in SPECIAL_WEIGHT_CASES:
        for feature in features:
            configs_by_feature[feature].append(case.layer_config)

    int_scale_configs = configs_by_feature["use_int_weight_scale"]
    fused_configs = configs_by_feature["use_fused_e8m0_scale"]
    packed_k_configs = configs_by_feature["use_packed_k_layout"]
    assert {config.a_dtype for config in int_scale_configs} == {dtypes.int4, dtypes.int8}
    assert {config.weight_scale_2_type.name for config in fused_configs} == {"CHANNEL", "TENSOR"}
    assert {config.a_dtype for config in packed_k_configs} == {dtypes.float8e4m3, dtypes.int8}
    assert any(config.use_int_weight_scale for config in packed_k_configs)

    odd_bit_fallback = next(case.layer_config for _, case in SPECIAL_WEIGHT_CASES if "odd-bit" in case.name)
    assert odd_bit_fallback.b_dtype.num_bits % 2 == 1
    assert odd_bit_fallback.use_packed_k_layout is False
<<<<<<< ours
=======


def test_use_ldmatrix_s4_default_matches_eligibility():
    skip_if_unsupported(mma_type="wgmma")
    config = _layer_config(
        a_dtype=dtypes.int8,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
        has_zero_point=False,
        mma_type=MmaType.WGMMA,
    )
    assert config.use_ldmatrix_s4 == config.can_use_ldmatrix_s4


def test_use_ldmatrix_s4_incompatible_geometry_rejected():
    skip_if_unsupported(mma_type="wgmma")
    from humming.testing.tuning import _is_legal_geometry

    config = _layer_config(
        a_dtype=dtypes.int8,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
        has_zero_point=False,
        mma_type=MmaType.WGMMA,
        use_ldmatrix_s4=None,
    )
    if not config.can_use_ldmatrix_s4:
        pytest.skip(str(config.ldmatrix_s4_rejection_reasons))
    assert not _is_legal_geometry(config, block_shape=(64, 128, 128), warp_shape=(64, 32, 128))
    assert _is_legal_geometry(config, block_shape=(64, 128, 64), warp_shape=(64, 32, 64))
>>>>>>> theirs
