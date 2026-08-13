import pytest

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType, WeightScale2Type
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)

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


SPECIAL_WEIGHT_CASES = (
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
)


@pytest.mark.parametrize(
    "required_features,test_case",
    SPECIAL_WEIGHT_CASES,
    ids=[case.name for _, case in SPECIAL_WEIGHT_CASES],
)
def test_special_weight_path(required_features, test_case):
    config = test_case.layer_config
    if "use_fused_e8m0_scale" in required_features and config.mma_type == MmaType.MXMMA:
        pytest.skip("fused E8M0 scale is not supported by MXMMA")

    for feature in required_features:
        assert getattr(config, feature) is True
    if "use_int_weight_scale" in required_features or "use_fused_e8m0_scale" in required_features:
        assert config.weight_scale_2_type != WeightScale2Type.NONE

    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


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
