import pytest

from humming import dtypes
from humming.config import (
    ComputeConfig,
    GemmType,
    LayerConfig,
    WeightScale2Type,
    WeightScaleType,
)
from humming.config.mma import get_default_mma_type
from humming.device import current_device
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
    _case(
        "channel-e8m0-bias",
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.float8e8m0,
        has_bias=True,
    ),
    _case(
        "channel-e4m3-bias",
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.float8e4m3,
        has_bias=True,
    ),
    _case(
        "secondary-channel-bias",
        a_dtype=dtypes.float16,
        c_dtype=dtypes.float16,
        b_dtype=dtypes.uint4,
        weight_scale_group_size=128,
        weight_scale_2_type=WeightScale2Type.CHANNEL,
        has_bias=True,
    ),
    _case(
        "secondary-tensor-bias",
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        weight_scale_group_size=128,
        weight_scale_2_type=WeightScale2Type.TENSOR,
        has_bias=True,
    ),
    *BS_DTYPE_CASES,
    *(
        _case(
            f"raw-fp8-group64-bs-{bs_dtype}",
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float8e4m3,
            bs_dtype=bs_dtype,
            input_scale_group_size=64,
            weight_scale_group_size=64,
            use_int_weight_scale=False,
            use_fused_e8m0_scale=False,
        )
        for bs_dtype in (dtypes.bfloat16, dtypes.float8e4m3)
    ),
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
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=get_default_mma_type(config).value)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


@pytest.mark.parametrize("sm_version", [80, 90, 100])
@pytest.mark.parametrize("a_dtype", [dtypes.bfloat16, dtypes.float8e4m3, dtypes.int8])
def test_group_scale_layout_is_independent_of_device(sm_version, a_dtype):
    config = LayerConfig(
        shape_n=1024,
        shape_k=1024,
        a_dtype=a_dtype,
        b_dtype=dtypes.uint3,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=64,
        use_int_weight_scale=False,
        use_fused_e8m0_scale=False,
        sm_version=sm_version,
    )
    assert config.should_apply_bs_on_c == (current_device.is_ppu and a_dtype.num_bits != 16)


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


@pytest.mark.parametrize("warp_n", (16, 32, 64))
@pytest.mark.parametrize("use_tma", (False, True))
@pytest.mark.parametrize("bs_dtype", (dtypes.bfloat16, dtypes.float8e4m3))
def test_raw_wgmma_ss_group_scale_layout(warp_n, use_tma, bs_dtype, monkeypatch):
    """Group scales follow SS's N16 fragments, including wide per-warp tiles."""
    skip_if_unsupported(a_dtype=dtypes.float8e4m3, mma_type="wgmma")
    case = _case(
        "raw-wgmma-ss-group64",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float8e4m3,
        bs_dtype=bs_dtype,
        input_scale_group_size=64,
        weight_scale_group_size=64,
        use_int_weight_scale=False,
        use_fused_e8m0_scale=False,
    )
    tuning = dict(
        mma_type="wgmma",
        block_shape=(16, warp_n * 4, 256),
        warp_shape=(16, warp_n, 128),
        num_stages=3,
        use_tma=use_tma,
        use_warp_spec=use_tma,
        use_stream_k=False,
    )
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: dict(tuning))
    results = KernelTestRunner(case).run((1, 17, 129))
    assert_kernel_test_shape_coverage(results, (1, 17, 129))


@pytest.mark.parametrize("warp_n", (16, 32, 64))
@pytest.mark.parametrize("input_group", (0, 64))
def test_raw_wgmma_ss_block_scale_layout(warp_n, input_group, monkeypatch):
    skip_if_unsupported(a_dtype=dtypes.float8e4m3, mma_type="wgmma")
    case = _case(
        "raw-wgmma-ss-block64",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float8e4m3,
        bs_dtype=dtypes.float32,
        weight_scale_type=WeightScaleType.BLOCK,
        weight_scale_group_size=64,
        weight_scale_group_size_n=64,
        input_scale_group_size=input_group,
    )
    tuning = dict(
        mma_type="wgmma",
        block_shape=(16, warp_n * 4, 128),
        warp_shape=(16, warp_n, 128),
        num_stages=3,
        use_tma=True,
        use_warp_spec=True,
        use_stream_k=False,
    )
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: dict(tuning))
    results = KernelTestRunner(case).run((1, 17, 129))
    assert_kernel_test_shape_coverage(results, (1, 17, 129))


@pytest.mark.parametrize(
    "a_dtype,b_dtype,input_group,weight_group,quant_mode",
    (
        ("float8e4m3", "float8e4m3", 0, 32, "dynamic_token"),
        ("float8e4m3", "float4e2m1", 32, 0, "dynamic_group"),
        ("float4e2m1", "float4e2m1", 0, 0, "dynamic_token"),
        ("float8e4m3", "float8e4m3", 0, 0, "static_tensor"),
        ("float8e4m3", "float4e2m1", 0, 32, "dynamic_token"),
        ("float4e2m1", "float4e2m1", 32, 0, "dynamic_group"),
    ),
)
def test_optional_native_group_scales(a_dtype, b_dtype, input_group, weight_group, quant_mode):
    if current_device.sm_version < 100:
        pytest.skip("native block-scaled MMA requires SM100 or newer")
    layer = LayerConfig(
        shape_n=256,
        shape_k=512,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=dtypes.bfloat16,
        as_dtype=dtypes.float8e8m0 if input_group else None,
        bs_dtype=dtypes.float8e8m0 if weight_group else dtypes.bfloat16,
        input_quant_mode=quant_mode,
        input_scale_group_size=input_group,
        weight_scale_group_size=weight_group,
        weight_scale_type="group" if weight_group else "channel",
    )
    skip_if_unsupported(a_dtype=layer.a_dtype, mma_type=get_default_mma_type(layer).value)
    case = KernelTestCase(
        name="optional-native-group-scales",
        layer_config=layer,
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        seed=2026,
    )
    results = KernelTestRunner(case).run((1, 17, 129))
    assert_kernel_test_shape_coverage(results, (1, 17, 129))
