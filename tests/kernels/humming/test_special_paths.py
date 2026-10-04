import dataclasses

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

import humming.testing.runner as runner_module
from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType, WeightScale2Type
from humming.config.mma import get_default_mma_type
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
            sm_version=90,
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
            sm_version=90,
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
            sm_version=90,
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
            sm_version=90,
        ),
    ),
    *(
        _kernel_case(
            required_features=("use_packed_k_layout", "use_fused_e8m0_scale"),
            name=f"packed-k-fused-e8m0-{a_dtype}-gs{group_size}-as{input_group_size}-{gemm_type.value}",
            layer_config=_layer_config(
                a_dtype=a_dtype,
                b_dtype=dtypes.float4e2m1,
                bs_dtype=dtypes.float8e8m0,
                input_scale_group_size=input_group_size,
                weight_scale_group_size=group_size,
                weight_scale_2_type=WeightScale2Type.CHANNEL
                if input_group_size == 0
                else WeightScale2Type.TENSOR,
                num_experts=0 if gemm_type == GemmType.DENSE else 8,
                sm_version=90,
                use_packed_k_layout=True,
            ),
            gemm_type=gemm_type,
        )
        for a_dtype in (dtypes.float8e4m3, dtypes.int8)
        for group_size in (32, 64, 128)
        for input_group_size in (0, 128)
        for gemm_type in (GemmType.DENSE, GemmType.GROUPED_CONTIGUOUS)
    ),
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
    if "use_fused_e8m0_scale" in required_features and get_default_mma_type(config) == MmaType.MXMMA:
        pytest.skip("fused E8M0 scale is not supported by MXMMA")

    for feature in required_features:
        assert getattr(config, feature) is True
    if "use_int_weight_scale" in required_features or "use_fused_e8m0_scale" in required_features:
        assert config.weight_scale_2_type != WeightScale2Type.NONE

    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=get_default_mma_type(config).value)
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


@pytest.mark.parametrize(
    "a_dtype,use_fused,has_zero_point",
    [
        (dtypes.float8e4m3, False, False),
        (dtypes.int8, False, False),
        (dtypes.float8e4m3, True, False),
        (dtypes.int8, True, False),
        (dtypes.float8e4m3, False, True),
    ],
    ids=["fp8", "int8", "fp8-fused", "int8-fused", "fp8-zp"],
)
@pytest.mark.parametrize(
    "warp_m,warp_k,warp_n,k_warps",
    [(64, 128, 16, 1), (64, 128, 16, 2), (64, 128, 32, 1), (64, 128, 64, 2), (176, 128, 16, 2)],
)
@pytest.mark.parametrize("use_warp_spec", [False, True])
def test_packed_k_geometry(
    monkeypatch, warp_m, warp_k, warp_n, k_warps, use_warp_spec, a_dtype, use_fused, has_zero_point
):
    skip_if_unsupported(a_dtype=a_dtype, mma_type="wgmma")
    case = next(
        case
        for _, case in SPECIAL_WEIGHT_CASES
        if case.layer_config.use_packed_k_layout
        and case.layer_config.use_fused_e8m0_scale == use_fused
        and case.layer_config.a_dtype == a_dtype
        and case.layer_config.has_zero_point == has_zero_point
        and (not use_fused or case.layer_config.input_scale_group_size == 128)
        and case.layer_config.weight_scale_group_size == (32 if use_fused else 128)
    )
    layer = dataclasses.replace(case.layer_config, num_experts=33)
    compute = dataclasses.replace(
        case.compute_config, gemm_type=GemmType.GROUPED_CONTIGUOUS, use_m_major_input_scale=True
    )
    case = dataclasses.replace(case, layer_config=layer, compute_config=compute)
    tuning = dict(
        block_shape=(warp_m, warp_n * 4, warp_k * k_warps),
        warp_shape=(warp_m, warp_n, warp_k),
        num_stages=3,
        use_warp_spec=use_warp_spec,
        use_stream_k=k_warps > 1,
        raster_group_m=8,
        multi_cast_size_a=1,
        multi_cast_size_b=1,
    )
    monkeypatch.setattr(
        runner_module,
        "generate_heuristics_configs",
        lambda layer, compute, shape_ms: [dict(tuning) for _ in shape_ms],
    )
    results = KernelTestRunner(case).run(shape_ms=[17, 257])
    assert_kernel_test_shape_coverage(results, [17, 257])


@pytest.mark.parametrize(
    "mma_type,block_m,chunk_rows,shape_n,gemm_type,use_tma,cta_group_size",
    (
        ("mma", 128, 96, 512, GemmType.DENSE, True, 1),
        ("mma", 128, 96, 504, GemmType.DENSE, True, 1),
        ("mma", 64, 32, 512, GemmType.INDEXED, False, 1),
        ("mma", 128, 96, 512, GemmType.GROUPED_CONTIGUOUS, True, 1),
        ("mma", 128, 96, 504, GemmType.GROUPED_MASKED, True, 1),
        ("mma", 64, 128, 512, GemmType.DENSE, True, 1),
        ("wgmma", 128, 96, 512, GemmType.DENSE, True, 1),
        ("wgmma", 128, 96, 504, GemmType.GROUPED_CONTIGUOUS, True, 1),
        ("umma", 48, 32, 512, GemmType.DENSE, True, 2),
        ("umma", 80, 64, 504, GemmType.DENSE, True, 1),
        ("umma", 80, 64, 512, GemmType.GROUPED_CONTIGUOUS, True, 1),
        ("umma", 48, 32, 504, GemmType.GROUPED_MASKED, True, 2),
        ("umma", 80, 64, 512, GemmType.INDEXED, False, 1),
        ("umma", 24, 64, 512, GemmType.DENSE, True, 1),
        ("umma", 48, 0, 512, GemmType.GROUPED_CONTIGUOUS, True, 2),
        ("umma", 128, 96, 512, GemmType.DENSE, True, 2),
    ),
)
def test_output_chunk_rows(
    mma_type, block_m, chunk_rows, shape_n, gemm_type, use_tma, cta_group_size, monkeypatch
):
    """Cover N slab stores, descriptor bounds, buffer reuse, scatter, and K reduction."""
    skip_if_unsupported(a_dtype=dtypes.bfloat16, mma_type=mma_type, use_tma=use_tma)
    layer = LayerConfig(
        shape_n=512,
        pad_shape_n=512 - shape_n,
        shape_k=1024,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
        has_bias=True,
        num_experts=0 if gemm_type == GemmType.DENSE else 4,
    )
    if mma_type == "umma" and not layer.is_umma_supported:
        pytest.skip("UMMA requires SM10x or SM11x")
    is_umma = mma_type == "umma"
    is_mma = mma_type == "mma"
    # Keep N=256 slab coverage without exhausting registers or SM120 shared memory.
    block_k = 128 if is_umma else 64
    config = dict(
        mma_type=mma_type,
        block_shape=(block_m, 256, block_k),
        warp_shape=(block_m if is_umma else 64, 32 if is_umma else 64, 128 if is_umma else 64),
        num_stages=2 if is_mma else 3,
        num_sms=6,
        num_ctas_per_sm=1,
        use_tma=use_tma,
        use_stream_k=True,
        smem_reuse_mode="last_stage" if is_mma else "none",
        umma_cta_group_size=cta_group_size,
        output_chunk_rows=chunk_rows,
    )
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "heuristic")
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: config)
    case = KernelTestCase(
        name="output-chunk-rows",
        layer_config=layer,
        compute_config=ComputeConfig(gemm_type=gemm_type),
        seed=2026,
    )
    shape_ms = (17, 13 * block_m + 1)
    results = KernelTestRunner(case).run(shape_ms)
    assert_kernel_test_shape_coverage(results, shape_ms)
