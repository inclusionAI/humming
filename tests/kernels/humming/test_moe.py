import pytest
import torch

from humming import dtypes, ops
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.config.mma import get_default_mma_type
from humming.device import current_device
from humming.kernel.humming import HummingKernel
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)
from humming.tune import get_heuristics_config

SHAPE_N = 1024
SHAPE_K = 1024
NUM_EXPERTS = 8
TOP_K = 2


def _case(
    name: str,
    gemm_type: GemmType,
    *,
    use_m_major_input_scale: bool = False,
    expert_max_tokens: int | None = None,
    shape_k: int = SHAPE_K,
    **layer_values,
) -> KernelTestCase:
    defaults = {
        "a_dtype": dtypes.bfloat16,
        "b_dtype": dtypes.uint4,
        "c_dtype": dtypes.bfloat16,
        "bs_dtype": dtypes.bfloat16,
    }
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=SHAPE_N,
            shape_k=shape_k,
            num_experts=NUM_EXPERTS,
            **(defaults | layer_values),
        ),
        compute_config=ComputeConfig(
            gemm_type=gemm_type,
            use_m_major_input_scale=use_m_major_input_scale,
        ),
        top_k=TOP_K,
        expert_max_tokens=expert_max_tokens,
        seed=2026,
        input_std_scale=0.5,
        weight_std_scale=0.5,
        bias_std_scale=0.5,
        atol=0.5 if use_m_major_input_scale else 0.2,
    )


MOE_CASES = (
    _case(
        "indexed-uint8-zp",
        GemmType.INDEXED,
        b_dtype=dtypes.uint8,
        weight_scale_group_size=64,
        has_zero_point=True,
    ),
    _case("grouped-masked-fp6", GemmType.GROUPED_MASKED, b_dtype=dtypes.float6e3m2),
    _case("grouped-contiguous-fp8", GemmType.GROUPED_CONTIGUOUS, b_dtype=dtypes.float8e3m4),
    _case(
        "grouped-contiguous-nvfp4",
        GemmType.GROUPED_CONTIGUOUS,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e4m3,
        weight_scale_group_size=16,
        weight_scale_2_type="tensor",
    ),
    _case(
        "indexed-mxfp4",
        GemmType.INDEXED,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        weight_scale_group_size=32,
    ),
    _case(
        "grouped-masked-block-scale",
        GemmType.GROUPED_MASKED,
        bs_dtype=dtypes.float32,
        weight_scale_type="block",
        weight_scale_group_size=64,
        weight_scale_group_size_n=64,
    ),
    _case(
        "indexed-secondary-channel",
        GemmType.INDEXED,
        b_dtype=dtypes.uint3,
        weight_scale_group_size=64,
        weight_scale_2_type="channel",
        has_bias=True,
    ),
    _case(
        "grouped-contiguous-fp-zp",
        GemmType.GROUPED_CONTIGUOUS,
        weight_scale_group_size=64,
        has_zero_point=True,
        is_fp_zero_point=True,
    ),
    _case("indexed", GemmType.INDEXED),
    _case(
        "indexed-static-input",
        GemmType.INDEXED,
        a_dtype=dtypes.float8e4m3,
        input_quant_mode="static_tensor",
    ),
    _case(
        "indexed-static-group-input",
        GemmType.INDEXED,
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        input_scale_group_size=16,
        input_quant_mode="static_tensor_dynamic_group",
        sm_version=current_device.sm_version if current_device.sm_version // 10 == 12 else 120,
    ),
    _case(
        "indexed-dynamic-group-token",
        GemmType.INDEXED,
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        input_scale_group_size=16,
        input_quant_mode="dynamic_group_token",
        sm_version=current_device.sm_version if current_device.sm_version // 10 == 12 else 120,
    ),
    _case("indexed-partial-k-tile", GemmType.INDEXED, shape_k=96),
    _case("grouped-contiguous", GemmType.GROUPED_CONTIGUOUS),
    _case(
        "grouped-contiguous-dynamic-group-token",
        GemmType.GROUPED_CONTIGUOUS,
        use_m_major_input_scale=True,
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        input_scale_group_size=16,
        input_quant_mode="dynamic_group_token",
        sm_version=current_device.sm_version if current_device.sm_version // 10 == 12 else 120,
    ),
    _case("grouped-masked", GemmType.GROUPED_MASKED),
    _case(
        "grouped-masked-dynamic-group-token",
        GemmType.GROUPED_MASKED,
        use_m_major_input_scale=True,
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        input_scale_group_size=16,
        input_quant_mode="dynamic_group_token",
        sm_version=current_device.sm_version if current_device.sm_version // 10 == 12 else 120,
    ),
    _case(
        "indexed-bias-pad-k",
        GemmType.INDEXED,
        has_bias=True,
        pad_shape_k=32,
    ),
    _case(
        "grouped-contiguous-pad-n",
        GemmType.GROUPED_CONTIGUOUS,
        pad_shape_n=24,
    ),
    _case(
        "grouped-masked-bias-pad-nk",
        GemmType.GROUPED_MASKED,
        has_bias=True,
        pad_shape_n=24,
        pad_shape_k=32,
    ),
    _case(
        "m-major-input-scale-grouped-masked",
        GemmType.GROUPED_MASKED,
        use_m_major_input_scale=True,
        a_dtype=dtypes.float8e4m3,
        input_scale_group_size=64,
        weight_scale_group_size=64,
    ),
    _case(
        "m-major-input-scale-grouped-contiguous",
        GemmType.GROUPED_CONTIGUOUS,
        use_m_major_input_scale=True,
        a_dtype=dtypes.float8e4m3,
        input_scale_group_size=64,
        weight_scale_group_size=64,
    ),
)


@pytest.mark.parametrize("test_case", MOE_CASES, ids=str)
def test_moe(test_case):
    config = test_case.layer_config
    assert config.num_experts == NUM_EXPERTS
    assert test_case.compute_config.gemm_type != GemmType.DENSE
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=get_default_mma_type(config).value)
    results = KernelTestRunner(test_case).run()
    if test_case.compute_config.gemm_type == GemmType.INDEXED:
        assert all(not result.tuning_config.use_tma_as for result in results)
        assert all(not result.tuning_config.use_tma_as2 for result in results)
    assert_kernel_test_shape_coverage(results)


def test_moe_case_coverage():
    assert {case.compute_config.gemm_type for case in MOE_CASES} == {
        GemmType.INDEXED,
        GemmType.GROUPED_CONTIGUOUS,
        GemmType.GROUPED_MASKED,
    }
    assert any(case.layer_config.has_bias for case in MOE_CASES)
    assert any(case.layer_config.pad_shape_n for case in MOE_CASES)
    assert any(case.layer_config.pad_shape_k for case in MOE_CASES)
    assert {
        case.compute_config.gemm_type for case in MOE_CASES if case.compute_config.use_m_major_input_scale
    } == {
        GemmType.GROUPED_CONTIGUOUS,
        GemmType.GROUPED_MASKED,
    }


def test_indexed_mma_tma_scale_pipeline(monkeypatch):
    skip_if_unsupported(a_dtype=dtypes.bfloat16, mma_type="mma", use_tma=True)
    case = next(case for case in MOE_CASES if case.name == "indexed-partial-k-tile")
    config = dict(
        mma_type="mma",
        block_shape=(32, 128, 32),
        warp_shape=(16, 64, 32),
        num_stages=4,
        num_ctas_per_sm=1,
        use_tma=True,
        use_tma_a=False,
        use_tma_b=False,
        use_tma_bs=True,
        use_tma_c=False,
        use_warp_spec=False,
        use_mbarrier=True,
        use_cp_async=False,
        use_stream_k=False,
        output_chunk_rows=32,
        smem_reuse_mode="none",
    )
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "heuristic")
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: dict(config))
    shape_ms = (1, 17, 257)
    results = KernelTestRunner(case).run(shape_ms)
    assert_kernel_test_shape_coverage(results, shape_ms)


@pytest.mark.parametrize("raster_group_m", [1, 8])
@pytest.mark.parametrize("use_stream_k", [False, True])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_grouped_contiguous_skips_unused_capacity(use_stream_k, offset_dtype, raster_group_m):
    """Graph replay uses live expert offsets and leaves unused rows untouched."""
    skip_if_unsupported(a_dtype=dtypes.bfloat16)
    layer = LayerConfig(
        shape_n=512,
        shape_k=256,
        num_experts=4,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        weight_scale_group_size=32,
    )
    compute = ComputeConfig(gemm_type=GemmType.GROUPED_CONTIGUOUS)
    runner = KernelTestRunner(
        KernelTestCase(
            name="grouped-contiguous-unused-capacity",
            layer_config=layer,
            compute_config=compute,
            seed=2026,
            weight_std_scale=256**-0.5,
        )
    )
    tuning = get_heuristics_config(layer, shape_m=64, gemm_type=compute.gemm_type)
    tuning["use_stream_k"] = use_stream_k
    tuning["raster_group_m"] = raster_group_m
    kernel = HummingKernel.prepare_kernels(
        layer.to_str(), compute.to_str(), tuning, device=torch.device("cuda", 0)
    )
    inputs = torch.empty(64, 256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(64, 512, device="cuda", dtype=torch.bfloat16)
    offsets = torch.zeros(5, device="cuda", dtype=offset_dtype)
    locks = torch.zeros(1024, device="cuda", dtype=torch.int32)

    def launch():
        ops.launch_kernel(
            configs=kernel,
            inputs=inputs,
            outputs=output,
            expert_layout=offsets,
            valid_shape_m=64,
            locks=locks,
            **runner.kernel_tensors,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for counts in ([0, 0, 0, 0], [0, 0, 0, 7], [13, 0, 5, 0], [3, 17, 0, 21], [16, 16, 16, 16]):
        offsets.copy_(torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=offset_dtype))
        inputs.normal_()
        inputs[sum(counts) :] = float("nan")
        output.fill_(-123)
        graph.replay()
        start = 0
        for expert, count in enumerate(counts):
            expert_input = inputs[start : start + count].float()
            reference = (expert_input @ runner.weight_ref[expert].float().T).bfloat16()
            torch.testing.assert_close(output[start : start + count], reference, atol=0.01, rtol=0.01)
            start += count
        assert (output[start:] == -123).all()


@pytest.mark.parametrize(
    "gemm_type",
    (GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED),
)
@pytest.mark.parametrize("multicast", (1, 2))
@pytest.mark.parametrize("use_stream_k", (False, True))
def test_raw_wgmma_tma_b_general_moe(gemm_type, multicast, use_stream_k, monkeypatch):
    skip_if_unsupported(a_dtype=dtypes.float8e4m3, mma_type="wgmma")
    case = _case(
        "raw-wgmma-tma-b-general",
        gemm_type,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float8e4m3,
        has_bias=True,
    )
    tuning = dict(
        mma_type="wgmma",
        block_shape=(16, 256, 128),
        warp_shape=(16, 32, 128),
        num_stages=3,
        use_tma=True,
        use_warp_spec=True,
        use_stream_k=use_stream_k,
        multi_cast_size_b=multicast,
        use_tma_a=gemm_type != GemmType.INDEXED,
        use_tma_c=gemm_type != GemmType.INDEXED,
    )
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: dict(tuning))
    if multicast > 1:
        with pytest.raises(AssertionError, match="B multicast requires dense GEMM"):
            KernelTestRunner(case).run((1,))
        return
    results = KernelTestRunner(case).run((1, 17, 129))
    assert_kernel_test_shape_coverage(results, (1, 17, 129))


@pytest.mark.parametrize("gemm_type", (GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED))
@pytest.mark.parametrize("a_dtype,b_dtype", (("float8e4m3", "float8e4m3"), ("bfloat16", "uint4")))
@pytest.mark.parametrize("pad_k", (0, 128, 32))
@pytest.mark.parametrize("use_stream_k", (False, True))
def test_wgmma_tma_a_general_moe(gemm_type, a_dtype, b_dtype, pad_k, use_stream_k, monkeypatch):
    skip_if_unsupported(a_dtype=a_dtype, mma_type="wgmma")
    case = _case(
        "wgmma-tma-a-general-moe",
        gemm_type,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        shape_k=1024,
        pad_shape_k=pad_k,
        has_bias=True,
    )
    tuning = dict(
        mma_type="wgmma",
        block_shape=(16, 128, 256),
        warp_shape=(16, 32, 64 if a_dtype == "bfloat16" else 128),
        num_stages=3,
        use_tma=True,
        use_warp_spec=True,
        use_stream_k=use_stream_k,
    )
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: dict(tuning))
    results = KernelTestRunner(case).run((1, 17, 129))
    assert_kernel_test_shape_coverage(results, (1, 17, 129))
