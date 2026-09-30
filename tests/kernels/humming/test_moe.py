import pytest
import torch

from humming import dtypes, ops
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType
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
            shape_k=SHAPE_K,
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
        mma_type=MmaType.MXMMA,
    ),
    _case(
        "indexed-dynamic-group-token",
        GemmType.INDEXED,
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        input_scale_group_size=16,
        input_quant_mode="dynamic_group_token",
        mma_type=MmaType.MXMMA,
    ),
    _case("grouped-contiguous", GemmType.GROUPED_CONTIGUOUS),
    _case(
        "grouped-contiguous-dynamic-group-token",
        GemmType.GROUPED_CONTIGUOUS,
        use_m_major_input_scale=True,
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        input_scale_group_size=16,
        input_quant_mode="dynamic_group_token",
        mma_type=MmaType.MXMMA,
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
        mma_type=MmaType.MXMMA,
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
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
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


@pytest.mark.parametrize("use_stream_k", [False, True])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_grouped_contiguous_skips_unused_capacity(use_stream_k, offset_dtype):
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
