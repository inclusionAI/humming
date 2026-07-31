import pytest

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig
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
    _case("grouped-contiguous", GemmType.GROUPED_CONTIGUOUS),
    _case("grouped-masked", GemmType.GROUPED_MASKED),
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
    assert_kernel_test_shape_coverage(results)


def test_grouped_masked_m_major_rejects_unaligned_expert_m():
    test_case = _case(
        "m-major-grouped-masked-unaligned",
        GemmType.GROUPED_MASKED,
        use_m_major_input_scale=True,
        expert_max_tokens=2,
        a_dtype=dtypes.float8e4m3,
        input_scale_group_size=64,
        weight_scale_group_size=64,
    )
    with pytest.raises(RuntimeError, match="input-scale M dimension 16-byte aligned"):
        KernelTestRunner(test_case).run((1,))


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
        case.compute_config.gemm_type
        for case in MOE_CASES
        if case.compute_config.use_m_major_input_scale
    } == {GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED}
