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
NUM_EXPERTS = 8
TOP_K = 2


def _case(
    name: str,
    *,
    a_dtype,
    b_dtype,
    bs_dtype,
    group_size: int,
    c_dtype=dtypes.bfloat16,
    has_zero_point: bool = False,
    input_group_size: int | None = None,
    weight_group_size: int | None = None,
    gemm_type: GemmType = GemmType.DENSE,
) -> KernelTestCase:
    input_group_size = group_size if input_group_size is None else input_group_size
    weight_group_size = group_size if weight_group_size is None else weight_group_size
    is_dense = gemm_type == GemmType.DENSE
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=SHAPE_N,
            shape_k=SHAPE_K,
            num_experts=0 if is_dense else NUM_EXPERTS,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            bs_dtype=bs_dtype,
            input_scale_group_size=input_group_size,
            weight_scale_group_size=weight_group_size,
            has_zero_point=has_zero_point,
            mma_type=MmaType.MXMMA,
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type),
        top_k=1 if is_dense else TOP_K,
        seed=2026,
    )


MXMMA_FORMAT_CASES = (
    _case(
        "e3m4-fp4-e8m0-g32",
        a_dtype=dtypes.float8e3m4,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e4m3-fp4-e8m0-g32-native",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e5m2-fp4-e8m0-g32-native",
        a_dtype=dtypes.float8e5m2,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e4m3-f6e2m3-e8m0-g32-native",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float6e2m3,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e5m2-f6e3m2-e8m0-g32-native",
        a_dtype=dtypes.float8e5m2,
        b_dtype=dtypes.float6e3m2,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e4m3-e4m3-e8m0-g32",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float8e4m3,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e2m1-e2m1-e8m0-g32",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    ),
    _case(
        "e4m3-e2m1-channel-input",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        input_group_size=0,
    ),
    _case(
        "e2m1-e2m1-channel-weight",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.bfloat16,
        group_size=32,
        weight_group_size=0,
    ),
    _case(
        "e2m1-e2m1-channel-input-channel-weight",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.bfloat16,
        group_size=32,
        input_group_size=0,
        weight_group_size=0,
    ),
    _case(
        "e4m3-e2m1-channel-input-indexed",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        input_group_size=0,
        gemm_type=GemmType.INDEXED,
    ),
    _case(
        "e2m1-e2m1-e4m3-g16",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e4m3,
        group_size=16,
    ),
    _case(
        "e0m3-e0m3-e8m0-g16",
        a_dtype=dtypes.float4e0m3,
        b_dtype=dtypes.float4e0m3,
        bs_dtype=dtypes.float8e8m0,
        group_size=16,
    ),
    _case(
        "e0m3-uint3-e4m3-g16",
        a_dtype=dtypes.float4e0m3,
        b_dtype=dtypes.uint3,
        bs_dtype=dtypes.float8e4m3,
        group_size=16,
    ),
)

MXMMA_ZERO_POINT_CASES = (
    _case(
        "e3m4-uint5-e8m0-g32-zp",
        a_dtype=dtypes.float8e3m4,
        b_dtype=dtypes.uint5,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        has_zero_point=True,
    ),
    _case(
        "e4m3-uint4-e8m0-g32-zp-fp16-output",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.uint4,
        c_dtype=dtypes.float16,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        has_zero_point=True,
    ),
    _case(
        "e5m2-uint3-e8m0-g32-zp",
        a_dtype=dtypes.float8e5m2,
        b_dtype=dtypes.uint3,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        has_zero_point=True,
    ),
    _case(
        "e2m1-uint2-e4m3-g16-zp",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.uint2,
        bs_dtype=dtypes.float8e4m3,
        group_size=16,
        has_zero_point=True,
    ),
    _case(
        "e2m1-uint2-e8m0-g32-zp",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.uint2,
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
        has_zero_point=True,
    ),
    _case(
        "e0m3-uint3-e4m3-g16-zp",
        a_dtype=dtypes.float4e0m3,
        b_dtype=dtypes.uint3,
        bs_dtype=dtypes.float8e4m3,
        group_size=16,
        has_zero_point=True,
    ),
)

MXMMA_CASES = MXMMA_FORMAT_CASES + MXMMA_ZERO_POINT_CASES


@pytest.mark.parametrize("test_case", MXMMA_CASES, ids=str)
def test_mxmma(test_case):
    config = test_case.layer_config
    assert config.mma_type == MmaType.MXMMA
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
    results = KernelTestRunner(test_case).run()
    for result in results:
        torch.testing.assert_close(
            result.outputs,
            result.outputs_ref,
            rtol=test_case.rtol,
            atol=test_case.atol,
        )
    assert_kernel_test_shape_coverage(results)


def test_mxmma_case_coverage():
    assert all(case.layer_config.mma_type == MmaType.MXMMA for case in MXMMA_CASES)
    assert {case.layer_config.a_dtype for case in MXMMA_CASES} == {
        dtypes.float4e0m3,
        dtypes.float4e2m1,
        dtypes.float8e3m4,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    }
    assert {case.layer_config.bs_dtype for case in MXMMA_CASES} == {
        dtypes.bfloat16,
        dtypes.float8e4m3,
        dtypes.float8e8m0,
    }
    assert {case.layer_config.weight_scale_group_size for case in MXMMA_CASES} == {
        0,
        16,
        32,
    }
    assert any(case.layer_config.mxmma_native_mixed for case in MXMMA_CASES)

    assert len(MXMMA_ZERO_POINT_CASES) == 6
    assert all(case.layer_config.has_zero_point for case in MXMMA_ZERO_POINT_CASES)
    assert {case.layer_config.a_dtype for case in MXMMA_ZERO_POINT_CASES} == {
        dtypes.float4e0m3,
        dtypes.float4e2m1,
        dtypes.float8e3m4,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    }


def _require_sm100_mxmma():
    from humming.config.config import _cuda_compiler_version
    from humming.jit.runtime import KernelRuntime

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native block-scaled UMMA requires SM100 family")
    if _cuda_compiler_version(KernelRuntime._get_compiler()) < (12, 9):
        pytest.skip("Native block-scaled UMMA requires CUDA 12.9 or newer")


@pytest.mark.parametrize("gemm_type", list(GemmType))
@pytest.mark.parametrize(
    "group_size,scale_dtype",
    ((32, dtypes.float8e8m0), (16, dtypes.float8e4m3)),
    ids=("mxfp4", "nvfp4"),
)
def test_sm100_mxmma_formats_and_moe(gemm_type, group_size, scale_dtype):
    """Native FP4 preserves group scales through routing and stage reuse."""
    import dataclasses

    from humming.kernel.humming import HummingKernel

    _require_sm100_mxmma()
    case = _case(
        "sm100-fp4",
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=scale_dtype,
        group_size=group_size,
        gemm_type=gemm_type,
    )
    case = dataclasses.replace(
        case,
        layer_config=dataclasses.replace(
            case.layer_config,
            shape_n=256,
            shape_k=768,
            num_experts=0 if gemm_type == GemmType.DENSE else 4,
        ),
    )
    runner = KernelTestRunner(case)
    shape_ms = (1, 17, 65, 257)
    kernels = runner.prepare_kernels(shape_ms)
    assert all(
        HummingKernel._id2kernel[int(kernel[0][2])].use_mxumma
        for variants in kernels.values()
        for kernel in variants
    )
    for result in runner.run(shape_ms):
        torch.testing.assert_close(
            result.outputs, result.outputs_ref, rtol=case.rtol, atol=case.atol
        )


@pytest.mark.parametrize(
    "b_dtype,zero_point,input_group,weight_group,scale_dtype,secondary_scale",
    (
        (dtypes.uint1, False, 32, 32, dtypes.float8e8m0, None),
        (dtypes.uint2, True, 16, 16, dtypes.float8e4m3, None),
        (dtypes.uint3, False, 32, 32, dtypes.float8e8m0, None),
        (dtypes.float4e2m1, False, 0, 32, dtypes.float8e8m0, None),
        (dtypes.float4e2m1, False, 32, 0, dtypes.bfloat16, None),
        (dtypes.float4e2m1, False, 16, 16, dtypes.float8e4m3, "tensor"),
    ),
)
def test_sm100_mxmma_weight_contract(
    b_dtype, zero_point, input_group, weight_group, scale_dtype, secondary_scale
):
    """Register dequantization and channel scales retain their FP4 contract."""
    import dataclasses

    _require_sm100_mxmma()
    case = _case(
        "sm100-weight-contract",
        a_dtype=dtypes.float4e2m1,
        b_dtype=b_dtype,
        bs_dtype=scale_dtype,
        group_size=32,
        input_group_size=input_group,
        weight_group_size=weight_group,
        has_zero_point=zero_point,
        gemm_type=GemmType.INDEXED,
    )
    case = dataclasses.replace(
        case,
        layer_config=dataclasses.replace(
            case.layer_config,
            shape_n=256,
            shape_k=256,
            num_experts=4,
            weight_scale_2_type=secondary_scale,
        ),
    )
    for result in KernelTestRunner(case).run((17, 129)):
        torch.testing.assert_close(
            result.outputs, result.outputs_ref, rtol=case.rtol, atol=case.atol
        )


@pytest.mark.parametrize("gemm_type", (GemmType.DENSE, GemmType.INDEXED))
@pytest.mark.parametrize("shape_k", (256, 2880))
def test_sm100_mxmma_public_layer_reuses_packed_weights(gemm_type, shape_k):
    """Packed scale strides survive K padding and prefill followed by decode."""
    from humming import ops
    from humming.forward import may_quant_input
    from humming.kernel.humming import HummingKernel
    from humming.layer import HummingLayer
    from humming.schema import HummingWeightSchema
    from humming.testing.data import generate_moe_tensors, generate_random_tensor
    from humming.tune import get_heuristics_config

    _require_sm100_mxmma()
    torch.manual_seed(2026)
    num_experts = 0 if gemm_type == GemmType.DENSE else 4
    schema = HummingWeightSchema(
        b_dtype="float4e2m1",
        bs_dtype="float8e8m0",
        weight_scale_group_size=32,
    )
    weight_shape = (4, 256, shape_k) if num_experts else (256, shape_k)
    weight = generate_random_tensor(weight_shape, torch.bfloat16, device="cuda")
    tensors = schema.quant_tensor(weight, schema, torch.bfloat16)
    weight_ref = schema.dequant_tensors(tensors)
    layer = HummingLayer(
        shape_n=256,
        shape_k=shape_k,
        pad_k_to_multiple=128,
        num_experts=num_experts,
        weight_config=schema,
        input_config={"dtype": "float4e2m1", "group_size": 32},
        torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_state_dict(tensors, strict=False)
    layer.transform()
    packed = {
        name: (value.data_ptr(), value.detach().view(torch.uint8).clone())
        for name, value in layer.named_parameters()
    }
    for shape_m in (17, 257, 1):
        inputs = generate_random_tensor((shape_m, shape_k), torch.bfloat16, device="cuda")
        quant, scales, _ = ops.process_input(
            inputs,
            quant_mode="dynamic_group",
            quant_dtype="float4e2m1",
            quant_group_size=32,
            group_scale_dtype="float8e8m0",
        )
        codes = ops.unpack_weight(quant.view(torch.int32), 4)
        inputs_ref = ops.dequant_weight(codes, 2, 1, True).float()
        inputs_ref *= scales.float().repeat_interleave(32, 1)
        kwargs = {}
        dispatch_m = shape_m
        if num_experts:
            ids = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
            ids = ids.expand(shape_m, -1).contiguous()
            tuning = get_heuristics_config(
                layer.humming_config,
                shape_m * 2,
                gemm_type=gemm_type,
            )
            _, _, sorted_ids, expert_ids, padded = generate_moe_tensors(
                ids,
                4,
                gemm_type,
                block_size_config=tuning["block_shape"][0],
            )
            kwargs = dict(
                sorted_ids=sorted_ids,
                expert_ids=expert_ids,
                num_tokens_padded=padded,
                top_k=2,
            )
            reference = torch.stack(
                [inputs_ref @ weight_ref[e].T for e in (0, 2)],
                dim=1,
            ).flatten(0, 1)
            dispatch_m *= 2
        else:
            reference = inputs_ref @ weight_ref.T
        outputs = layer(inputs, **kwargs, compute_config={"gemm_type": gemm_type.value})
        torch.testing.assert_close(
            outputs,
            reference.to(torch.bfloat16),
            rtol=0.01,
            atol=0.05,
        )
        prequant, prescale = may_quant_input(layer.humming_config, inputs)
        prescale = prescale.view(torch.uint8)
        preserved_input, normalized_scale = may_quant_input(
            layer.humming_config,
            prequant,
            input_scale=prescale,
        )
        assert preserved_input.data_ptr() == prequant.data_ptr()
        assert normalized_scale.data_ptr() == prescale.data_ptr()
        assert normalized_scale.dtype == torch.int32
        prequant_outputs = layer(
            prequant,
            input_scale=prescale,
            **kwargs,
            compute_config={"gemm_type": gemm_type.value},
        )
        torch.testing.assert_close(
            prequant_outputs,
            reference.to(torch.bfloat16),
            rtol=0.01,
            atol=0.05,
        )
        prepared = HummingKernel.prepare_kernels(
            layer.humming_config.to_str(),
            {"gemm_type": gemm_type.value},
            None,
        ).reshape(-1, 4)
        selected = [row for row in prepared if int(row[0]) < dispatch_m <= int(row[1])]
        assert len(selected) == 1
        assert HummingKernel._id2kernel[int(selected[0][2])].use_mxumma
    for name, value in layer.named_parameters():
        pointer, original = packed[name]
        assert pointer == value.data_ptr()
        assert torch.equal(value.detach().view(torch.uint8), original)


@pytest.mark.parametrize("block_k", (128, 256))
@pytest.mark.parametrize(
    "group_size,scale_dtype", ((32, "float8e8m0"), (16, "float8e4m3"))
)
def test_sm100_mxmma_minimum_pipeline_stages(
    block_k, group_size, scale_dtype, monkeypatch
):
    """Retire operand and scale readers before wrapping the three-stage ring."""
    _require_sm100_mxmma()
    case = KernelTestCase(
        name="sm100-stage-reuse",
        layer_config=LayerConfig(
            shape_n=256,
            shape_k=block_k * 7,
            num_experts=4,
            a_dtype="float4e2m1",
            b_dtype="float4e2m1",
            c_dtype="bfloat16",
            bs_dtype=scale_dtype,
            input_scale_group_size=group_size,
            weight_scale_group_size=group_size,
        ),
        compute_config=ComputeConfig(gemm_type=GemmType.GROUPED_MASKED),
        top_k=2,
        seed=2026,
    )

    def minimum_stages(layer_config, shape_m, gemm_type, **kwargs):
        return {
            "mma_type": "mxmma",
            "block_shape": (64, 128, block_k),
            "warp_shape": (64, 32, block_k),
            "num_stages": 3,
            "num_sms": 2,
            "use_warp_spec": True,
            "use_tma": True,
            "use_stream_k": False,
            "use_pdl": False,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", minimum_stages)
    for result in KernelTestRunner(case).run((17, 257)):
        torch.testing.assert_close(
            result.outputs,
            result.outputs_ref,
            rtol=case.rtol,
            atol=case.atol,
        )


@pytest.mark.parametrize("compiler_version", ((12, 8), (12, 9)))
def test_sm100_mxmma_compiler_eligibility(compiler_version, monkeypatch):
    """Automatic FP4 dispatch requires the SM100-family compiler target."""
    monkeypatch.setattr(
        "humming.config.config._cuda_compiler_version",
        lambda compiler: compiler_version,
    )
    config = LayerConfig(
        sm_version=100,
        shape_n=256,
        shape_k=256,
        a_dtype="float4e2m1",
        b_dtype="float4e2m1",
        c_dtype="bfloat16",
        bs_dtype="float8e8m0",
        input_scale_group_size=32,
        weight_scale_group_size=32,
    )
    eligible = compiler_version >= (12, 9)
    assert config.mxmma_supported == eligible
    assert config.mma_type == (MmaType.MXMMA if eligible else MmaType.MMA)
