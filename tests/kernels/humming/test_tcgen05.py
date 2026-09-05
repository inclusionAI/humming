import dataclasses

import pytest
import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType
from humming.config.config import _cuda_compiler_version
from humming.jit.runtime import KernelRuntime
from humming.kernel.humming import HummingKernel
from humming.layer import HummingLayer
from humming.schema import HummingWeightSchema
from humming.testing import KernelTestCase, KernelTestRunner
from humming.testing.data import generate_moe_tensors, generate_random_tensor
from humming.tune import get_heuristics_config
from humming.tune.sm100 import Sm100Heuristics

WEIGHT_CONFIGS = {
    "uint4": dict(b_dtype="uint4", weight_scale_group_size=128),
    "uint4-zp": dict(b_dtype="uint4", weight_scale_group_size=128, has_zero_point=True),
    "uint4-fp-zp": dict(
        b_dtype="uint4",
        weight_scale_group_size=128,
        has_zero_point=True,
        is_fp_zero_point=True,
    ),
    "nvfp4": dict(
        b_dtype="float4e2m1",
        bs_dtype="float8e4m3",
        weight_scale_group_size=16,
        weight_scale_2_type="tensor",
    ),
    "mxfp4": dict(
        b_dtype="float4e2m1",
        bs_dtype="float8e8m0",
        weight_scale_group_size=32,
    ),
    "fp8": dict(b_dtype="float8e4m3"),
}


@pytest.fixture(autouse=True)
def require_sm100_family(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("TCGen5 BF16 requires an SM100-family GPU")
    if _cuda_compiler_version(KernelRuntime._get_compiler()) < (12, 9):
        pytest.skip("TCGen5 sm100f requires CUDA 12.9 or newer")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    def force_tcgen05(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_tcgen05_config(layer_config, shape_m, gemm_type) | {
            "mma_type": "tcgen05"
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", force_tcgen05)


def _case(name, gemm_type, **weight_values):
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=256,
            shape_k=256,
            num_experts=0 if gemm_type == GemmType.DENSE else 4,
            a_dtype=dtypes.bfloat16,
            c_dtype=dtypes.bfloat16,
            mma_type=MmaType.TCGEN05,
            **(dict(bs_dtype="bfloat16") | weight_values),
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type),
        top_k=2,
        seed=2026,
    )


def _assert_results(case, shape_ms):
    runner = KernelTestRunner(case)
    kernels = runner.prepare_kernels(shape_ms)
    assert all(
        HummingKernel._id2kernel[int(kernel[0][2])].mma_type == MmaType.TCGEN05
        for variants in kernels.values()
        for kernel in variants
    )
    results = runner.run(shape_ms)
    assert {result.shape_m for result in results} == set(shape_ms)
    for result in results:
        torch.testing.assert_close(
            result.outputs, result.outputs_ref, rtol=case.rtol, atol=case.atol
        )


@pytest.mark.parametrize("gemm_type", list(GemmType))
@pytest.mark.parametrize("weight_name", WEIGHT_CONFIGS)
def test_tcgen05_common_weights_and_moe(weight_name, gemm_type):
    """Decode, tile tails, and prefill use the same quantization contract."""
    case = _case(weight_name, gemm_type, **WEIGHT_CONFIGS[weight_name])
    _assert_results(case, (1, 17, 65, 257))


@pytest.mark.parametrize(
    "gemm_type,block_m,shape_k",
    (
        (GemmType.DENSE, 64, 64),
        (GemmType.INDEXED, 64, 128),
        (GemmType.INDEXED, 128, 256),
        (GemmType.GROUPED_CONTIGUOUS, 64, 192),
        (GemmType.GROUPED_MASKED, 128, 448),
    ),
)
def test_tcgen05_pipeline_stage_reuse(gemm_type, block_m, shape_k, monkeypatch):
    """Retire async reads before reuse across persistent tiles and experts."""
    weights = WEIGHT_CONFIGS["uint4-zp"] | {"weight_scale_group_size": 64}
    case = _case("pipeline-stage-reuse", gemm_type, **weights)
    case = dataclasses.replace(
        case, layer_config=dataclasses.replace(case.layer_config, shape_k=shape_k)
    )

    def minimum_stages(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_tcgen05_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (block_m, 128, 64),
            "warp_shape": (block_m, 32, 64),
            "num_stages": 3,
            "num_sms": 2,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", minimum_stages)
    _assert_results(case, (17, 257))


@pytest.mark.parametrize("gemm_type", list(GemmType))
@pytest.mark.parametrize(
    "b_dtype",
    (
        "float3e1m1",
        "float3e2m0",
        "float4e3m0",
        "float5e2m2",
        "float5e4m0",
        "float6e2m3",
        "float6e3m2",
        "float6e4m1",
        "float7e2m4",
        "float7e4m2",
        "float7e6m0",
        "float8e1m6",
        "float8e3m4",
        "float8e5m2",
    ),
)
def test_tcgen05_floating_weight_formats(b_dtype, gemm_type):
    case = _case(b_dtype, gemm_type, b_dtype=b_dtype)
    _assert_results(case, (17, 129))


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("gemm_type", list(GemmType))
def test_tcgen05_integer_weight_widths(bits, gemm_type):
    case = _case(
        f"uint{bits}",
        gemm_type,
        b_dtype=f"uint{bits}",
        weight_scale_group_size=64,
        has_zero_point=True,
    )
    _assert_results(case, (17, 129))


@pytest.mark.parametrize("gemm_type", list(GemmType))
@pytest.mark.parametrize(
    "weight_values",
    [
        dict(b_dtype="uint3", bs_dtype="float8e5m2", weight_scale_group_size=64),
        dict(
            b_dtype="uint3", weight_scale_group_size=64, weight_scale_2_type="channel"
        ),
        dict(b_dtype="uint4", bs_dtype="float32", weight_scale_type="tensor"),
        dict(
            b_dtype="uint4",
            bs_dtype="float32",
            weight_scale_group_size=64,
            weight_scale_group_size_n=64,
            weight_scale_type="block",
        ),
    ],
    ids=("fp8-scale", "channel-secondary-scale", "tensor-scale", "block-scale"),
)
def test_tcgen05_scale_contract(weight_values, gemm_type):
    case = _case("scale", gemm_type, **weight_values)
    _assert_results(case, (17, 257))


def _public_problem(weight_ref, shape_m, gemm_type, block_m):
    device = weight_ref.device
    shape_k = weight_ref.shape[-1]
    if gemm_type == GemmType.DENSE:
        inputs = generate_random_tensor(
            (shape_m, shape_k), torch.bfloat16, device=device
        )
        return dict(inputs=inputs), slice(None), inputs.float() @ weight_ref.T

    # Leave experts 1 and 3 empty; experts 0 and 2 have tail tiles.
    topk_ids = torch.tensor([0, 2], device=device, dtype=torch.int32)
    topk_ids = topk_ids.expand(shape_m, -1).contiguous()
    _, layout, sorted_ids, expert_ids, padded = generate_moe_tensors(
        topk_ids,
        4,
        gemm_type,
        block_size_config=block_m,
        expert_max_tokens=shape_m + 3,
    )
    if gemm_type == GemmType.INDEXED:
        inputs = generate_random_tensor(
            (shape_m, shape_k), torch.bfloat16, device=device
        )
        reference = torch.stack(
            [inputs.float() @ weight_ref[e].T for e in (0, 2)], dim=1
        ).flatten(0, 1)
        return (
            dict(
                inputs=inputs,
                sorted_ids=sorted_ids,
                expert_ids=expert_ids,
                num_tokens_padded=padded,
                top_k=2,
            ),
            slice(None),
            reference,
        )

    total_m = (
        shape_m * 2 if gemm_type == GemmType.GROUPED_CONTIGUOUS else 4 * (shape_m + 3)
    )
    inputs = generate_random_tensor((total_m, shape_k), torch.bfloat16, device=device)
    output_ids, references = [], []
    for expert in (0, 2):
        offset = (
            int(layout[expert])
            if gemm_type == GemmType.GROUPED_CONTIGUOUS
            else expert * (shape_m + 3)
        )
        ids = torch.arange(offset, offset + shape_m, device=device)
        output_ids.append(ids)
        references.append(inputs[ids].float() @ weight_ref[expert].T)
    return (
        dict(inputs=inputs, expert_layout=layout, valid_shape_m=shape_m * 2),
        torch.cat(output_ids),
        torch.cat(references),
    )


def _public_layer(weight_name, gemm_type, shape_n=256, shape_k=256):
    torch.manual_seed(2026)
    schema = HummingWeightSchema(**WEIGHT_CONFIGS[weight_name])
    num_experts = 0 if gemm_type == GemmType.DENSE else 4
    weight_shape = (4, shape_n, shape_k) if num_experts else (shape_n, shape_k)
    weight = generate_random_tensor(weight_shape, torch.bfloat16, device="cuda")
    tensors = schema.quant_tensor(weight, schema, torch.bfloat16)
    if num_experts and "weight_scale_2" in tensors:
        tensors["weight_scale_2"] *= torch.arange(
            1, num_experts + 1, device=weight.device
        ).reshape_as(tensors["weight_scale_2"])
    weight_ref = schema.dequant_tensors(tensors)
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        weight_config=schema,
        input_config={"dtype": "bfloat16"},
        torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_state_dict(tensors, strict=False)
    layer.transform()
    return layer, weight_ref


def _selected_backend(layer, gemm_type, kwargs, tuning=None):
    prepared = HummingKernel.prepare_kernels(
        layer.humming_config.to_str(),
        {"gemm_type": gemm_type.value},
        tuning,
    ).reshape(-1, 4)
    dispatch_m = kwargs.get("valid_shape_m", 0) or kwargs["inputs"].shape[0] * (
        kwargs.get("top_k", 1) if gemm_type == GemmType.INDEXED else 1
    )
    selected = [row for row in prepared if int(row[0]) < dispatch_m <= int(row[1])]
    assert len(selected) == 1
    return HummingKernel._id2kernel[int(selected[0][2])].mma_type


@pytest.mark.parametrize("gemm_type", list(GemmType))
@pytest.mark.parametrize("weight_name", WEIGHT_CONFIGS)
def test_tcgen05_public_layer_switches_without_repacking(weight_name, gemm_type):
    """MMA and TCGen5 consume one transformed layer, including routed calls."""
    layer, weight_ref = _public_layer(weight_name, gemm_type)
    num_experts = layer.num_experts
    packed = {
        name: (value.data_ptr(), value.detach().view(torch.uint8).clone())
        for name, value in layer.named_parameters()
    }
    for shape_m, mma_type in (
        (17, None),
        (257, MmaType.TCGEN05),
        (257, MmaType.MMA),
        (17, None),
    ):
        torch.manual_seed(2026 + shape_m)
        config = (
            layer.humming_config
            if mma_type is None
            else dataclasses.replace(layer.humming_config, mma_type=mma_type)
        )
        get_config = (
            Sm100Heuristics.get_tcgen05_config
            if mma_type == MmaType.TCGEN05
            else get_heuristics_config
        )
        tuning = get_config(
            config,
            shape_m=shape_m * (2 if num_experts else 1),
            gemm_type=gemm_type,
        )
        if mma_type is not None:
            tuning |= {"mma_type": mma_type.value}
        kwargs, output_ids, reference = _public_problem(
            weight_ref, shape_m, gemm_type, tuning["block_shape"][0]
        )
        outputs = layer(
            **kwargs,
            compute_config={"gemm_type": gemm_type.value},
            tuning_config=tuning if mma_type is not None else None,
        )
        actual_mma = _selected_backend(
            layer, gemm_type, kwargs, tuning if mma_type is not None else None
        )
        assert actual_mma == (mma_type or MmaType.MMA)
        torch.testing.assert_close(
            outputs[output_ids], reference.to(torch.bfloat16), rtol=0.01, atol=0.05
        )
    for name, value in layer.named_parameters():
        pointer, original = packed[name]
        assert pointer == value.data_ptr()
        assert torch.equal(value.detach().view(torch.uint8), original)


@pytest.mark.parametrize(
    "gemm_type", (GemmType.DENSE, GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS)
)
def test_tcgen05_default_prefill_switches_back_to_decode(gemm_type):
    """The public default selects TCGen5 for prefill on a reused packed layer."""
    layer, weight_ref = _public_layer("uint4", gemm_type, 5120, 2048)
    packed = {
        name: (value.data_ptr(), value.detach().view(torch.uint8).clone())
        for name, value in layer.named_parameters()
    }
    for shape_m, expected in (
        (17, MmaType.MMA),
        (1024, MmaType.TCGEN05),
        (17, MmaType.MMA),
    ):
        tuning = get_heuristics_config(
            layer.humming_config,
            shape_m=shape_m * (2 if layer.humming_config.num_experts else 1),
            gemm_type=gemm_type,
        )
        kwargs, output_ids, reference = _public_problem(
            weight_ref, shape_m, gemm_type, tuning["block_shape"][0]
        )
        outputs = layer(**kwargs, compute_config={"gemm_type": gemm_type.value})
        assert _selected_backend(layer, gemm_type, kwargs) == expected
        torch.testing.assert_close(
            outputs[output_ids], reference.to(torch.bfloat16), rtol=0.01, atol=0.05
        )
    for name, value in layer.named_parameters():
        pointer, original = packed[name]
        assert pointer == value.data_ptr()
        assert torch.equal(value.detach().view(torch.uint8), original)


def test_tcgen05_grouped_metadata_limits_cta_residency():
    """Retain one CTA when scales and expert metadata exhaust shared memory."""
    layer = LayerConfig(
        shape_n=2048,
        shape_k=2048,
        num_experts=256,
        a_dtype=dtypes.bfloat16,
        c_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=16,
        weight_scale_2_type="channel",
        has_zero_point=True,
        is_fp_zero_point=True,
        has_bias=True,
        mma_type=MmaType.TCGEN05,
    )
    for group_size, expected_ctas in ((16, 1), (128, 2)):
        tuning = Sm100Heuristics.get_tcgen05_config(
            dataclasses.replace(layer, weight_scale_group_size=group_size),
            shape_m=128 * layer.num_experts,
            gemm_type=GemmType.GROUPED_CONTIGUOUS,
        )
        assert tuning["num_ctas_per_sm"] == expected_ctas
        assert tuning["num_stages"] == 5
