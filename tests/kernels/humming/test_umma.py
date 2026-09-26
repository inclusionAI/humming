import dataclasses

import pytest
import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType, SmemReuseMode
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
        pytest.skip("UMMA BF16 requires an SM100-family GPU")
    if _cuda_compiler_version(KernelRuntime._get_compiler()) < (12, 9):
        pytest.skip("UMMA sm100f requires CUDA 12.9 or newer")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    def force_umma(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {"mma_type": "umma"}

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", force_umma)


def _case(name, gemm_type, **weight_values):
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=256,
            shape_k=256,
            num_experts=0 if gemm_type == GemmType.DENSE else 4,
            a_dtype=dtypes.bfloat16,
            c_dtype=dtypes.bfloat16,
            mma_type=MmaType.UMMA,
            **(dict(bs_dtype="bfloat16") | weight_values),
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type),
        top_k=2,
        seed=2026,
    )


def _assert_results(case, shape_ms):
    runner = KernelTestRunner(case)
    kernels = runner.prepare_kernels(shape_ms)
    for variants in kernels.values():
        for kernel in variants:
            compiled = HummingKernel._id2kernel[int(kernel[0][2])]
            assert compiled.mma_type == MmaType.UMMA
            assert compiled.num_threads == 384
            assert compiled.num_math_threads == 128
            assert compiled.num_load_threads == 128
            compiled.assert_smem_size_matches_estimate()
    results = runner.run(shape_ms)
    assert {result.shape_m for result in results} == set(shape_ms)
    for result in results:
        torch.testing.assert_close(result.outputs, result.outputs_ref, rtol=case.rtol, atol=case.atol)


@pytest.mark.parametrize("block_n,block_k", ((256, 64), (128, 128)))
def test_umma_operand_wait_once_per_iteration(block_n, block_k, monkeypatch):
    # Odd stages must not wait again for a second output group or fragment.
    def select_config(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (24, block_n, block_k),
            "warp_shape": (24, 32, block_k),
            "num_stages": 3,
            "num_ctas_per_sm": 1,
            "use_stream_k": False,
            "use_tma": True,
            "use_tma_a": True,
            "use_tma_c": True,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_config)
    case = _case("operand-wait", GemmType.DENSE, **WEIGHT_CONFIGS["nvfp4"])
    case = dataclasses.replace(case, layer_config=dataclasses.replace(case.layer_config, shape_k=1024))
    _assert_results(case, (64, 257))


@pytest.mark.parametrize("gemm_type", list(GemmType))
@pytest.mark.parametrize("weight_name", WEIGHT_CONFIGS)
@pytest.mark.parametrize("block_n", (128, 256))
def test_umma_common_weights_and_moe(weight_name, gemm_type, block_n, monkeypatch):
    """Decode, tile tails, and prefill use the same quantization contract."""

    def select_block_n(layer_config, shape_m, gemm_type, **kwargs):
        tuning = Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type)
        block_m = tuning["block_shape"][0]
        return tuning | {
            "block_shape": (block_m, block_n, 64),
            "warp_shape": (block_m, 32, 64),
            "num_ctas_per_sm": 1,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_block_n)
    case = _case(weight_name, gemm_type, **WEIGHT_CONFIGS[weight_name])
    _assert_results(case, (1, 17, 65, 257))


@pytest.mark.parametrize(
    "block_m,block_n,warp_m,num_stages,use_tma_c",
    (
        (8, 128, 8, 2, False),
        (8, 256, 8, 2, True),
        (8, 128, 8, 3, False),
        (8, 256, 8, 3, True),
        (128, 64, 128, 3, True),
        (128, 64, 128, 3, False),
        (16, 64, 16, 3, True),
        (16, 128, 16, 3, True),
        (24, 128, 24, 3, True),
        (40, 128, 40, 3, False),
        (56, 256, 56, 3, True),
        (248, 128, 248, 3, True),
        (48, 128, 48, 3, True),
        (128, 128, 128, 3, True),
        (128, 128, 128, 3, False),
        (128, 128, 128, 5, True),
        (128, 256, 128, 3, True),
        (32, 512, 32, 3, True),
    ),
)
@pytest.mark.parametrize("output_dtype", (dtypes.bfloat16, dtypes.float16))
def test_umma_native_output_partitions(
    block_m, block_n, warp_m, num_stages, use_tma_c, output_dtype, monkeypatch
):
    """Native TMEM output covers supported M sizes and multiple N partitions."""
    case = _case("native-output-partitions", GemmType.DENSE, **WEIGHT_CONFIGS["uint4"])
    layer_config = dataclasses.replace(
        case.layer_config,
        shape_n=max(256, block_n),
        a_dtype=output_dtype,
        c_dtype=output_dtype,
        bs_dtype=output_dtype,
    )
    case = dataclasses.replace(case, layer_config=layer_config)

    def select_partitions(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (block_m, block_n, 64),
            "warp_shape": (warp_m, 32, 64),
            "num_stages": num_stages,
            "num_ctas_per_sm": 1,
            "use_tma_a": True,
            "use_tma_c": use_tma_c,
            "use_stream_k": False,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_partitions)
    _assert_results(case, (17, 257))


@pytest.mark.parametrize("output_dtype", (dtypes.bfloat16, dtypes.float16))
@pytest.mark.parametrize("block_m,block_n", ((8, 64), (128, 64), (128, 256)))
@pytest.mark.parametrize(
    "weight_values",
    (
        dict(b_dtype="uint4", weight_scale_group_size=128, has_bias=True),
        dict(b_dtype="uint4", weight_scale_type="channel"),
        dict(b_dtype="uint4", weight_scale_type="channel", has_bias=True),
        dict(b_dtype="uint4", bs_dtype="float8e8m0", weight_scale_type="channel"),
        dict(b_dtype="uint4", bs_dtype="float8e8m0", weight_scale_type="channel", has_bias=True),
        dict(b_dtype="uint4", bs_dtype="float8e4m3", weight_scale_type="channel"),
        dict(b_dtype="uint4", bs_dtype="float8e4m3", weight_scale_type="channel", has_bias=True),
        dict(b_dtype="uint4", weight_scale_group_size=128, weight_scale_2_type="channel"),
        dict(b_dtype="uint4", weight_scale_group_size=128, weight_scale_2_type="channel", has_bias=True),
    ),
)
def test_umma_native_output_channel_and_bias(weight_values, block_m, block_n, output_dtype, monkeypatch):
    """Native output applies each column's scale and bias after TMEM conversion."""
    case = _case("native-output-channel-bias", GemmType.DENSE, **weight_values)
    layer_config = dataclasses.replace(
        case.layer_config,
        shape_n=256,
        a_dtype=output_dtype,
        c_dtype=output_dtype,
        bs_dtype=case.layer_config.bs_dtype if "bs_dtype" in weight_values else output_dtype,
    )
    case = dataclasses.replace(case, layer_config=layer_config)

    def select_native_output(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (block_m, block_n, 64),
            "warp_shape": (block_m, 32, 64),
            "num_stages": 3,
            "num_ctas_per_sm": 1,
            "use_tma_a": True,
            "use_tma_c": True,
            "use_stream_k": False,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_native_output)
    _assert_results(case, (17, 257))


@pytest.mark.parametrize(
    "weight_values",
    (
        dict(b_dtype="uint8", weight_scale_group_size=128, has_bias=True),
        dict(b_dtype="float8e4m3", weight_scale_group_size=128, has_bias=True),
        dict(b_dtype="float8e4m3", has_bias=True),
        dict(b_dtype="uint4", weight_scale_group_size=128, has_zero_point=True),
        dict(b_dtype="uint4", weight_scale_group_size=128, has_zero_point=True, is_fp_zero_point=True),
        dict(b_dtype="uint4", bs_dtype="float32", weight_scale_type="tensor", has_bias=True),
        dict(b_dtype="uint4", weight_scale_group_size=128, weight_scale_2_type="tensor", has_bias=True),
    ),
)
@pytest.mark.parametrize("block_n", (64, 256))
def test_umma_native_output_weight_types(weight_values, block_n, monkeypatch):
    """Weight format does not determine stage readiness or native output support."""
    case = _case("native-output-weight-type", GemmType.DENSE, **weight_values)

    def select_native_output(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (128, block_n, 64),
            "warp_shape": (128, 32, 64),
            "num_stages": 3,
            "num_ctas_per_sm": 1,
            "use_tma_a": True,
            "use_tma_c": True,
            "use_stream_k": False,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_native_output)
    _assert_results(case, (17, 257))


@pytest.mark.parametrize("shape_n,shape_k,num_sms", ((256, 256, 3), (128, 8192, 64)))
@pytest.mark.parametrize("use_tma_c", (False, True))
@pytest.mark.parametrize(
    "weight_values",
    (
        dict(b_dtype="uint4", weight_scale_group_size=128, has_bias=True),
        dict(b_dtype="uint4", weight_scale_type="channel", has_bias=True),
    ),
)
def test_umma_native_output_stream_k_bias(weight_values, use_tma_c, shape_n, shape_k, num_sms, monkeypatch):
    """Only the first K slice contributes bias to native output."""
    case = _case("native-output-stream-k", GemmType.DENSE, **weight_values)

    case = dataclasses.replace(
        case, layer_config=dataclasses.replace(case.layer_config, shape_n=shape_n, shape_k=shape_k)
    )

    def select_stream_k(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (128, 128, 64),
            "warp_shape": (128, 32, 64),
            "num_stages": 2,
            "num_ctas_per_sm": 1,
            "num_sms": num_sms,
            "use_tma": True,
            "use_tma_c": use_tma_c,
            "use_stream_k": True,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_stream_k)
    _assert_results(case, (17, 257))


@pytest.mark.parametrize(
    "gemm_type,block_m,shape_k",
    (
        (GemmType.DENSE, 16, 256),
        (GemmType.INDEXED, 32, 256),
        (GemmType.DENSE, 48, 256),
        (GemmType.DENSE, 96, 256),
        (GemmType.DENSE, 64, 64),
        (GemmType.INDEXED, 64, 128),
        (GemmType.INDEXED, 128, 256),
        (GemmType.GROUPED_CONTIGUOUS, 64, 192),
        (GemmType.GROUPED_MASKED, 128, 448),
        (GemmType.DENSE, 176, 256),
        (GemmType.INDEXED, 176, 256),
        (GemmType.DENSE, 192, 256),
        (GemmType.INDEXED, 192, 256),
    ),
)
@pytest.mark.parametrize("block_n", (128, 256))
@pytest.mark.parametrize("weight_name", ("uint4", "uint4-zp"))
def test_umma_pipeline_stage_reuse(gemm_type, block_m, shape_k, block_n, weight_name, monkeypatch):
    """Retire async reads before reuse across persistent tiles and experts."""
    weights = WEIGHT_CONFIGS[weight_name] | {"weight_scale_group_size": 64}
    case = _case("pipeline-stage-reuse", gemm_type, **weights)
    case = dataclasses.replace(case, layer_config=dataclasses.replace(case.layer_config, shape_k=shape_k))

    def minimum_stages(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (block_m, block_n, 64),
            "warp_shape": (block_m, 32, 64),
            "num_stages": 3,
            "num_sms": 2,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", minimum_stages)
    _assert_results(case, (17, 257))


@pytest.mark.parametrize("num_ctas_per_sm,block_n", ((2, 256), (3, 128)))
def test_umma_multi_cta_sparse_decode(num_ctas_per_sm, block_n, monkeypatch):
    """Multiple CTAs can reuse indexed stages across successive output tiles."""
    case = _case("multi-cta-sparse-decode", GemmType.INDEXED, **WEIGHT_CONFIGS["uint4"])
    case = dataclasses.replace(case, layer_config=dataclasses.replace(case.layer_config, shape_n=1024))

    def select_multiple_ctas(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (8, block_n, 64),
            "warp_shape": (8, 32, 64),
            "num_stages": 2,
            "num_ctas_per_sm": num_ctas_per_sm,
            "num_sms": 2,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_multiple_ctas)
    _assert_results(case, (1, 17))


@pytest.mark.parametrize("weight_name", ("uint4", "uint4-zp"))
@pytest.mark.parametrize(
    "shape_n,shape_k,num_experts,shape_ms",
    ((2048, 512, 32, (1, 17, 2816)), (1024, 2048, 64, (17, 32, 49, 257, 5632))),
)
def test_umma_indexed_multi_expert_tiles(weight_name, shape_n, shape_k, num_experts, shape_ms):
    case = _case("indexed-multi-expert", GemmType.INDEXED, **WEIGHT_CONFIGS[weight_name])
    layer_config = dataclasses.replace(
        case.layer_config, shape_n=shape_n, shape_k=shape_k, num_experts=num_experts
    )
    _assert_results(dataclasses.replace(case, layer_config=layer_config), shape_ms)


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
def test_umma_floating_weight_formats(b_dtype, gemm_type):
    case = _case(b_dtype, gemm_type, b_dtype=b_dtype)
    _assert_results(case, (17, 129))


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("gemm_type", list(GemmType))
def test_umma_integer_weight_widths(bits, gemm_type):
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
        dict(b_dtype="uint3", weight_scale_group_size=64, weight_scale_2_type="channel"),
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
def test_umma_scale_contract(weight_values, gemm_type):
    case = _case("scale", gemm_type, **weight_values)
    _assert_results(case, (17, 257))


def _public_problem(weight_ref, shape_m, gemm_type, block_m):
    device = weight_ref.device
    shape_k = weight_ref.shape[-1]
    if gemm_type == GemmType.DENSE:
        inputs = generate_random_tensor((shape_m, shape_k), torch.bfloat16, device=device)
        return dict(inputs=inputs), slice(None), inputs.float() @ weight_ref.T

    # Leave experts 1 and 3 empty; experts 0 and 2 have tail tiles.
    topk_ids = torch.tensor([0, 2], device=device, dtype=torch.int32)
    topk_ids = topk_ids.expand(shape_m, -1).contiguous()
    expert_max_tokens = shape_m + 3
    _, layout, sorted_ids, expert_ids, padded = generate_moe_tensors(
        topk_ids,
        4,
        gemm_type,
        block_size_config=block_m,
        expert_max_tokens=expert_max_tokens,
    )
    if gemm_type == GemmType.INDEXED:
        inputs = generate_random_tensor((shape_m, shape_k), torch.bfloat16, device=device)
        reference = torch.stack([inputs.float() @ weight_ref[e].T for e in (0, 2)], dim=1).flatten(0, 1)
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

    total_m = shape_m * 2 if gemm_type == GemmType.GROUPED_CONTIGUOUS else 4 * expert_max_tokens
    inputs = generate_random_tensor((total_m, shape_k), torch.bfloat16, device=device)
    output_ids, references = [], []
    for expert in (0, 2):
        offset = (
            int(layout[expert]) if gemm_type == GemmType.GROUPED_CONTIGUOUS else expert * expert_max_tokens
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
        tensors["weight_scale_2"] *= torch.arange(1, num_experts + 1, device=weight.device).reshape_as(
            tensors["weight_scale_2"]
        )
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
def test_umma_public_layer_switches_without_repacking(weight_name, gemm_type):
    """MMA and UMMA consume one transformed layer, including routed calls."""
    layer, weight_ref = _public_layer(weight_name, gemm_type)
    num_experts = layer.num_experts
    packed = {
        name: (value.data_ptr(), value.detach().view(torch.uint8).clone())
        for name, value in layer.named_parameters()
    }
    for shape_m, mma_type in (
        (17, MmaType.MMA),
        (257, MmaType.UMMA),
        (257, MmaType.MMA),
        (17, MmaType.MMA),
    ):
        torch.manual_seed(2026 + shape_m)
        config = dataclasses.replace(layer.humming_config, mma_type=mma_type)
        get_config = Sm100Heuristics.get_umma_config if mma_type == MmaType.UMMA else get_heuristics_config
        tuning = get_config(
            config,
            shape_m=shape_m * (2 if num_experts else 1),
            gemm_type=gemm_type,
        )
        tuning |= {"mma_type": mma_type.value}
        kwargs, output_ids, reference = _public_problem(
            weight_ref, shape_m, gemm_type, tuning["block_shape"][0]
        )
        outputs = layer(
            **kwargs,
            compute_config={"gemm_type": gemm_type.value},
            tuning_config=tuning,
        )
        actual_mma = _selected_backend(layer, gemm_type, kwargs, tuning)
        assert actual_mma == mma_type
        torch.testing.assert_close(outputs[output_ids], reference.to(torch.bfloat16), rtol=0.01, atol=0.05)
    for name, value in layer.named_parameters():
        pointer, original = packed[name]
        assert pointer == value.data_ptr()
        assert torch.equal(value.detach().view(torch.uint8), original)


@pytest.mark.parametrize("block_n,block_k", ((256, 64), (512, 32)))
@pytest.mark.parametrize("smem_reuse_mode", list(SmemReuseMode))
@pytest.mark.parametrize("gemm_type", (GemmType.DENSE, GemmType.INDEXED))
@pytest.mark.parametrize("num_stages", (2, 4))
@pytest.mark.parametrize(
    "weight_values",
    (
        WEIGHT_CONFIGS["nvfp4"],
        dict(b_dtype="uint4", weight_scale_group_size=128, has_bias=True),
        dict(b_dtype="uint4", weight_scale_type="channel", has_bias=True),
    ),
)
def test_umma_smem_reuse_mode(
    block_n, block_k, smem_reuse_mode, gemm_type, num_stages, weight_values, monkeypatch
):
    case = _case("smem-reuse", gemm_type, **weight_values)
    layer_config = dataclasses.replace(case.layer_config, shape_n=1024, shape_k=1024)
    case = dataclasses.replace(case, layer_config=layer_config)

    def select_storage(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (32, block_n, block_k),
            "warp_shape": (32, 32, block_k),
            "num_stages": num_stages,
            "num_ctas_per_sm": 1,
            "num_sms": 2,
            "use_stream_k": False,
            "smem_reuse_mode": smem_reuse_mode,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_storage)
    _assert_results(case, (17, 1025))


@pytest.mark.parametrize("weight_name", ("uint4", "nvfp4"))
@pytest.mark.parametrize("output_dtype", (dtypes.float16, dtypes.bfloat16))
@pytest.mark.parametrize("use_tma_a", (False, True))
@pytest.mark.parametrize("num_stages", (3, 4))
def test_umma_k32(weight_name, output_dtype, use_tma_a, num_stages, monkeypatch):
    def select_k32(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (32, 128, 32),
            "warp_shape": (32, 32, 32),
            "num_stages": num_stages,
            "num_sms": 2,
            "num_ctas_per_sm": 1,
            "use_tma_a": use_tma_a,
            "use_stream_k": False,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_k32)
    case = _case(weight_name, GemmType.DENSE, **WEIGHT_CONFIGS[weight_name])
    layer_config = dataclasses.replace(
        case.layer_config,
        a_dtype=output_dtype,
        c_dtype=output_dtype,
        bs_dtype=output_dtype if weight_name == "uint4" else case.layer_config.bs_dtype,
        shape_k=1024,
    )
    _assert_results(dataclasses.replace(case, layer_config=layer_config), (1, 17, 257))


@pytest.mark.parametrize("gemm_type", (GemmType.DENSE, GemmType.INDEXED))
@pytest.mark.parametrize("use_stream_k", (False, True))
def test_umma_k32_tile_reuse(gemm_type, use_stream_k, monkeypatch):
    def select_k32(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (8, 256, 32),
            "warp_shape": (8, 32, 32),
            "num_stages": 5,
            "num_sms": 2,
            "num_ctas_per_sm": 1,
            "use_stream_k": use_stream_k,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_k32)
    case = _case("uint4", gemm_type, **WEIGHT_CONFIGS["uint4"])
    case = dataclasses.replace(case, layer_config=dataclasses.replace(case.layer_config, shape_k=1024))
    _assert_results(case, (1, 17, 257))


@pytest.mark.parametrize(
    "gemm_type,weight_name,use_tma_a,block_n,num_stages",
    (
        (GemmType.DENSE, "uint4-zp", True, 128, 3),
        (GemmType.DENSE, "fp8", True, 128, 3),
        (GemmType.GROUPED_CONTIGUOUS, "nvfp4", True, 256, 3),
        (GemmType.GROUPED_MASKED, "uint4-zp", True, 256, 4),
        (GemmType.DENSE, "nvfp4", False, 128, 3),
        (GemmType.DENSE, "nvfp4", True, 512, 4),
    ),
)
def test_umma_operand_buffer_selection(gemm_type, weight_name, use_tma_a, block_n, num_stages, monkeypatch):
    """Exercise stage-matched and capacity-limited operands with either loading path."""

    def select_operands(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (32, block_n, 64),
            "warp_shape": (32, 32, 64),
            "num_stages": num_stages,
            "num_sms": 2,
            "num_ctas_per_sm": 1,
            "use_tma_a": use_tma_a,
            "use_stream_k": False,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_operands)
    case = _case("operand-buffers", gemm_type, **WEIGHT_CONFIGS[weight_name])
    layer_config = dataclasses.replace(case.layer_config, shape_n=1024, shape_k=1024)
    _assert_results(dataclasses.replace(case, layer_config=layer_config), (17, 257))
