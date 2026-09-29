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
            assert compiled.num_threads == 256 + 128 * compiled.umma_num_dequant_warpgroups
            assert compiled.num_math_threads == 128
            assert compiled.num_load_threads in (64, 96)
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
            "use_tma": True,
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
            "use_tma": True,
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
            "use_tma": True,
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
            "use_tma": True,
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
@pytest.mark.parametrize("compiler", ("nvcc", "nvrtc"))
def test_umma_operand_buffer_selection(
    gemm_type, weight_name, use_tma_a, block_n, num_stages, compiler, monkeypatch
):
    """Exercise stage-matched and capacity-limited operands with either loading path."""
    monkeypatch.setenv("HUMMING_COMPILER", compiler)

    # Compiler versions can select different native dequantization and packing.
    monkeypatch.setattr(KernelRuntime, "_instances", {})
    monkeypatch.setattr(HummingKernel, "_str2kernel_cache", {})

    def select_operands(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (32, block_n, 64),
            "warp_shape": (32, 32, 64),
            "num_stages": num_stages,
            "num_sms": 2,
            "num_ctas_per_sm": 1,
            "use_tma": True,
            "use_tma_a": use_tma_a,
            "use_stream_k": False,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_operands)
    case = _case("operand-buffers", gemm_type, **WEIGHT_CONFIGS[weight_name])
    layer_config = dataclasses.replace(case.layer_config, shape_n=1024, shape_k=1024)
    _assert_results(dataclasses.replace(case, layer_config=layer_config), (17, 257))


@pytest.mark.parametrize(
    "cta_group_size,block_m,block_k,num_stages,weight_name,output_dtype",
    (
        (1, 96, 64, 4, "uint4", dtypes.bfloat16),
        (1, 256, 64, 4, "nvfp4", dtypes.bfloat16),
        (2, 96, 32, 3, "uint4", dtypes.bfloat16),
        (2, 256, 64, 6, "uint4", dtypes.bfloat16),
        (2, 256, 64, 9, "nvfp4", dtypes.bfloat16),
        (2, 192, 128, 4, "uint4-zp", dtypes.bfloat16),
        (2, 96, 64, 5, "uint4-fp-zp", dtypes.float16),
        (2, 256, 64, 6, "nvfp4", dtypes.float16),
    ),
)
@pytest.mark.parametrize("compiler", ("nvcc", "nvrtc"))
@pytest.mark.parametrize("use_stream_k", (False, True))
def test_umma_chunked_output(
    cta_group_size,
    block_m,
    block_k,
    num_stages,
    weight_name,
    output_dtype,
    compiler,
    use_stream_k,
    monkeypatch,
):
    """Reuse both output buffers across tiles, including odd chunk counts and M tails."""
    monkeypatch.setenv("HUMMING_COMPILER", compiler)
    # Compiler versions can select different native dequantization and packing.
    monkeypatch.setattr(KernelRuntime, "_instances", {})
    monkeypatch.setattr(HummingKernel, "_str2kernel_cache", {})

    def select_output(layer_config, shape_m, gemm_type, **kwargs):
        return {
            "mma_type": "umma",
            "block_shape": (block_m, 128, block_k),
            "warp_shape": (block_m, 32, block_k),
            "num_stages": num_stages,
            "num_sms": 6,
            "num_ctas_per_sm": 1,
            "use_tma": True,
            "use_stream_k": use_stream_k,
            "smem_reuse_mode": "none",
            "umma_cta_group_size": cta_group_size,
            "umma_output_chunk_rows": 32,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_output)
    case = _case("chunked-output", GemmType.DENSE, **WEIGHT_CONFIGS[weight_name])
    bs_dtype = output_dtype if weight_name.startswith("uint4") else case.layer_config.bs_dtype
    layer_config = dataclasses.replace(
        case.layer_config,
        shape_n=512,
        shape_k=1024,
        a_dtype=output_dtype,
        c_dtype=output_dtype,
        bs_dtype=bs_dtype,
    )
    _assert_results(dataclasses.replace(case, layer_config=layer_config), (17, 13 * block_m + 1))


@pytest.mark.parametrize(
    "weight_values,use_tma_channel",
    (
        (dict(b_dtype="uint4", weight_scale_group_size=128, has_bias=True), True),
        (dict(b_dtype="uint4", weight_scale_group_size=0, has_bias=True), True),
        (dict(b_dtype="uint4", weight_scale_group_size=0, has_zero_point=True), True),
        (
            dict(
                b_dtype="uint4",
                weight_scale_group_size=0,
                has_zero_point=True,
                is_fp_zero_point=True,
                has_bias=True,
            ),
            False,
        ),
        (dict(b_dtype="uint4", bs_dtype="float8e4m3", weight_scale_group_size=0, has_bias=True), False),
        (dict(b_dtype="uint4", bs_dtype="float8e8m0", weight_scale_group_size=0, has_bias=True), True),
        (
            dict(
                b_dtype="float4e2m1",
                bs_dtype="float8e4m3",
                weight_scale_group_size=16,
                weight_scale_2_type="channel",
                has_bias=True,
            ),
            False,
        ),
    ),
)
@pytest.mark.parametrize("output_dtype", (dtypes.bfloat16, dtypes.float16))
@pytest.mark.parametrize("use_stream_k", (False, True))
def test_umma_cooperative_channel_parameters(
    weight_values, use_tma_channel, output_dtype, use_stream_k, monkeypatch
):
    """Channel buffers may be reused only after output and dequant consumers read them."""

    def select_output(layer_config, shape_m, gemm_type, **kwargs):
        return {
            "mma_type": "umma",
            "block_shape": (96, 128, 64),
            "warp_shape": (96, 32, 64),
            "num_stages": 6,
            "num_sms": 6,
            "num_ctas_per_sm": 1,
            "use_tma": True,
            "use_tma_bs": layer_config.is_group_weight_scale or use_tma_channel,
            "use_tma_bs2": use_tma_channel,
            "use_tma_bias": use_tma_channel,
            "use_tma_bzp": use_tma_channel,
            "use_stream_k": use_stream_k,
            "smem_reuse_mode": "none",
            "umma_cta_group_size": 2,
            "umma_output_chunk_rows": 32,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_output)
    case = _case("cooperative-channel", GemmType.DENSE, **weight_values)
    bs_dtype = weight_values.get("bs_dtype", output_dtype)
    layer_config = dataclasses.replace(
        case.layer_config,
        shape_n=512,
        shape_k=1024,
        a_dtype=output_dtype,
        c_dtype=output_dtype,
        bs_dtype=bs_dtype,
    )
    _assert_results(dataclasses.replace(case, layer_config=layer_config), (17, 1249))


@pytest.mark.parametrize(
    "gemm_type,block_m,block_n,block_k,use_tma_c,reuse_mode,cta_group_size",
    (
        (GemmType.DENSE, 8, 64, 64, True, "none", 1),
        (GemmType.DENSE, 48, 64, 64, True, "none", 2),
        (GemmType.DENSE, 16, 512, 32, False, "none", 2),
        (GemmType.DENSE, 24, 256, 64, True, "none", 1),
        (GemmType.DENSE, 56, 512, 64, True, "none", 1),
        (GemmType.DENSE, 40, 128, 64, False, "all_stages", 1),
        (GemmType.DENSE, 48, 128, 64, True, "last_stage", 2),
        (GemmType.DENSE, 64, 256, 64, False, "all_stages", 2),
        (GemmType.INDEXED, 8, 64, 64, False, "none", 1),
        (GemmType.INDEXED, 48, 128, 64, False, "none", 2),
        (GemmType.INDEXED, 40, 256, 64, False, "all_stages", 1),
        (GemmType.GROUPED_CONTIGUOUS, 24, 128, 64, True, "none", 1),
        (GemmType.GROUPED_CONTIGUOUS, 40, 256, 64, False, "last_stage", 1),
        (GemmType.GROUPED_MASKED, 40, 64, 64, True, "all_stages", 1),
        (GemmType.GROUPED_CONTIGUOUS, 48, 128, 64, True, "none", 2),
    ),
)
@pytest.mark.parametrize("compiler", ("nvcc", "nvrtc"))
@pytest.mark.parametrize("use_stream_k", (False, True))
def test_umma_chunked_output_layout(
    gemm_type,
    block_m,
    block_n,
    block_k,
    use_tma_c,
    reuse_mode,
    cta_group_size,
    use_stream_k,
    compiler,
    monkeypatch,
):
    monkeypatch.setenv("HUMMING_COMPILER", compiler)
    monkeypatch.setattr(KernelRuntime, "_instances", {})
    monkeypatch.setattr(HummingKernel, "_str2kernel_cache", {})

    def select_output(layer_config, shape_m, gemm_type, **kwargs):
        return {
            "mma_type": "umma",
            "block_shape": (block_m, block_n, block_k),
            "warp_shape": (block_m, 32, block_k),
            "num_stages": 3,
            "num_sms": 6,
            "num_ctas_per_sm": 1,
            "use_tma": True,
            "use_tma_a": gemm_type != GemmType.INDEXED,
            "use_tma_c": use_tma_c,
            "use_stream_k": use_stream_k,
            "smem_reuse_mode": reuse_mode,
            "umma_cta_group_size": cta_group_size,
            "umma_output_chunk_rows": 32,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_output)
    case = _case("chunked-layout", gemm_type, b_dtype="uint4", weight_scale_group_size=0, has_bias=True)
    layer_config = dataclasses.replace(case.layer_config, shape_n=1024, shape_k=1024)
    _assert_results(dataclasses.replace(case, layer_config=layer_config), (17, 13 * block_m + 1))


@pytest.mark.parametrize("fp8_dtype", (dtypes.float8e4m3, dtypes.float8e5m2))
@pytest.mark.parametrize(
    "input_quant_mode,weight_scale_type", (("static_tensor", "tensor"), ("dynamic_token", "channel"))
)
@pytest.mark.parametrize("output_dtype", (dtypes.bfloat16, dtypes.float16))
@pytest.mark.parametrize(
    "weight_dtype",
    (None, dtypes.float4e2m1, dtypes.float6e2m3, dtypes.float6e3m2, dtypes.float8e5m2),
)
def test_umma_fp8(fp8_dtype, input_quant_mode, weight_scale_type, output_dtype, weight_dtype):
    weight_dtype = fp8_dtype if weight_dtype is None else weight_dtype
    case = _case("fp8", GemmType.DENSE, b_dtype=weight_dtype, weight_scale_type=weight_scale_type)
    case = dataclasses.replace(
        case,
        layer_config=dataclasses.replace(
            case.layer_config,
            a_dtype=fp8_dtype,
            input_quant_mode=input_quant_mode,
            c_dtype=output_dtype,
            bs_dtype=output_dtype,
            shape_k=2048,
        ),
    )
    _assert_results(case, (16, 64, 257))


@pytest.mark.parametrize(
    "gemm_type,output_dtype,block_n,block_k,use_tma,has_bias",
    (
        (GemmType.DENSE, dtypes.float16, 128, 64, False, True),
        (GemmType.DENSE, dtypes.bfloat16, 256, 128, True, True),
        (GemmType.INDEXED, dtypes.bfloat16, 256, 64, False, True),
        (GemmType.GROUPED_CONTIGUOUS, dtypes.float16, 128, 128, True, False),
        (GemmType.GROUPED_MASKED, dtypes.bfloat16, 64, 64, True, False),
    ),
)
@pytest.mark.parametrize("output_chunk_rows", (0, 32))
def test_umma_fp8_loading_and_output(
    gemm_type, output_dtype, block_n, block_k, use_tma, has_bias, output_chunk_rows, monkeypatch
):
    def select_config(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (24, block_n, block_k),
            "warp_shape": (24, 32, block_k),
            "umma_output_chunk_rows": output_chunk_rows,
            "num_stages": 4,
            "num_ctas_per_sm": 1,
            "num_sms": 2,
            "use_tma": use_tma,
            "use_tma_a": use_tma,
            "use_tma_c": use_tma,
            "use_stream_k": False,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_config)
    case = _case("fp8-output", gemm_type, b_dtype=dtypes.float8e4m3, weight_scale_type="channel")
    layer_config = dataclasses.replace(
        case.layer_config,
        a_dtype=dtypes.float8e4m3,
        c_dtype=output_dtype,
        bs_dtype=output_dtype,
        input_quant_mode="dynamic_token",
        has_bias=has_bias,
    )
    _assert_results(dataclasses.replace(case, layer_config=layer_config), (17, 257))


def test_umma_fp8_persistent_weight_reuse(monkeypatch):
    def select_config(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (256, 128, 128),
            "warp_shape": (256, 32, 128),
            "num_stages": 3,
            "num_ctas_per_sm": 1,
            "num_sms": 2,
            "use_stream_k": False,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_config)
    case = _case("fp8-persistent", GemmType.DENSE, b_dtype=dtypes.float8e4m3, weight_scale_type="tensor")
    layer_config = dataclasses.replace(
        case.layer_config,
        a_dtype=dtypes.float8e4m3,
        shape_k=8192,
        input_quant_mode="static_tensor",
    )
    _assert_results(dataclasses.replace(case, layer_config=layer_config), (257, 4096))


@pytest.mark.parametrize("shape_k", (64, 256))
@pytest.mark.parametrize("activation_dtype", ("float8e4m3", "float8e3m4"))
@pytest.mark.parametrize("weight_dtype", ("float8e4m3", "float8e3m4", "float4e2m1", "float6e2m3"))
def test_umma_fp8_public_dispatch(shape_k, activation_dtype, weight_dtype):
    schema = HummingWeightSchema(b_dtype=weight_dtype, weight_scale_type="tensor")
    weight = generate_random_tensor((256, shape_k), torch.bfloat16, device="cuda")
    tensors = schema.quant_tensor(weight, schema, torch.bfloat16)
    weight_ref = schema.dequant_tensors(tensors).float()
    tensors["input_scale"] = torch.tensor([0.25], device="cuda")
    layer = HummingLayer(
        shape_n=256,
        shape_k=shape_k,
        weight_config=schema,
        input_config={"dtype": activation_dtype, "quant_mode": "static_tensor"},
        torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_state_dict(tensors, strict=False)
    layer.transform()
    assert layer.humming_config.mma_type == MmaType.UMMA
    runner = KernelTestRunner(
        KernelTestCase(
            name="fp8-public",
            layer_config=layer.humming_config,
            compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        )
    )
    for shape_m in (17, 257):
        inputs = torch.randn((shape_m, shape_k), device="cuda", dtype=torch.bfloat16)
        inputs_ref, _, _, _ = runner.prepare_inputs(inputs, tensors["input_scale"])
        outputs = layer(inputs)
        expected = inputs_ref @ weight_ref.T
        torch.testing.assert_close(outputs, expected.to(torch.bfloat16), rtol=0.01, atol=0.05)
        backend = _selected_backend(layer, GemmType.DENSE, {"inputs": inputs})
        expect_umma = shape_m == 257 or weight_dtype != "float8e4m3" or activation_dtype == "float8e3m4"
        assert backend == (MmaType.UMMA if expect_umma else MmaType.MMA)


@pytest.mark.parametrize(
    "block_n,block_k,weight_name,gemm_type,num_stages,use_tma,num_ctas",
    (
        (128, 128, "fp8", GemmType.DENSE, 3, True, 1),
        (128, 64, "fp8", GemmType.DENSE, 4, False, 1),
        (256, 64, "nvfp4", GemmType.INDEXED, 3, False, 1),
        (512, 64, "uint4", GemmType.DENSE, 4, True, 1),
        (64, 32, "nvfp4", GemmType.DENSE, 4, True, 2),
    ),
)
def test_umma_cooperative_dequant(
    block_n, block_k, weight_name, gemm_type, num_stages, use_tma, num_ctas, monkeypatch
):
    def select_config(layer_config, shape_m, gemm_type, **kwargs):
        return Sm100Heuristics.get_umma_config(layer_config, shape_m, gemm_type) | {
            "block_shape": (32, block_n, block_k),
            "warp_shape": (32, 32, block_k),
            "num_stages": num_stages,
            "num_ctas_per_sm": num_ctas,
            "umma_num_dequant_warpgroups": 2,
            "num_sms": 2,
            "use_stream_k": True,
            "use_tma": use_tma,
            "use_tma_a": use_tma,
            "use_tma_c": use_tma,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_config)
    case = _case("cooperative-dequant", gemm_type, **WEIGHT_CONFIGS[weight_name])
    layer_config = dataclasses.replace(case.layer_config, shape_n=1024, shape_k=1024)
    if weight_name == "fp8":
        layer_config = dataclasses.replace(
            layer_config, a_dtype=dtypes.float8e4m3, input_quant_mode="dynamic_token"
        )
    _assert_results(dataclasses.replace(case, layer_config=layer_config), (17, 257))


@pytest.mark.parametrize(
    "block_m,block_n,block_k,use_tma,groups,stream_k,a_dtype,b_dtype,gemm_type",
    (
        (64, 128, 128, False, 1, False, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.DENSE),
        (64, 128, 64, False, 1, True, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.DENSE),
        (64, 128, 128, True, 1, False, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.DENSE),
        (32, 64, 64, True, 1, True, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.DENSE),
        (32, 512, 128, True, 1, False, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.DENSE),
        (8, 64, 256, False, 1, False, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.DENSE),
        (32, 512, 128, False, 2, False, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.DENSE),
        (96, 128, 256, True, 1, False, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.DENSE),
        (160, 128, 256, True, 1, False, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.DENSE),
        (24, 256, 128, True, 2, True, dtypes.float8e5m2, dtypes.float4e2m1, GemmType.DENSE),
        (256, 128, 128, True, 2, False, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.DENSE),
        (8, 64, 64, False, 1, True, dtypes.float8e4m3, dtypes.float6e3m2, GemmType.DENSE),
        (32, 128, 128, True, 1, False, dtypes.float8e4m3, dtypes.float8e5m2, GemmType.DENSE),
        (24, 256, 128, True, 1, True, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.GROUPED_CONTIGUOUS),
        (24, 128, 64, False, 1, True, dtypes.float8e4m3, dtypes.float4e2m1, GemmType.INDEXED),
    ),
)
def test_umma_mxf8_mxf4(
    block_m, block_n, block_k, use_tma, groups, stream_k, a_dtype, b_dtype, gemm_type, monkeypatch
):
    def select_config(layer_config, shape_m, gemm_type, **kwargs):
        return {
            "mma_type": "umma",
            "block_shape": (block_m, block_n, block_k),
            "warp_shape": (block_m, 32, block_k),
            "num_stages": 3,
            "num_ctas_per_sm": 1,
            "num_sms": 3,
            "use_warp_spec": True,
            "smem_reuse_mode": "none",
            "use_tma": use_tma,
            "use_tma_a": use_tma,
            "use_tma_c": use_tma,
            "use_tma_as": use_tma,
            "use_stream_k": stream_k,
            "umma_num_dequant_warpgroups": groups,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_config)
    case = KernelTestCase(
        name="umma-mxf8-mxf4",
        layer_config=LayerConfig(
            shape_n=max(256, block_n),
            shape_k=1024,
            num_experts=0 if gemm_type == GemmType.DENSE else 4,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=dtypes.bfloat16,
            as_dtype=dtypes.float8e8m0,
            bs_dtype=dtypes.float8e8m0,
            input_scale_group_size=32,
            weight_scale_group_size=32,
            mma_type=MmaType.UMMA,
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type, use_m_major_input_scale=use_tma),
        seed=2026,
    )
    _assert_results(case, (17, 257))


@pytest.mark.parametrize(
    "activation_dtype,group_size,scale_dtype,quant_mode",
    (
        ("float8e3m4", 32, "float8e8m0", "dynamic_group"),
        ("float4e0m3", 16, "float8e4m3", "dynamic_group_token"),
        ("float8e4m3", 32, "float8e8m0", "dynamic_group"),
        ("float4e2m1", 32, "float8e8m0", "dynamic_group"),
        ("float4e2m1", 16, "float8e4m3", "dynamic_group_token"),
        ("float4e2m1", 16, "float8e4m3", "static_tensor_dynamic_group"),
        ("float8e4m3", 32, "float8e8m0", "static_tensor_dynamic_group"),
    ),
)
def test_umma_mxf8_mxf4_public_dispatch(activation_dtype, group_size, scale_dtype, quant_mode):
    schema = HummingWeightSchema(
        b_dtype="float4e2m1", bs_dtype=scale_dtype, weight_scale_group_size=group_size
    )
    weight = generate_random_tensor((256, 256), torch.bfloat16, device="cuda")
    tensors = schema.quant_tensor(weight, schema, torch.bfloat16)
    weight_ref = schema.dequant_tensors(tensors).float()
    static_scale = None
    if quant_mode == "static_tensor_dynamic_group":
        static_scale = torch.tensor([0.25], device="cuda")
        tensors["input_scale_2"] = static_scale
    layer = HummingLayer(
        shape_n=256,
        shape_k=256,
        weight_config=schema,
        input_config={
            "dtype": activation_dtype,
            "group_size": group_size,
            "scale_dtype": scale_dtype,
            "quant_mode": quant_mode,
        },
        torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_state_dict(tensors, strict=False)
    layer.transform()
    assert layer.humming_config.use_block_scaled_mma
    assert layer.humming_config.mma_type == MmaType.UMMA
    runner = KernelTestRunner(
        KernelTestCase(
            name="mx-public",
            layer_config=layer.humming_config,
            compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        )
    )
    for shape_m in (1, 17, 257):
        inputs = torch.randn((shape_m, 256), device="cuda", dtype=torch.bfloat16)
        inputs_ref, _, _, _ = runner.prepare_inputs(inputs, static_scale)
        outputs = layer(inputs)
        expected = inputs_ref @ weight_ref.T
        torch.testing.assert_close(outputs, expected.to(torch.bfloat16), rtol=0.01, atol=0.05)


@pytest.mark.parametrize("compiler", ("nvcc", "nvrtc"))
@pytest.mark.parametrize(
    "a_dtype,b_dtype,microscale,gemm_type,block_m,block_n,block_k,stream_k",
    (
        (dtypes.float8e3m4, dtypes.float8e3m4, False, GemmType.DENSE, 64, 128, 128, True),
        (dtypes.float8e3m4, dtypes.float8e3m4, True, GemmType.DENSE, 64, 128, 128, False),
        (dtypes.float8e3m4, dtypes.float4e2m1, True, GemmType.INDEXED, 32, 128, 128, True),
        (dtypes.float8e4m3, dtypes.float8e3m4, True, GemmType.GROUPED_CONTIGUOUS, 64, 128, 128, True),
        (dtypes.float8e3m4, dtypes.float8e5m2, False, GemmType.DENSE, 64, 128, 64, False),
        (dtypes.float8e4m3, dtypes.float8e4m3, False, GemmType.DENSE, 64, 128, 128, False),
        (dtypes.float8e5m2, dtypes.float4e2m1, False, GemmType.DENSE, 96, 128, 64, True),
        (dtypes.float8e4m3, dtypes.float6e3m2, False, GemmType.DENSE, 64, 256, 128, True),
        (dtypes.float8e4m3, dtypes.float8e5m2, False, GemmType.GROUPED_CONTIGUOUS, 64, 128, 128, False),
        (dtypes.float8e4m3, dtypes.float4e2m1, True, GemmType.DENSE, 64, 128, 128, False),
        (dtypes.float8e5m2, dtypes.float8e4m3, True, GemmType.DENSE, 96, 128, 64, True),
        (dtypes.float8e4m3, dtypes.float6e2m3, True, GemmType.DENSE, 64, 256, 128, True),
        (dtypes.float8e4m3, dtypes.float4e2m1, True, GemmType.DENSE, 160, 128, 256, False),
        (dtypes.float8e4m3, dtypes.float4e2m1, True, GemmType.GROUPED_CONTIGUOUS, 64, 128, 128, True),
        (dtypes.float8e4m3, dtypes.float4e2m1, True, GemmType.DENSE, 32, 512, 128, True),
        (dtypes.float8e4m3, dtypes.float8e4m3, False, GemmType.INDEXED, 32, 128, 64, True),
        (dtypes.float8e4m3, dtypes.float4e2m1, True, GemmType.INDEXED, 32, 128, 128, False),
        (dtypes.float8e4m3, dtypes.float4e2m1, True, GemmType.GROUPED_MASKED, 48, 128, 128, True),
    ),
)
def test_umma_cooperative_fp8(
    a_dtype, b_dtype, microscale, gemm_type, block_m, block_n, block_k, stream_k, compiler, monkeypatch
):
    monkeypatch.setenv("HUMMING_COMPILER", compiler)
    monkeypatch.setattr(KernelRuntime, "_instances", {})
    monkeypatch.setattr(HummingKernel, "_str2kernel_cache", {})

    def select_config(layer_config, shape_m, gemm_type, **kwargs):
        return {
            "mma_type": "umma",
            "block_shape": (block_m, block_n, block_k),
            "warp_shape": (block_m, 32, block_k),
            "num_stages": 3,
            "num_ctas_per_sm": 1,
            "num_sms": 4,
            "use_warp_spec": True,
            "use_tma": gemm_type != GemmType.INDEXED,
            "use_tma_as": microscale and gemm_type != GemmType.INDEXED,
            "use_stream_k": stream_k,
            "smem_reuse_mode": "none",
            "umma_cta_group_size": 2,
            "umma_output_chunk_rows": 32,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_config)
    output_dtype = dtypes.float16 if a_dtype == dtypes.float8e5m2 else dtypes.bfloat16
    layer_config = LayerConfig(
        shape_n=2 * block_n,
        shape_k=1024,
        num_experts=0 if gemm_type == GemmType.DENSE else 4,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=output_dtype,
        bs_dtype=dtypes.float8e8m0 if microscale else output_dtype,
        as_dtype=dtypes.float8e8m0 if microscale else None,
        input_scale_group_size=32 if microscale else 0,
        weight_scale_group_size=32 if microscale else 0,
        input_quant_mode="dynamic_group" if microscale else "dynamic_token",
        weight_scale_type="group" if microscale else "channel",
        has_bias=not microscale,
        mma_type=MmaType.UMMA,
    )
    case = KernelTestCase(
        name="cooperative-fp8",
        layer_config=layer_config,
        compute_config=ComputeConfig(
            gemm_type=gemm_type,
            use_m_major_input_scale=microscale and gemm_type != GemmType.INDEXED,
        ),
        top_k=2,
        seed=2026,
    )
    _assert_results(case, (17, 13 * block_m + 1))


@pytest.mark.parametrize("compiler", ("nvcc", "nvrtc"))
@pytest.mark.parametrize(
    "a_dtype,b_dtype,group_size,scale_dtype,quant_mode,cta_group_size,gemm_type,use_tma",
    (
        (
            dtypes.float4e2m1,
            dtypes.float4e2m1,
            32,
            dtypes.float8e8m0,
            "dynamic_group",
            1,
            GemmType.DENSE,
            True,
        ),
        (
            dtypes.float4e2m1,
            dtypes.float4e2m1,
            32,
            dtypes.float8e8m0,
            "static_tensor_dynamic_group",
            2,
            GemmType.DENSE,
            True,
        ),
        (
            dtypes.float4e2m1,
            dtypes.float4e2m1,
            16,
            dtypes.float8e4m3,
            "dynamic_group_token",
            1,
            GemmType.DENSE,
            True,
        ),
        (
            dtypes.float4e2m1,
            dtypes.float4e2m1,
            16,
            dtypes.float8e4m3,
            "static_tensor_dynamic_group",
            2,
            GemmType.DENSE,
            False,
        ),
        (
            dtypes.float4e2m1,
            dtypes.float4e2m1,
            16,
            dtypes.float8e8m0,
            "dynamic_group",
            1,
            GemmType.GROUPED_CONTIGUOUS,
            True,
        ),
        (
            dtypes.float4e2m1,
            dtypes.float4e2m1,
            16,
            dtypes.float8e4m3,
            "dynamic_group_token",
            2,
            GemmType.INDEXED,
            False,
        ),
        (
            dtypes.float4e0m3,
            dtypes.float4e0m3,
            16,
            dtypes.float8e4m3,
            "dynamic_group_token",
            1,
            GemmType.DENSE,
            True,
        ),
        (
            dtypes.float4e0m3,
            dtypes.float4e0m3,
            16,
            dtypes.float8e8m0,
            "static_tensor_dynamic_group",
            2,
            GemmType.DENSE,
            True,
        ),
        (
            dtypes.float4e0m3,
            dtypes.float4e2m1,
            16,
            dtypes.float8e4m3,
            "dynamic_group_token",
            2,
            GemmType.INDEXED,
            False,
        ),
        (
            dtypes.float4e2m1,
            dtypes.float4e0m3,
            16,
            dtypes.float8e8m0,
            "dynamic_group",
            1,
            GemmType.GROUPED_CONTIGUOUS,
            True,
        ),
    ),
)
@pytest.mark.parametrize("block_shape", ((64, 128, 128), (48, 256, 256)))
def test_umma_fp4_activation(
    a_dtype,
    b_dtype,
    group_size,
    scale_dtype,
    quant_mode,
    cta_group_size,
    gemm_type,
    use_tma,
    compiler,
    block_shape,
    monkeypatch,
):
    monkeypatch.setenv("HUMMING_COMPILER", compiler)
    monkeypatch.setattr(KernelRuntime, "_instances", {})
    monkeypatch.setattr(HummingKernel, "_str2kernel_cache", {})

    def select_config(layer_config, shape_m, gemm_type, **kwargs):
        return dict(
            mma_type="umma",
            block_shape=block_shape,
            warp_shape=(block_shape[0], 32, block_shape[2]),
            num_stages=3,
            num_ctas_per_sm=1,
            num_sms=4,
            use_warp_spec=True,
            use_tma=use_tma,
            use_tma_as=use_tma,
            use_stream_k=True,
            smem_reuse_mode="none",
            umma_cta_group_size=cta_group_size,
            umma_output_chunk_rows=32,
        )

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_config)
    config = LayerConfig(
        shape_n=2 * block_shape[1],
        shape_k=1024,
        num_experts=0 if gemm_type == GemmType.DENSE else 4,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=dtypes.bfloat16,
        as_dtype=scale_dtype,
        bs_dtype=scale_dtype,
        input_scale_group_size=group_size,
        weight_scale_group_size=group_size,
        input_quant_mode=quant_mode,
        weight_scale_type="group",
        weight_scale_2_type="tensor",
        has_bias=True,
        mma_type=MmaType.UMMA,
    )
    case = KernelTestCase(
        name="fp4-activation",
        layer_config=config,
        compute_config=ComputeConfig(gemm_type=gemm_type, use_m_major_input_scale=use_tma),
        top_k=2,
        seed=2026,
    )
    _assert_results(case, (17, 833))


@pytest.mark.parametrize(
    "gemm_type", (GemmType.DENSE, GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED, GemmType.INDEXED)
)
@pytest.mark.parametrize("cta_group_size", (1, 2))
@pytest.mark.parametrize(
    "quant_mode,has_channel_data", (("static_tensor_dynamic_group", True), ("dynamic_group_token", False))
)
def test_umma_secondary_input_scale(gemm_type, cta_group_size, quant_mode, has_channel_data, monkeypatch):
    has_token_scale = quant_mode == "dynamic_group_token"
    block_k = 128 if has_token_scale else 64
    group_size = 16 if has_token_scale else 32

    def select_config(layer_config, shape_m, gemm_type, **kwargs):
        return dict(
            mma_type="umma",
            block_shape=(48, 128, block_k),
            warp_shape=(48, 32, block_k),
            num_stages=3,
            num_ctas_per_sm=1,
            num_sms=4,
            use_warp_spec=True,
            use_tma=gemm_type != GemmType.INDEXED,
            use_stream_k=True,
            smem_reuse_mode="none",
            umma_cta_group_size=cta_group_size,
            umma_output_chunk_rows=32,
        )

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_config)
    config = LayerConfig(
        shape_n=256,
        shape_k=1024,
        num_experts=0 if gemm_type == GemmType.DENSE else 4,
        a_dtype=dtypes.float4e2m1 if has_token_scale else dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.float16,
        as_dtype=dtypes.float8e4m3 if has_token_scale else dtypes.float8e8m0,
        bs_dtype=dtypes.float8e4m3 if has_token_scale else dtypes.float8e8m0,
        input_scale_group_size=group_size,
        weight_scale_group_size=group_size,
        input_quant_mode=quant_mode,
        weight_scale_type="group",
        weight_scale_2_type="channel" if has_channel_data else "tensor",
        has_bias=has_channel_data,
        mma_type=MmaType.UMMA,
    )
    case = KernelTestCase(
        name="secondary-input-scale",
        layer_config=config,
        compute_config=ComputeConfig(
            gemm_type=gemm_type,
            use_m_major_input_scale=gemm_type != GemmType.INDEXED,
        ),
        top_k=2,
        seed=2026,
    )
    _assert_results(case, (17, 625))
