import dataclasses

import pytest
import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType
from humming.config.config import _cuda_compiler_version
from humming.config.mma import get_default_mma_type
from humming.jit.runtime import KernelRuntime
from humming.kernel.humming import HummingKernel
from humming.layer import HummingLayer
from humming.schema import HummingWeightSchema
from humming.testing import KernelTestCase, KernelTestRunner
from humming.testing.data import generate_moe_tensors, generate_random_tensor
from humming.tune.sm100 import Sm100Heuristics, Sm100MmaHeuristics, Sm100UmmaHeuristics

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
def require_umma_device(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("UMMA requires an SM10x or SM11x GPU")
    if _cuda_compiler_version(KernelRuntime._get_compiler()) < (12, 9):
        pytest.skip("UMMA requires CUDA 12.9 or newer")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    # These regressions require the exact backend and geometry selected below.
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "heuristic")

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
            dequant_threads = 0 if compiled.use_umma_ss else 128 * compiled.umma_num_dequant_warpgroups
            expected_threads = 256 + dequant_threads
            assert compiled.num_threads == expected_threads
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


@pytest.mark.parametrize(
    "gemm_type,block_m,shape_k,block_n,weight_name",
    (
        (GemmType.DENSE, 64, 64, 128, "uint4"),
        (GemmType.GROUPED_CONTIGUOUS, 64, 192, 256, "uint4-zp"),
        (GemmType.INDEXED, 176, 256, 128, "uint4"),
    ),
)
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


@pytest.mark.parametrize(
    "gemm_type,weight_name",
    (
        (GemmType.DENSE, "uint4"),
        (GemmType.INDEXED, "uint4-zp"),
        (GemmType.GROUPED_CONTIGUOUS, "uint4-fp-zp"),
        (GemmType.GROUPED_MASKED, "nvfp4"),
        (GemmType.DENSE, "mxfp4"),
        (GemmType.INDEXED, "fp8"),
    ),
)
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
        config = layer.humming_config
        get_config = Sm100MmaHeuristics.get_config
        if mma_type == MmaType.UMMA:
            get_config = Sm100Heuristics.get_umma_config
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


@pytest.mark.parametrize(
    "gemm_type,use_stream_k",
    (
        (GemmType.DENSE, False),
        (GemmType.INDEXED, True),
    ),
)
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


@pytest.mark.parametrize(
    "shape_k,activation_dtype,weight_dtype",
    (
        (64, "float8e4m3", "float8e4m3"),
        (256, "float8e3m4", "float8e3m4"),
        (64, "float8e4m3", "float4e2m1"),
        (256, "float8e3m4", "float6e2m3"),
    ),
)
def test_umma_fp8_public_dispatch(shape_k, activation_dtype, weight_dtype, monkeypatch):
    # This regression checks production small-M dispatch, not test backend preference.
    monkeypatch.delenv("HUMMING_TEST_TUNING_SOURCE", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
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
    assert get_default_mma_type(layer.humming_config) == MmaType.UMMA
    assert layer.humming_config.use_raw_weight
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
        (512, 64, "uint4", GemmType.DENSE, 4, True, 1),
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
    "weight_dtype,activation_dtype,group_size,scale_dtype,quant_mode",
    (
        ("float4e2m1", "float8e3m4", 32, "float8e8m0", "dynamic_group"),
        ("uint2", "float4e0m3", 16, "float8e4m3", "dynamic_group_token"),
        ("float4e2m1", "float8e4m3", 32, "float8e8m0", "dynamic_group"),
        ("uint2", "float4e2m1", 32, "float8e8m0", "dynamic_group"),
        ("float4e2m1", "float4e2m1", 16, "float8e4m3", "dynamic_group_token"),
        ("uint2", "float4e2m1", 16, "float8e4m3", "static_tensor_dynamic_group"),
        ("float4e2m1", "float8e4m3", 32, "float8e8m0", "static_tensor_dynamic_group"),
    ),
)
def test_umma_mxf8_mxf4_public_dispatch(activation_dtype, group_size, scale_dtype, quant_mode, weight_dtype):
    schema = HummingWeightSchema(
        b_dtype=weight_dtype, bs_dtype=scale_dtype, weight_scale_group_size=group_size
    )
    weight = generate_random_tensor((256, 256), torch.bfloat16, device="cuda")
    tensors = schema.quant_tensor(weight, schema, torch.bfloat16, allow_negative_scale=False)
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
    assert get_default_mma_type(layer.humming_config) == MmaType.UMMA
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


@pytest.mark.parametrize(
    "dtype,weight_dtype,block_m,gemm_type,shape_k",
    (
        (dtypes.float8e3m4, dtypes.float8e3m4, 64, GemmType.DENSE, 512),
        (dtypes.float4e2m1, dtypes.float4e2m1, 16, GemmType.DENSE, 512),
        (dtypes.float4e0m3, dtypes.float4e0m3, None, GemmType.GROUPED_MASKED, 1024),
    ),
)
def test_umma_ss_small_tile(dtype, weight_dtype, block_m, gemm_type, shape_k, monkeypatch):
    block_k = 128 if dtype.num_bits == 4 else 64

    def select_config(layer_config, shape_m, gemm_type, **kwargs):
        if block_m is None:
            return Sm100UmmaHeuristics.get_config(layer_config, shape_m, gemm_type=gemm_type)
        return dict(
            mma_type="umma",
            block_shape=(block_m, 128, block_k),
            warp_shape=(block_m, 32, block_k),
            num_stages=3,
            num_sms=4,
            use_tma=True,
            use_tma_a=gemm_type != GemmType.INDEXED,
            use_tma_c=gemm_type != GemmType.INDEXED,
            use_stream_k=True,
            smem_reuse_mode="none",
        )

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_config)
    is_fp4 = dtype.num_bits == 4
    config = LayerConfig(
        shape_n=256,
        shape_k=shape_k,
        num_experts=0 if gemm_type == GemmType.DENSE else 4,
        a_dtype=dtype,
        b_dtype=weight_dtype,
        c_dtype=dtypes.bfloat16,
        as_dtype=dtypes.float8e4m3 if is_fp4 else None,
        bs_dtype=dtypes.float8e4m3 if is_fp4 else dtypes.bfloat16,
        input_quant_mode="dynamic_group_token" if is_fp4 else None,
        input_scale_group_size=16 if is_fp4 else 0,
        weight_scale_group_size=16 if is_fp4 else 0,
        weight_scale_type="group" if is_fp4 else "channel",
    )
    case = KernelTestCase(
        name="ss-small-tile",
        layer_config=config,
        compute_config=ComputeConfig(
            gemm_type=gemm_type,
            use_m_major_input_scale=block_m is None and gemm_type != GemmType.INDEXED,
        ),
        top_k=2,
        seed=2026,
    )
    _assert_results(case, (1, 137, 833) if block_m is None else (1, 137))


@pytest.mark.parametrize(
    "gemm_type,block_m,block_n,block_k,use_tma_c,reuse_mode,cta_group_size,use_tma,use_stream_k",
    (
        (GemmType.DENSE, 40, 128, 64, False, "all_stages", 1, True, True),
        (GemmType.DENSE, 48, 128, 64, True, "last_stage", 2, True, True),
        (GemmType.GROUPED_MASKED, 40, 128, 64, True, "all_stages", 1, True, True),
    ),
)
def test_umma_chunked_output_layout(
    gemm_type,
    block_m,
    block_n,
    block_k,
    use_tma_c,
    reuse_mode,
    cta_group_size,
    use_tma,
    use_stream_k,
    monkeypatch,
):

    def select_output(layer_config, shape_m, gemm_type, **kwargs):
        return {
            "mma_type": "umma",
            "block_shape": (block_m, block_n, block_k),
            "warp_shape": (block_m, 32, block_k),
            "num_stages": 3,
            "num_sms": 6,
            "num_ctas_per_sm": 1,
            "use_tma": use_tma or use_tma_c,
            "use_tma_b": use_tma,
            "use_tma_bs": use_tma,
            "use_tma_bzp": use_tma,
            "use_tma_a": use_tma and gemm_type != GemmType.INDEXED,
            "use_tma_c": use_tma_c,
            "use_stream_k": use_stream_k,
            "smem_reuse_mode": reuse_mode,
            "umma_cta_group_size": cta_group_size,
            "output_chunk_rows": 32,
        }

    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_output)
    case = _case(
        "chunked-layout",
        gemm_type,
        b_dtype="uint4",
        weight_scale_group_size=0,
        has_zero_point=True,
        has_bias=True,
    )
    layer_config = dataclasses.replace(case.layer_config, shape_n=1024, shape_k=1024)
    _assert_results(dataclasses.replace(case, layer_config=layer_config), (17, 13 * block_m + 1))


@pytest.mark.parametrize(
    "a_dtype,b_dtype,microscale,gemm_type,block_m,block_n,block_k,stream_k",
    (
        (dtypes.float8e4m3, dtypes.float4e2m1, True, GemmType.DENSE, 160, 128, 256, False),
        (dtypes.float8e4m3, dtypes.float4e2m1, True, GemmType.DENSE, 32, 512, 128, True),
        (dtypes.float8e4m3, dtypes.float4e2m1, True, GemmType.GROUPED_MASKED, 48, 128, 128, True),
    ),
)
def test_umma_cooperative_fp8(
    a_dtype,
    b_dtype,
    microscale,
    gemm_type,
    block_m,
    block_n,
    block_k,
    stream_k,
    monkeypatch,
):

    def select_config(layer_config, shape_m, gemm_type, **kwargs):
        return {
            "mma_type": "umma",
            "block_shape": (block_m, block_n, block_k),
            "warp_shape": (block_m, 32, block_k),
            "num_stages": 2 if block_n == 512 else 3,
            "num_ctas_per_sm": 1,
            "num_sms": 4,
            "use_warp_spec": True,
            "use_tma": True,
            "use_tma_a": gemm_type != GemmType.INDEXED,
            "use_tma_c": gemm_type != GemmType.INDEXED,
            "use_tma_b": True,
            "use_tma_as": microscale and gemm_type != GemmType.INDEXED,
            "use_stream_k": stream_k,
            "smem_reuse_mode": "none",
            "umma_cta_group_size": 2,
            "output_chunk_rows": 32,
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


@pytest.mark.parametrize(
    "a_dtype,b_dtype,group_size,scale_dtype,quant_mode,cta_group_size,gemm_type,use_tma,block_shape",
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
            (48, 256, 256),
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
            (128, 128, 128),
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
            (240, 128, 256),
        ),
    ),
)
def test_umma_fp4_activation(
    a_dtype,
    b_dtype,
    group_size,
    scale_dtype,
    quant_mode,
    cta_group_size,
    gemm_type,
    use_tma,
    block_shape,
    monkeypatch,
):

    def select_config(layer_config, shape_m, gemm_type, **kwargs):
        return dict(
            mma_type="umma",
            block_shape=block_shape,
            warp_shape=(block_shape[0], 32, block_shape[2]),
            num_stages=3,
            num_ctas_per_sm=1,
            num_sms=4,
            use_warp_spec=True,
            use_tma=True,
            use_tma_a=use_tma,
            use_tma_c=use_tma,
            use_tma_b=True,
            use_tma_as=use_tma,
            use_stream_k=True,
            smem_reuse_mode="none",
            umma_cta_group_size=cta_group_size,
            output_chunk_rows=32,
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
    )
    case = KernelTestCase(
        name="fp4-activation",
        layer_config=config,
        compute_config=ComputeConfig(gemm_type=gemm_type, use_m_major_input_scale=use_tma),
        top_k=2,
        seed=2026,
    )
    _assert_results(case, (17, 833))
