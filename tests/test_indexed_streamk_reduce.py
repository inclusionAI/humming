import json
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from humming import dtypes, ops
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.transform import (
    transform_humming_weight,
)
from humming.tune.sm90_h20 import Sm90H20Heuristics
from humming.utils.test import (
    generate_random_inputs,
    skip_if_unsupported,
)


def test_h20_indexed_mxfp4_enables_fp32_streamk_reduce():
    layer_config = LayerConfig(
        shape_n=1024,
        shape_k=2048,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=0,
        weight_scale_group_size=32,
        weight_scale_type="group",
        weight_scale_2_type="tensor",
        use_fused_e8m0_scale=True,
        num_experts=16,
        mma_type="wgmma",
    )

    config = Sm90H20Heuristics.get_config(
        layer_config,
        shape_m=1024,
        use_f16_accum=False,
        gemm_type=GemmType.INDEXED,
    )

    assert config["use_stream_k"] is True
    assert config["use_fp32_stream_k_reduce"] is True


def test_streamk_workspace_is_scoped_to_cuda_stream():
    skip_if_unsupported(a_dtype=dtypes.bfloat16, mma_type="mma")
    device = torch.device("cuda")
    locks = torch.zeros((8,), dtype=torch.int32, device=device)
    kernel = SimpleNamespace(block_shape=(8, 16, 32))

    ops._streamk_workspaces.clear()
    default_workspace = ops._get_streamk_workspace(locks, locks, kernel)
    other_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(other_stream):
        other_workspace = ops._get_streamk_workspace(locks, locks, kernel)
    default_workspace_again = ops._get_streamk_workspace(locks, locks, kernel)

    assert default_workspace.data_ptr() == default_workspace_again.data_ptr()
    assert default_workspace.data_ptr() != other_workspace.data_ptr()
    ops._streamk_workspaces.clear()


def _make_indexed_metadata(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_topk_ids = topk_ids.reshape(-1)
    invalid_id = flat_topk_ids.numel()
    sorted_ids_by_expert = []
    expert_ids = []

    for expert_id in range(num_experts):
        token_ids = torch.where(flat_topk_ids == expert_id)[0].to(torch.int32)
        num_blocks = math.ceil(token_ids.numel() / block_size)
        padded_size = num_blocks * block_size
        if padded_size != token_ids.numel():
            token_ids = F.pad(
                token_ids,
                (0, padded_size - token_ids.numel()),
                value=invalid_id,
            )
        sorted_ids_by_expert.append(token_ids)
        expert_ids.extend([expert_id] * num_blocks)

    sorted_ids = torch.cat(sorted_ids_by_expert).to(torch.int32)
    return (
        sorted_ids,
        torch.tensor(expert_ids, dtype=torch.int32, device=topk_ids.device),
        torch.tensor(sorted_ids.numel(), dtype=torch.int32, device=topk_ids.device),
    )


def test_indexed_streamk_fp32_reduce_is_batch_invariant():
    """Keep split-K partials in FP32 until the final BF16 output conversion."""
    skip_if_unsupported(a_dtype=dtypes.bfloat16, mma_type="mma")

    small_m = 32
    large_m = 128
    shape_n = 1024
    shape_k = 2048
    num_experts = 16
    top_k = 8
    block_m = 32
    block_n = 256

    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    weight = torch.randint(
        0,
        16,
        (num_experts, shape_n, shape_k),
        dtype=torch.int32,
        device="cuda",
    )
    weight = transform_humming_weight(weight, dtypes.uint4, dtypes.bfloat16)
    weight_scale = torch.ones(
        (num_experts,),
        dtype=torch.float32,
        device=weight.device,
    )
    _, _, inputs, _ = generate_random_inputs(
        m=large_m,
        k=shape_k,
        group_size=0,
        dtype=dtypes.bfloat16,
    )
    generator = torch.Generator(device=inputs.device).manual_seed(19)
    scores = torch.randn(
        (large_m, num_experts),
        generator=generator,
        device=inputs.device,
    )
    topk_ids = torch.topk(scores, top_k, dim=1).indices.to(torch.int32)

    layer_config = LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        num_experts=num_experts,
        has_bias=False,
        mma_type="mma",
        weight_scale_type="tensor",
    )
    compute_config = ComputeConfig(
        use_f16_accum=False,
        gemm_type="indexed",
    )
    tuning_config = json.dumps(
        {
            "block_shape": (block_m, block_n, 64),
            "warp_shape": (block_m, 64, 64),
            "num_sms": 20,
            "num_stages": 3,
            "use_warp_spec": False,
            "use_tma": False,
            "use_cp_async": False,
            "use_stream_k": True,
            "use_fp32_stream_k_reduce": True,
        }
    )
    no_stream_config = json.loads(tuning_config)
    no_stream_config["use_stream_k"] = False
    no_stream_config["use_fp32_stream_k_reduce"] = False
    no_stream_config = json.dumps(no_stream_config)
    locks = torch.zeros((1024,), dtype=torch.int32, device=inputs.device)

    def run(shape_m: int, selected_tuning_config: str = tuning_config) -> torch.Tensor:
        sorted_ids, expert_ids, num_tokens_padded = _make_indexed_metadata(
            topk_ids[:shape_m],
            num_experts,
            block_m,
        )
        locks.zero_()
        output = torch.empty(
            (shape_m * top_k, shape_n),
            dtype=torch.bfloat16,
            device=inputs.device,
        )
        ops.humming_gemm(
            layer_config=layer_config.to_str(),
            compute_config=compute_config.to_str(),
            tuning_config=selected_tuning_config,
            inputs=inputs[:shape_m],
            weight=weight,
            weight_scale=weight_scale,
            outputs=output,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            locks=locks,
            top_k=top_k,
        )
        torch.cuda.synchronize(inputs.device)
        return output

    small_output = run(small_m)
    large_output = run(large_m)[: small_m * top_k]
    no_stream_output = run(small_m, no_stream_config)
    torch.testing.assert_close(small_output, no_stream_output, rtol=0.02, atol=0.02)

    reference = large_output.float()
    diff = (small_output.float() - reference).abs().flatten()
    reference_abs = reference.abs().flatten()
    relative_diff = diff / reference_abs.clamp_min(1e-12)
    not_close = diff > 0.02 + 0.02 * reference_abs

    assert torch.quantile(relative_diff, 0.99).item() < 0.02
    assert not_close.sum().item() <= 16
