import pytest
from humming import dtypes
import torch
from humming.utils.test import (
    generate_random_weight,
    generate_random_inputs,
    generate_random_moe_tensors,
)
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
)
from humming.kernel.humming import HummingKernel


@pytest.mark.parametrize("m", [1, 18, 512])
@pytest.mark.parametrize("num_experts", [4, 19])
@pytest.mark.parametrize("top_k", [2, 3])
@pytest.mark.parametrize("is_moe_down", [True, False])
def test_moe(m, num_experts, top_k, is_moe_down):
    c_dtype = dtypes.bfloat16
    a_dtype = dtypes.bfloat16
    b_dtype = dtypes.uint4
    bs_dtype = dtypes.bfloat16

    topk_ids, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded = (
        generate_random_moe_tensors(
            m,
            num_experts=num_experts,
            topk=top_k,
            block_size=32,
        )
    )

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=0,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
        num_experts=num_experts,
    )

    _, weight_ref, weight, weight_scale, _, _ = random_weight_data
    weight = prepare_humming_weight(weight, b_dtype, a_dtype)
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=True)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=m * (top_k if is_moe_down else 1),
        k=1024,
        group_size=0,
        dtype=a_dtype,
    )
    if is_moe_down:
        inputs = inputs.view(m, top_k, -1)

    humming_kernel = HummingKernel(
        problem_shape=(0, 1024, 1024),
        block_shape=(32, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(32, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_stages=3,
        use_warp_spec=False,
        use_tma=False,
        use_cp_async=False,
        has_bias=False,
        mma_type="mma",
        use_stream_k=False,
        is_moe=True,
        top_k=top_k,
        is_moe_down=is_moe_down,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.empty((m, top_k, 1024), dtype=torch_dtype, device=inputs.device)

    outputs = humming_kernel(
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        topk_weights=topk_weights,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        sorted_token_ids=sorted_token_ids,
    )

    outputs = outputs.view(-1, outputs.size(-1))
    outputs_ref = torch.empty_like(outputs)

    for expert_id in range(num_experts):
        outputs_index = torch.where(topk_ids.view(-1) == expert_id)[0]
        inputs_index = outputs_index if is_moe_down else outputs_index // top_k
        if inputs_index.size(0):
            tmp = inputs_ref[inputs_index].matmul(weight_ref[expert_id].T)
            if is_moe_down:
                tmp = tmp.float() * topk_weights.view(-1, 1)[outputs_index]
            outputs_ref[outputs_index] = tmp.to(torch_dtype)

    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.1)
