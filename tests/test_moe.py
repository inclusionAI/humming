import pytest
import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.transform import (
    transform_humming_weight,
    transform_humming_weight_scale,
)
from humming.utils.test import (
    generate_random_inputs,
    generate_random_moe_tensors,
    generate_random_weight,
    skip_if_unsupported,
)


@pytest.mark.parametrize("m", [1, 18, 512])
@pytest.mark.parametrize("num_experts", [4, 19])
@pytest.mark.parametrize("top_k", [1, 2, 3])
@pytest.mark.parametrize("block_shape_m", [16, 48, 64])
def test_indexed_gemm(m, num_experts, top_k, block_shape_m):
    c_dtype = dtypes.bfloat16
    a_dtype = dtypes.bfloat16
    b_dtype = dtypes.uint4
    bs_dtype = dtypes.bfloat16

    moe_tensors = generate_random_moe_tensors(
        m,
        num_experts=num_experts,
        top_k=top_k,
        block_size_config=block_shape_m,
    )
    topk_ids, _, sorted_token_ids, expert_ids, num_tokens_padded = moe_tensors

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=0,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
        num_experts=num_experts,
    )

    _, weight_ref, weight, weight_scale, _, _ = random_weight_data
    weight = transform_humming_weight(weight, b_dtype, a_dtype)
    weight_scale = transform_humming_weight_scale(weight_scale, to_apply_on_c=True)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=m,
        k=1024,
        group_size=0,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=(block_shape_m, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(block_shape_m, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_experts=num_experts,
        num_stages=3,
        use_warp_spec=False,
        use_tma=False,
        use_cp_async=False,
        has_bias=False,
        mma_type="mma",
        use_stream_k=False,
        gemm_type="indexed",
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.empty((m * top_k, 1024), dtype=torch_dtype, device=inputs.device)

    outputs = ops.launch_kernel(
        configs=[humming_kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        sorted_ids=sorted_token_ids,
        top_k=top_k,
    )

    outputs = outputs.view(-1, outputs.size(-1))
    outputs_ref = torch.empty_like(outputs)

    for expert_id in range(num_experts):
        outputs_index = torch.where(topk_ids.view(-1) == expert_id)[0]
        inputs_index = outputs_index // top_k
        if inputs_index.size(0):
            tmp = inputs_ref[inputs_index].matmul(weight_ref[expert_id].T)
            outputs_ref[outputs_index] = tmp.to(torch_dtype)

    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.1)


@pytest.mark.parametrize("input_scale_group_size", [128])
def test_indexed_gemm_input_scale(input_scale_group_size):
    a_dtype = "float8e4m3"
    mma_type = "wgmma"
    skip_if_unsupported(a_dtype=a_dtype, mma_type=mma_type)

    c_dtype = dtypes.bfloat16
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.uint4
    bs_dtype = dtypes.bfloat16

    m = 256
    n = 1024
    k = 1024
    num_experts = 8
    top_k = 1
    block_shape_m = 48

    moe_tensors = generate_random_moe_tensors(
        m,
        num_experts=num_experts,
        top_k=top_k,
        block_size_config=block_shape_m,
    )
    topk_ids, _, sorted_token_ids, expert_ids, num_tokens_padded = moe_tensors

    random_weight_data = generate_random_weight(
        n=n,
        k=k,
        group_size=0,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
        num_experts=num_experts,
    )

    _, weight_ref, weight, weight_scale, _, _ = random_weight_data
    weight = transform_humming_weight(weight, b_dtype, a_dtype, use_wgmma=mma_type == "wgmma")
    weight_scale = transform_humming_weight_scale(weight_scale, to_apply_on_c=True)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=m,
        k=k,
        group_size=input_scale_group_size,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        shape_n=n,
        shape_k=k,
        block_shape=(block_shape_m, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(block_shape_m, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_experts=num_experts,
        num_stages=3,
        use_warp_spec=False,
        use_tma=False,
        use_cp_async=False,
        has_bias=False,
        input_scale_group_size=input_scale_group_size,
        mma_type=mma_type,
        use_stream_k=False,
        gemm_type="indexed",
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.empty((m * top_k, n), dtype=torch_dtype, device=inputs.device)

    outputs = ops.launch_kernel(
        configs=[humming_kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        sorted_ids=sorted_token_ids,
        top_k=top_k,
    )

    outputs = outputs.view(-1, outputs.size(-1))
    outputs_ref = torch.empty_like(outputs)

    for expert_id in range(num_experts):
        outputs_index = torch.where(topk_ids.view(-1) == expert_id)[0]
        inputs_index = outputs_index // top_k
        if inputs_index.size(0):
            tmp = inputs_ref[inputs_index].matmul(weight_ref[expert_id].T)
            outputs_ref[outputs_index] = tmp.to(torch_dtype)

    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.1)


@pytest.mark.parametrize("m", [1, 18, 512, 1274, 3487])
@pytest.mark.parametrize("num_experts", [4, 19])
@pytest.mark.parametrize("top_k", [1, 2, 3])
@pytest.mark.parametrize("block_shape_m", [16, 48, 64])
@pytest.mark.parametrize("expert_max_tokens", [None, 4, 64, 233, 666])
@pytest.mark.parametrize("use_tma", [True, False])
def test_grouped_gemm(m, num_experts, top_k, block_shape_m, expert_max_tokens, use_tma):
    c_dtype = dtypes.bfloat16
    a_dtype = dtypes.bfloat16
    b_dtype = dtypes.uint4
    bs_dtype = dtypes.bfloat16
    gemm_type = "grouped_contiguous" if expert_max_tokens is None else "grouped_masked"

    try:
        _, expert_layout, *_ = generate_random_moe_tensors(
            m,
            num_experts=num_experts,
            top_k=top_k,
            gemm_type=gemm_type,
            expert_max_tokens=expert_max_tokens,
        )
    except AssertionError as e:
        if "expert_max_tokens" in str(e):
            return
        raise

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=0,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
        num_experts=num_experts,
    )

    _, weight_ref, weight, weight_scale, _, _ = random_weight_data
    weight = transform_humming_weight(weight, b_dtype, a_dtype)
    weight_scale = transform_humming_weight_scale(weight_scale, to_apply_on_c=True)

    m_new = m * top_k
    if expert_max_tokens is not None:
        m_new = num_experts * expert_max_tokens

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=m_new,
        k=1024,
        group_size=0,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=(block_shape_m, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(block_shape_m, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_experts=num_experts,
        num_stages=3,
        use_warp_spec=False,
        has_bias=False,
        mma_type="mma",
        use_tma=use_tma,
        use_stream_k=False,
        gemm_type="grouped_contiguous" if expert_max_tokens is None else "grouped_masked",
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((m_new, 1024), dtype=torch_dtype, device=inputs.device)

    outputs = ops.launch_kernel(
        configs=[humming_kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        expert_layout=expert_layout,
    )

    outputs = outputs.view(-1, outputs.size(-1))
    outputs_ref = torch.zeros_like(outputs)

    for expert_id in range(num_experts):
        if expert_max_tokens is None:
            offset1 = expert_layout[expert_id]
            if expert_id == num_experts - 1:
                offset2 = m_new
            else:
                offset2 = expert_layout[expert_id + 1]
        else:
            offset1 = expert_max_tokens * expert_id
            offset2 = offset1 + expert_layout[expert_id]

        if offset2 == offset1:
            continue

        outputs_ref[offset1:offset2] = inputs_ref[offset1:offset2].matmul(weight_ref[expert_id].T)

    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.1)
