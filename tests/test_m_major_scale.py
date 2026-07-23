import pytest
import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.transform import transform_humming_weight, transform_humming_weight_scale
from humming.utils.test import (
    generate_random_inputs,
    generate_random_moe_tensors,
    generate_random_weight,
    skip_if_unsupported,
)


@pytest.mark.parametrize("a_dtype", ["float8e4m3"])
@pytest.mark.parametrize("b_dtype", ["float4e1m2"])
@pytest.mark.parametrize("c_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("shape_m", [256, 258])
@pytest.mark.parametrize("use_tma_as", [False, True])
def test_m_major_dense(a_dtype, b_dtype, c_dtype, shape_m, use_tma_as):
    skip_if_unsupported(a_dtype=a_dtype, use_warp_spec=True, use_tma=use_tma_as)
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.DataType.from_str(b_dtype)
    c_dtype = dtypes.DataType.from_str(c_dtype)
    group_size = 128

    _, weight_ref, weight, weight_scale, _, _ = generate_random_weight(
        n=1024, k=1024, group_size=group_size, dtype=b_dtype, scale_dtype=dtypes.float8e4m3
    )
    weight = transform_humming_weight(weight, b_dtype, a_dtype, use_wgmma=False)
    to_apply_on_c = a_dtype.num_bits != 16
    weight_scale = transform_humming_weight_scale(weight_scale, to_apply_on_c=to_apply_on_c)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=shape_m, k=1024, group_size=group_size, dtype=a_dtype
    )
    # store the scale M-major [num_groups, M] with M padded to a multiple of 4.
    num_groups = 1024 // group_size
    m_pad = (shape_m + 3) // 4 * 4
    input_scale_m_major = torch.zeros((num_groups, m_pad), dtype=torch.float32, device=inputs.device)
    input_scale_m_major[:, :shape_m] = input_scale.transpose(0, 1)

    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=(64, 128, 128),
        warp_shape=(64, 32, 128),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=dtypes.float8e4m3,
        num_stages=3,
        use_warp_spec=True,
        use_cp_async=True,
        use_tma=use_tma_as,
        use_m_major_input_scale=True,
        use_tma_as=use_tma_as,
        input_scale_group_size=group_size,
        weight_scale_group_size=group_size,
        weight_scale_type="group",
        mma_type="mma",
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((shape_m, 1024), dtype=torch_dtype, device=inputs.device)
    outputs = ops.launch_kernel(
        configs=[humming_kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale_m_major,
        weight_scale=weight_scale,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.1)


@pytest.mark.parametrize("a_dtype", ["float8e4m3"])
@pytest.mark.parametrize("b_dtype", ["uint4"])
@pytest.mark.parametrize("c_dtype", ["bfloat16"])
@pytest.mark.parametrize("expert_max_tokens", [None, 512])
@pytest.mark.parametrize("use_tma_as", [False, True])
def test_m_major_grouped(a_dtype, b_dtype, c_dtype, expert_max_tokens, use_tma_as):
    skip_if_unsupported(a_dtype=a_dtype, use_tma=use_tma_as)
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.DataType.from_str(b_dtype)
    c_dtype = dtypes.DataType.from_str(c_dtype)
    group_size = 128
    m, num_experts, top_k = 512, 4, 2
    gemm_type = "grouped_contiguous" if expert_max_tokens is None else "grouped_masked"

    # Every expert's token count must be a multiple of 4 (16B) to keep the M-major
    # scale aligned. masked uses expert_max_tokens; contiguous uses an aligned layout.
    if expert_max_tokens is None:
        tokens_per_expert = 512
        m_new = num_experts * tokens_per_expert
        expert_layout = torch.arange(0, m_new + 1, tokens_per_expert, dtype=torch.int64, device="cuda:0")
    else:
        _, expert_layout, *_ = generate_random_moe_tensors(
            m,
            num_experts=num_experts,
            top_k=top_k,
            gemm_type=gemm_type,
            expert_max_tokens=expert_max_tokens,
        )
        m_new = num_experts * expert_max_tokens

    _, weight_ref, weight, weight_scale, _, _ = generate_random_weight(
        n=1024,
        k=1024,
        group_size=0,
        dtype=b_dtype,
        scale_dtype=dtypes.bfloat16,
        num_experts=num_experts,
    )
    weight = transform_humming_weight(weight, b_dtype, a_dtype)
    weight_scale = transform_humming_weight_scale(weight_scale, to_apply_on_c=True)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=m_new, k=1024, group_size=group_size, dtype=a_dtype
    )
    input_scale = input_scale.transpose(0, 1).contiguous()

    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=(64, 128, 64),
        warp_shape=(64, 32, 64),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=dtypes.bfloat16,
        num_experts=num_experts,
        num_stages=3,
        use_warp_spec=False,
        use_cp_async=True,
        use_tma=use_tma_as,
        use_m_major_input_scale=True,
        use_tma_as=use_tma_as,
        input_scale_group_size=group_size,
        weight_scale_group_size=0,
        mma_type="mma",
        use_stream_k=False,
        gemm_type=gemm_type,
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

    outputs_ref = torch.zeros_like(outputs)
    for expert_id in range(num_experts):
        if expert_max_tokens is None:
            offset1 = expert_layout[expert_id]
            offset2 = expert_layout[expert_id + 1]
        else:
            offset1 = expert_max_tokens * expert_id
            offset2 = offset1 + expert_layout[expert_id]
        if offset2 == offset1:
            continue
        outputs_ref[offset1:offset2] = inputs_ref[offset1:offset2].matmul(weight_ref[expert_id].T)

    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.1)
