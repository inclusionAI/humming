"""MXFP4 W4A8 kernels with grouped FP8 input scales.

The production DeepEP FP8-dispatch path uses grouped FP8 activations together with
MXFP4 weights and E8M0 group scales.  The default layer policy should keep using
fused E8M0; non-fused coverage is kept only as an explicit kernel-level regression
test for the FP8 weight-scale tail decode path.
"""

import torch

import humming.layer as layer_module
from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.layer import HummingLayerMeta, HummingLayerMethod
from humming.utils.test import (
    generate_random_inputs,
    generate_random_moe_tensors,
    generate_random_weight,
    skip_if_unsupported,
)
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
)

A_DTYPE = dtypes.float8e4m3
B_DTYPE = dtypes.float4e2m1
BS_DTYPE = dtypes.float8e8m0
C_DTYPE = dtypes.bfloat16

N = 1024
K = 1024
INPUT_GROUP = 128
WEIGHT_GROUP = 32
RTOL = 0.05
ATOL = 0.5

NUM_EXPERTS = 8
TOP_K = 1


def _amplify_grouped_input_scale(inputs_ref: torch.Tensor, input_scale: torch.Tensor) -> None:
    ngroups = input_scale.shape[1]
    pattern = (2.0 ** ((torch.arange(ngroups, device=input_scale.device) % 5) - 2)).float()
    input_scale *= pattern
    inputs_ref *= pattern.repeat_interleave(INPUT_GROUP)


def _make_meta(input_scale_group_size: int = INPUT_GROUP) -> HummingLayerMeta:
    return HummingLayerMeta(
        shape_n=N,
        shape_k=K,
        a_dtype=A_DTYPE,
        b_dtype=B_DTYPE,
        c_dtype=C_DTYPE,
        bs_dtype=BS_DTYPE,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=WEIGHT_GROUP,
        mma_type="wgmma",
    )


def _prepare_inputs(m: int):
    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=m,
        k=K,
        group_size=INPUT_GROUP,
        dtype=A_DTYPE,
    )
    assert input_scale is not None
    _amplify_grouped_input_scale(inputs_ref, input_scale)
    return inputs_ref, inputs, input_scale


def _prepare_weight(*, use_fused_e8m0_scale: bool, num_experts: int | None = None):
    _, weight_ref, weight, weight_scale, _, global_scale = generate_random_weight(
        n=N,
        k=K,
        group_size=WEIGHT_GROUP,
        dtype=B_DTYPE,
        scale_dtype=BS_DTYPE,
        num_experts=num_experts,
    )
    assert weight_scale is not None

    if use_fused_e8m0_scale:
        meta = HummingLayerMeta(
            shape_n=N,
            shape_k=K,
            a_dtype=A_DTYPE,
            b_dtype=B_DTYPE,
            c_dtype=C_DTYPE,
            bs_dtype=BS_DTYPE,
            input_scale_group_size=INPUT_GROUP,
            weight_scale_group_size=WEIGHT_GROUP,
            weight_scale_type="group",
            num_experts=num_experts or 0,
            mma_type="wgmma",
            use_fused_e8m0_scale=True,
        )
        weight, weight_scale, global_scale = HummingLayerMethod.may_process_fused_e8m0_scale(
            meta,
            weight=weight,
            weight_scale=weight_scale,
            global_scale=global_scale,
        )
        weight = prepare_humming_weight(
            weight,
            B_DTYPE,
            A_DTYPE,
            use_wgmma=True,
            use_fused_e8m0_scale=True,
            interleave_mode=2,
        )
        weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=False)
    else:
        weight = prepare_humming_weight(weight, B_DTYPE, A_DTYPE, use_wgmma=True)
        weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=False)

    return weight_ref, weight, weight_scale, global_scale


def _dense_kernel(*, use_fused_e8m0_scale: bool):
    return HummingKernel(
        shape_n=N,
        shape_k=K,
        block_shape=(64, 128, 128),
        warp_shape=(64, 16, 128),
        a_dtype=A_DTYPE,
        b_dtype=B_DTYPE,
        c_dtype=C_DTYPE,
        bs_dtype=BS_DTYPE,
        num_stages=3,
        use_warp_spec=False,
        input_scale_group_size=INPUT_GROUP,
        weight_scale_group_size=WEIGHT_GROUP,
        weight_scale_type="group_tensor" if use_fused_e8m0_scale else "group",
        use_fused_e8m0_scale=use_fused_e8m0_scale,
        use_f16_accum=False,
        use_tma=False,
        use_cp_async=False,
        mma_type="wgmma",
        use_stream_k=False,
    )


def test_layer_auto_fuses_e8m0_with_grouped_input_scale():
    for input_scale_group_size in [0, INPUT_GROUP]:
        meta = _make_meta(input_scale_group_size)
        assert meta.use_fused_e8m0_scale is True
        assert meta.is_group_weight_scale is True
        assert meta.is_tensor_weight_scale is True


def test_layer_quant_input_uses_meta_group_size(monkeypatch):
    captured = {}

    def fake_quant_input(*, inputs, outputs, dtype, group_size):
        captured["inputs"] = inputs
        captured["outputs"] = outputs
        captured["dtype"] = dtype
        captured["group_size"] = group_size
        return "quanted", "scale"

    monkeypatch.setattr(layer_module.ops, "quant_input", fake_quant_input)

    layer = torch.nn.Module()
    layer.humming_metas = {"": _make_meta(INPUT_GROUP)}
    inputs = torch.empty((1, K), dtype=torch.bfloat16)

    quanted_input, input_scale = HummingLayerMethod.may_quant_input(layer, inputs)

    assert quanted_input == "quanted"
    assert input_scale == "scale"
    assert captured["inputs"] is inputs
    assert captured["dtype"] == str(A_DTYPE)
    assert captured["group_size"] == INPUT_GROUP


def test_dense_fused_mxfp4_grouped_fp8_input_wgmma():
    skip_if_unsupported(a_dtype=A_DTYPE, mma_type="wgmma")

    weight_ref, weight, weight_scale, global_scale = _prepare_weight(use_fused_e8m0_scale=True)
    inputs_ref, inputs, input_scale = _prepare_inputs(m=128)
    kernel = _dense_kernel(use_fused_e8m0_scale=True)

    outputs = torch.zeros((128, N), dtype=torch.bfloat16, device=inputs.device)
    outputs = ops.launch_kernel(
        configs=[kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        global_scale=global_scale,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch.bfloat16)
    torch.testing.assert_close(outputs, outputs_ref, rtol=RTOL, atol=ATOL)


def test_dense_nonfused_mxfp4_grouped_fp8_input_wgmma_tail_decode():
    skip_if_unsupported(a_dtype=A_DTYPE, mma_type="wgmma")

    weight_ref, weight, weight_scale, _ = _prepare_weight(use_fused_e8m0_scale=False)
    inputs_ref, inputs, input_scale = _prepare_inputs(m=128)
    kernel = _dense_kernel(use_fused_e8m0_scale=False)

    outputs = torch.zeros((128, N), dtype=torch.bfloat16, device=inputs.device)
    outputs = ops.launch_kernel(
        configs=[kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch.bfloat16)
    torch.testing.assert_close(outputs, outputs_ref, rtol=RTOL, atol=ATOL)


def test_grouped_masked_moe_fused_mxfp4_grouped_fp8_input_wgmma():
    skip_if_unsupported(a_dtype=A_DTYPE, mma_type="wgmma")

    expert_max_tokens = 256
    expert_layout_vals = [0, 1, 63, 64, 65, 129, 200, expert_max_tokens]
    assert len(expert_layout_vals) == NUM_EXPERTS
    expert_layout = torch.tensor(expert_layout_vals, dtype=torch.int32, device="cuda:0")

    weight_ref, weight, weight_scale, global_scale = _prepare_weight(
        use_fused_e8m0_scale=True,
        num_experts=NUM_EXPERTS,
    )
    m_total = NUM_EXPERTS * expert_max_tokens
    inputs_ref, inputs, input_scale = _prepare_inputs(m=m_total)

    kernel = HummingKernel(
        shape_n=N,
        shape_k=K,
        block_shape=(64, 128, 128),
        warp_shape=(64, 16, 128),
        a_dtype=A_DTYPE,
        b_dtype=B_DTYPE,
        c_dtype=C_DTYPE,
        bs_dtype=BS_DTYPE,
        num_experts=NUM_EXPERTS,
        num_stages=3,
        use_warp_spec=False,
        input_scale_group_size=INPUT_GROUP,
        weight_scale_group_size=WEIGHT_GROUP,
        weight_scale_type="group_tensor",
        use_fused_e8m0_scale=True,
        use_f16_accum=False,
        has_bias=False,
        use_tma=False,
        use_cp_async=False,
        mma_type="wgmma",
        use_stream_k=False,
        gemm_type="grouped_masked",
    )

    outputs = torch.zeros((m_total, N), dtype=torch.bfloat16, device=inputs.device)
    outputs = ops.launch_kernel(
        configs=[kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        global_scale=global_scale,
        expert_layout=expert_layout,
    ).view(-1, N)

    outputs_ref = torch.zeros_like(outputs)
    active_mask = torch.zeros(m_total, dtype=torch.bool, device=outputs.device)
    for expert_id, num_tokens in enumerate(expert_layout_vals):
        start = expert_max_tokens * expert_id
        end = start + num_tokens
        if end > start:
            active_mask[start:end] = True
            outputs_ref[start:end] = (
                inputs_ref[start:end].matmul(weight_ref[expert_id].T).to(torch.bfloat16)
            )

    assert (outputs[~active_mask] == 0).all(), "padding rows must be exactly zero"
    torch.testing.assert_close(outputs[active_mask], outputs_ref[active_mask], rtol=RTOL, atol=ATOL)


def test_indexed_moe_fused_mxfp4_grouped_fp8_input_wgmma():
    skip_if_unsupported(a_dtype=A_DTYPE, mma_type="wgmma")

    m = 256
    block_m = 48
    topk_ids, _, sorted_token_ids, expert_ids, num_tokens_padded = generate_random_moe_tensors(
        m,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        block_size_config=block_m,
    )

    weight_ref, weight, weight_scale, global_scale = _prepare_weight(
        use_fused_e8m0_scale=True,
        num_experts=NUM_EXPERTS,
    )
    inputs_ref, inputs, input_scale = _prepare_inputs(m=m)

    kernel = HummingKernel(
        shape_n=N,
        shape_k=K,
        block_shape=(block_m, 128, 64),
        warp_shape=(block_m, 32, 64),
        a_dtype=A_DTYPE,
        b_dtype=B_DTYPE,
        c_dtype=C_DTYPE,
        bs_dtype=BS_DTYPE,
        num_experts=NUM_EXPERTS,
        num_stages=3,
        use_warp_spec=False,
        input_scale_group_size=INPUT_GROUP,
        weight_scale_group_size=WEIGHT_GROUP,
        weight_scale_type="group_tensor",
        use_fused_e8m0_scale=True,
        use_f16_accum=False,
        has_bias=False,
        use_tma=False,
        use_cp_async=False,
        mma_type="wgmma",
        use_stream_k=False,
        gemm_type="indexed",
    )

    outputs = torch.empty((m * TOP_K, N), dtype=torch.bfloat16, device=inputs.device)
    outputs = ops.launch_kernel(
        configs=[kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        global_scale=global_scale,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        sorted_ids=sorted_token_ids,
        top_k=TOP_K,
    ).view(-1, N)

    outputs_ref = torch.empty_like(outputs)
    for expert_id in range(NUM_EXPERTS):
        outputs_index = torch.where(topk_ids.view(-1) == expert_id)[0]
        inputs_index = outputs_index // TOP_K
        if inputs_index.size(0):
            outputs_ref[outputs_index] = (
                inputs_ref[inputs_index].matmul(weight_ref[expert_id].T).to(torch.bfloat16)
            )

    torch.testing.assert_close(outputs, outputs_ref, rtol=RTOL, atol=ATOL)
