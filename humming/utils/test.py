import torch

from humming import dtypes
from humming.kernel.dequant_weight import DequantKernel
from humming.utils.weight import quantize_weight
import math


def generate_random_inputs(
    m: int,
    k: int,
    group_size: int = 0,
    dtype: dtypes.DataType = dtypes.float32,
):
    group_size = group_size if group_size > 0 else k
    input_scale = torch.randn((m, k // group_size), dtype=torch.float32, device="cuda:0")
    assert k % group_size == 0

    inputs_orig = torch.randn((m, k), dtype=torch.float32, device="cuda:0")
    init_scale = torch.rand((m, k // group_size), dtype=torch.float32, device="cuda:0")
    inputs_orig = inputs_orig * init_scale.repeat_interleave(group_size, 1)
    inputs_orig = inputs_orig / inputs_orig.std()

    if dtype in [dtypes.float16, dtypes.bfloat16]:
        torch_dtype = torch.float16 if dtype == dtypes.float16 else torch.bfloat16
        inputs = inputs_orig.to(torch_dtype)
        inputs_ref = inputs.float()
        input_scale = None
    else:
        inputs, input_scale, *_ = quantize_weight(
            inputs_orig,
            dtype=dtype,
            scale_dtype=dtypes.float32,
            group_size=group_size,
            has_dynamic_zp=False,
            has_global_scale=False,
        )

        inputs_ref = inputs.float()
        if dtype.is_floating_point_type:
            kernel = DequantKernel(device_index=inputs.device.index)
            inputs_ref = kernel(
                inputs,
                outputs=None,
                exponent_bits=dtype.exponent_bits,
                mantissa_bits=dtype.mantissa_bits,
                is_signed=True,
            )

        if dtype in [dtypes.int4, dtypes.int8]:
            inputs = inputs.to(torch.int8)
            if dtype == dtypes.int4:
                inputs = inputs.to(torch.uint8) & 0xF
                inputs = inputs[..., 1::2] * 16 + inputs[..., ::2]
                inputs = inputs.view(torch.uint8)
        elif dtype == dtypes.float4e2m1:
            inputs = inputs.to(torch.uint8)
            inputs = inputs[..., 1::2] * 16 + inputs[..., ::2]
        elif dtype == dtypes.float8e4m3:
            inputs = inputs.to(torch.uint8).view(torch.float8_e4m3fn)
        elif dtype == dtypes.float8e5m2:
            inputs = inputs.to(torch.uint8).view(torch.float8_e5m2)

        inputs_ref = inputs_ref * input_scale.repeat_interleave(group_size, 1)

    return inputs_orig, inputs_ref, inputs, input_scale


def generate_random_weight(
    n,
    k,
    group_size,
    dtype,
    scale_dtype,
    num_experts=None,
    has_global_scale=False,
    has_dynamic_zp=False,
):
    e = 1 if num_experts is None else num_experts
    dtype_orig = dtype
    group_size = group_size if group_size > 0 else k
    if has_dynamic_zp:
        assert dtype.is_integer_type and not dtype.is_signed, (
            "dynamic zero point only supports for uint dtype"
        )

    if dtype.is_integer_type and dtype.is_signed:
        dtype = dtypes.InergerType(False, dtype.num_bits)

    weight_orig = torch.randn((e, n, k), dtype=torch.float32, device="cuda:0")
    init_weight_scale = torch.rand((e, n, k // group_size), dtype=torch.float32, device="cuda:0")
    init_weight_scale = init_weight_scale + 0.01
    init_weight_bias = torch.randn((e, n, k // group_size), dtype=torch.float32, device="cuda:0")
    init_weight_bias = init_weight_bias

    weight_orig = weight_orig + init_weight_bias.repeat_interleave(group_size, -1)
    weight_orig = weight_orig * init_weight_scale.repeat_interleave(group_size, -1)
    weight_orig = weight_orig / weight_orig.std()

    quanted_weight, weight_scale, zero_point, global_scale = quantize_weight(
        weight_orig,
        dtype=dtype,
        scale_dtype=scale_dtype,
        group_size=group_size,
        has_dynamic_zp=has_dynamic_zp,
        has_global_scale=has_global_scale,
    )

    if dtype.is_integer_type and has_dynamic_zp:
        weight_ref = quanted_weight.float() - zero_point.repeat_interleave(group_size, -1)
    elif dtype.is_integer_type and not has_dynamic_zp:
        weight_ref = quanted_weight.float() - 2 ** (dtype.num_bits - 1)
    else:
        weight_ref = torch.empty_like(quanted_weight).view(torch.float32)
        dequant_kernel = DequantKernel()
        dequant_kernel(
            inputs=quanted_weight,
            outputs=weight_ref,
            exponent_bits=dtype.exponent_bits,
            mantissa_bits=dtype.mantissa_bits,
            is_signed=dtype.is_signed,
        )

    if weight_scale is not None:
        weight_ref = weight_ref * weight_scale.float().repeat_interleave(group_size, -1)

    if has_global_scale:
        weight_ref = weight_ref * global_scale.view(-1, 1, 1)

    if dtype_orig.is_integer_type and dtype_orig.is_signed:
        quanted_weight = quanted_weight - 2 ** (dtype.num_bits - 1)

    if num_experts is None:
        weight_orig = weight_orig.squeeze(0)
        weight_ref = weight_ref.squeeze(0)
        quanted_weight = quanted_weight.squeeze(0)
        if weight_scale is not None:
            weight_scale = weight_scale.squeeze(0)
        if global_scale is not None:
            global_scale = global_scale.squeeze(0)
        if zero_point is not None:
            zero_point = zero_point.squeeze(0)

    return weight_orig, weight_ref, quanted_weight, weight_scale, zero_point, global_scale


def generate_random_bias(n, dtype):
    assert dtype in [dtypes.float16, dtypes.bfloat16]
    torch_dtype = torch.float16 if dtype == dtypes.float16 else torch.bfloat16
    bias = torch.randn((n,), dtype=torch_dtype, device="cuda:0")
    return bias


def generate_random_moe_tensors(m, num_experts, topk, block_size):
    tensor = torch.randn((m, num_experts), dtype=torch.float32, device="cuda:0")
    topk_weights, topk_ids = tensor.topk(topk, 1)
    topk_ids = topk_ids.int()

    # TODO: moe_align_block_size cuda kernel
    part_token_ids_list = []
    expert_id_list = []
    for expert_id in range(num_experts):
        part_token_ids = torch.where(topk_ids.view(-1) == expert_id)[0]
        num_blocks = math.ceil(part_token_ids.size(0) / block_size)
        padded_size = num_blocks * block_size
        pad_size = padded_size - part_token_ids.size(0)
        part_token_ids = torch.nn.functional.pad(
            part_token_ids,
            pad=(0, pad_size),
            value=topk_ids.nelement(),
        )
        part_token_ids_list.append(part_token_ids)
        expert_id_list += [expert_id] * num_blocks

    sorted_token_ids = torch.cat(part_token_ids_list, dim=0).to(torch.int32)
    expert_ids = torch.tensor(expert_id_list, dtype=torch.int32, device=tensor.device)
    num_tokens_post_padded = torch.tensor(
        sorted_token_ids.size(0),
        dtype=torch.int32,
        device=tensor.device,
    )

    return topk_ids, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded
