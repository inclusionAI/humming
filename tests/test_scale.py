import pytest
import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.utils.test import (
    generate_random_inputs,
    generate_random_weight,
    skip_if_unsupported,
)
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
)


@pytest.mark.parametrize("a_dtype", ["float16", "bfloat16", "float8e4m3", "int8", "int4"])
@pytest.mark.parametrize(
    "b_dtype",
    [
        "uint3",
        "uint4",
        "uint5",
        "uint8",
        "int4",
        "int8",
        "float3e2m0",
        "float4e1m2",
        "float8e1m6",
    ],
)
@pytest.mark.parametrize("c_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("bs_dtype", ["float16", "bfloat16", "float8e5m2", "float8e4m3"])
@pytest.mark.parametrize("input_scale_group_size", [16, 32, 64, 128, 0])
@pytest.mark.parametrize("weight_scale_group_size", [16, 32, 64, 128, 0])
@pytest.mark.parametrize("mma_type", ["mma", "wgmma"])
def test_scale(
    a_dtype,
    b_dtype,
    c_dtype,
    bs_dtype,
    input_scale_group_size,
    weight_scale_group_size,
    mma_type,
):
    skip_if_unsupported(a_dtype=a_dtype, mma_type=mma_type)
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.DataType.from_str(b_dtype)
    c_dtype = dtypes.DataType.from_str(c_dtype)
    bs_dtype = dtypes.DataType.from_str(bs_dtype)

    if b_dtype.is_integer_type and a_dtype.is_integer_type:
        if a_dtype.num_bits < b_dtype.num_bits:
            return
        elif a_dtype.num_bits == b_dtype.num_bits and not b_dtype.is_signed:
            return
        elif a_dtype.num_bits > b_dtype.num_bits and b_dtype.is_signed:
            return
    elif b_dtype.is_integer_type and a_dtype.is_floating_point_type:
        if b_dtype.is_signed or b_dtype.num_bits > a_dtype.mantissa_bits + 2:
            return
    elif b_dtype.is_floating_point_type and a_dtype.is_floating_point_type:
        if b_dtype.exponent_bits > a_dtype.exponent_bits:
            return
        if b_dtype.mantissa_bits > a_dtype.mantissa_bits:
            return
    elif b_dtype.is_floating_point_type and a_dtype.is_integer_type:
        return

    if a_dtype in [dtypes.float16, dtypes.bfloat16] and a_dtype != c_dtype:
        return
    if bs_dtype in [dtypes.float16, dtypes.bfloat16] and bs_dtype != c_dtype:
        return

    if mma_type == "wgmma" and a_dtype == dtypes.int4:
        return
    if input_scale_group_size > 0 and input_scale_group_size < 256 // a_dtype.num_bits:
        return
    if weight_scale_group_size > 0 and weight_scale_group_size < 256 // a_dtype.num_bits:
        return
    if input_scale_group_size > 0 and weight_scale_group_size > 0:
        if input_scale_group_size != weight_scale_group_size:
            return

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=weight_scale_group_size,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
    )

    _, weight_ref, weight, weight_scale, _, _ = random_weight_data
    if a_dtype == dtypes.int8 and b_dtype == dtypes.int8:
        weight = (weight.view(torch.int8) + 128).view(torch.int32)

    if mma_type == "mma":
        to_apply_on_c = weight_scale_group_size == 0 or a_dtype.num_bits != 16
    elif mma_type == "wgmma":
        to_apply_on_c = weight_scale_group_size == 0

    weight = prepare_humming_weight(weight, b_dtype, a_dtype, use_wgmma=mma_type == "wgmma")
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=to_apply_on_c)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=128,
        k=1024,
        group_size=input_scale_group_size,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=(16, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(16, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_stages=3,
        use_warp_spec=False,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        weight_scale_type="group" if weight_scale_group_size else "channel",
        use_tma=False,
        use_cp_async=False,
        mma_type=mma_type,
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((128, 1024), dtype=torch_dtype, device=inputs.device)
    assert weight_scale is not None
    outputs = ops.launch_kernel(
        configs=[humming_kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    if a_dtype.num_bits == 16 and weight_scale.size(-2) > 1:
        weight_ref = weight_ref.to(torch_dtype).float()

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.1)


@pytest.mark.parametrize("a_dtype", ["float16", "bfloat16", "float8e4m3", "int8"])
@pytest.mark.parametrize("b_dtype", ["uint5", "float6e1m4"])
@pytest.mark.parametrize("c_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("bs_dtype", [None, "float16", "bfloat16", "float8e4m3"])
@pytest.mark.parametrize("input_scale_group_size", [0, 32, 64])
@pytest.mark.parametrize("weight_scale_group_size", [16, 32, 64])
@pytest.mark.parametrize("use_f16_accum", [True, False])
def test_global_scale(
    a_dtype,
    b_dtype,
    c_dtype,
    bs_dtype,
    input_scale_group_size,
    weight_scale_group_size,
    use_f16_accum,
):
    skip_if_unsupported(a_dtype=a_dtype)
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.DataType.from_str(b_dtype)
    c_dtype = dtypes.DataType.from_str(c_dtype)
    if bs_dtype is not None:
        bs_dtype = dtypes.DataType.from_str(bs_dtype)
    else:
        weight_scale_group_size = 0

    if use_f16_accum and c_dtype != dtypes.float16:
        return

    if use_f16_accum and not isinstance(a_dtype, dtypes.FloatingPointType):
        return

    if b_dtype.is_integer_type and a_dtype.is_integer_type:
        if a_dtype.num_bits < b_dtype.num_bits:
            return
        elif a_dtype.num_bits == b_dtype.num_bits and not b_dtype.is_signed:
            return
        elif a_dtype.num_bits > b_dtype.num_bits and b_dtype.is_signed:
            return
    elif b_dtype.is_integer_type and a_dtype.is_floating_point_type:
        if b_dtype.is_signed or b_dtype.num_bits > a_dtype.mantissa_bits + 2:
            return
    elif b_dtype.is_floating_point_type and a_dtype.is_floating_point_type:
        if b_dtype.exponent_bits > a_dtype.exponent_bits:
            return
        if b_dtype.mantissa_bits > a_dtype.mantissa_bits:
            return
    elif b_dtype.is_floating_point_type and a_dtype.is_integer_type:
        return

    if a_dtype in [dtypes.float16, dtypes.bfloat16] and a_dtype != c_dtype:
        return
    if bs_dtype in [dtypes.float16, dtypes.bfloat16] and bs_dtype != c_dtype:
        return

    if weight_scale_group_size > 0 and weight_scale_group_size < 256 // a_dtype.num_bits:
        return
    if input_scale_group_size > 0 and input_scale_group_size < 256 // a_dtype.num_bits:
        return
    if input_scale_group_size > 0 and weight_scale_group_size > 0:
        if input_scale_group_size != weight_scale_group_size:
            return

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=weight_scale_group_size,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
        has_global_scale=True,
    )

    _, weight_ref, weight, weight_scale, _, global_scale = random_weight_data
    global_scale = global_scale.float().view(1)
    if a_dtype == dtypes.int8 and b_dtype == dtypes.int8:
        weight = (weight.view(torch.int8) + 128).view(torch.int32)

    to_apply_on_c = weight_scale_group_size == 0 or a_dtype.num_bits != 16

    weight = prepare_humming_weight(weight, b_dtype, a_dtype)
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=to_apply_on_c)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=128,
        k=1024,
        group_size=input_scale_group_size,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=(16, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(16, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype or dtypes.float32,
        num_stages=3,
        use_warp_spec=False,
        input_scale_group_size=input_scale_group_size,
        weight_scale_type="tensor" if bs_dtype is None else "group_tensor",
        weight_scale_group_size=weight_scale_group_size,
        use_f16_accum=use_f16_accum,
        use_tma=False,
        use_cp_async=False,
        mma_type="mma",
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((128, 1024), dtype=torch_dtype, device=inputs.device)

    outputs = ops.launch_kernel(
        configs=[humming_kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        global_scale=global_scale,
    )

    if a_dtype.num_bits == 16 and weight_scale is not None and weight_scale.size(-2) > 1:
        weight_ref = weight_ref.to(torch_dtype).float()

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.5)


@pytest.mark.parametrize("a_dtype", ["int8", "int4"])
@pytest.mark.parametrize("b_dtype", ["uint3", "int4"])
@pytest.mark.parametrize("c_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("bs_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("weight_scale_group_size", [16, 32, 64])
@pytest.mark.parametrize("has_global_scale", [True, False])
def test_int_weight_scale(
    a_dtype,
    b_dtype,
    c_dtype,
    bs_dtype,
    weight_scale_group_size,
    has_global_scale,
):
    skip_if_unsupported(a_dtype=a_dtype)
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.DataType.from_str(b_dtype)
    c_dtype = dtypes.DataType.from_str(c_dtype)
    bs_dtype = dtypes.DataType.from_str(bs_dtype)

    if b_dtype.is_integer_type and a_dtype.is_integer_type:
        if a_dtype.num_bits < b_dtype.num_bits:
            return
        elif a_dtype.num_bits == b_dtype.num_bits and not b_dtype.is_signed:
            return
        elif a_dtype.num_bits > b_dtype.num_bits and b_dtype.is_signed:
            return
    elif b_dtype.is_integer_type and a_dtype.is_floating_point_type:
        if b_dtype.is_signed or b_dtype.num_bits > a_dtype.mantissa_bits + 2:
            return
    elif b_dtype.is_floating_point_type and a_dtype.is_floating_point_type:
        if b_dtype.exponent_bits > a_dtype.exponent_bits:
            return
        if b_dtype.mantissa_bits > a_dtype.mantissa_bits:
            return
    elif b_dtype.is_floating_point_type and a_dtype.is_integer_type:
        return

    if a_dtype in [dtypes.float16, dtypes.bfloat16] and a_dtype != c_dtype:
        return
    if bs_dtype in [dtypes.float16, dtypes.bfloat16] and bs_dtype != c_dtype:
        return

    if weight_scale_group_size > 0 and weight_scale_group_size < 256 // a_dtype.num_bits:
        return

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=weight_scale_group_size,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
        has_global_scale=has_global_scale,
    )

    _, weight_ref, weight, weight_scale, _, global_scale = random_weight_data
    dtype = weight_scale.dtype
    if a_dtype == dtypes.int8 and b_dtype == dtypes.int8:
        weight = (weight.view(torch.int8) + 128).view(torch.int32)

    to_apply_on_c = weight_scale_group_size == 0 or a_dtype.num_bits != 16

    weight_ref = weight_ref / weight_scale.repeat_interleave(weight_scale_group_size, 1)
    scale_factor = weight_scale.float().abs().max() / 1024
    weight_scale = (weight_scale.float() / scale_factor).round().to(torch.int16)
    weight_ref = weight_ref * weight_scale.repeat_interleave(weight_scale_group_size, 1).float()
    weight_ref = weight_ref * scale_factor
    weight_scale = weight_scale.view(dtype)

    weight = prepare_humming_weight(weight, b_dtype, a_dtype)
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=to_apply_on_c)
    if has_global_scale:
        global_scale = global_scale * scale_factor
    else:
        global_scale = torch.full((1,), scale_factor, dtype=torch.float32, device=weight_ref.device)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=128,
        k=1024,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=(16, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(16, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_stages=3,
        use_warp_spec=False,
        weight_scale_group_size=weight_scale_group_size,
        weight_scale_type="group_tensor",
        use_int_weight_scale=True,
        use_tma=False,
        use_cp_async=False,
        mma_type="mma",
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((128, 1024), dtype=torch_dtype, device=inputs.device)

    outputs = ops.launch_kernel(
        configs=[humming_kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        global_scale=global_scale,
    )

    if a_dtype.num_bits == 16 and weight_scale.size(-2) > 1:
        weight_ref = weight_ref.to(torch_dtype).float()

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.5)


@pytest.mark.parametrize("c_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("weight_scale_group_size", [32, 64])
@pytest.mark.parametrize("has_global_scale", [True, False])
@pytest.mark.parametrize("input_scale_group_size", [0, 128])
@pytest.mark.parametrize("mma_type", ["mma", "wgmma"])
def test_fused_e8m0_weight_scale(
    c_dtype, weight_scale_group_size, has_global_scale, input_scale_group_size, mma_type
):
    a_dtype = dtypes.float8e4m3
    b_dtype = dtypes.float4e2m1
    skip_if_unsupported(a_dtype=a_dtype, mma_type=mma_type)
    c_dtype = dtypes.DataType.from_str(c_dtype)
    bs_dtype = dtypes.float8e8m0

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=weight_scale_group_size,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
        has_global_scale=has_global_scale,
    )

    _, weight_ref, weight, weight_scale, _, global_scale = random_weight_data

    bs_dtype = dtypes.float8e8m0

    origin_dtype = weight_scale.dtype
    weight_scale = weight_scale.view(torch.uint8)
    scale_max = weight_scale.max()
    scale_min = weight_scale.min()
    scale_range = scale_max - scale_min
    max_range = 11
    if a_dtype == dtypes.float8e5m2:
        max_range = max_range - 1
    max_range = torch.tensor(max_range, dtype=torch.uint8, device=scale_range.device)
    scale_range = scale_range.minimum(max_range)
    scale_min_new = scale_max - scale_range
    weight_scale = weight_scale.maximum(scale_min_new) - scale_min_new + 1
    weight_scale = weight_scale & 0xF
    weight_scale = weight_scale.view(origin_dtype)

    scale_factor = 2 ** (scale_min_new.view(-1).float() - 127)
    scale_factor = scale_factor / 2
    weight = prepare_humming_weight(weight, b_dtype, a_dtype, interleave_mode=2)
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=False)
    if has_global_scale:
        global_scale = global_scale * scale_factor
    else:
        global_scale = scale_factor

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=1024,
        k=1024,
        group_size=input_scale_group_size,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=(16, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(16, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_stages=3,
        use_warp_spec=False,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        weight_scale_type="group_tensor",
        use_fused_e8m0_scale=True,
        use_tma=False,
        use_cp_async=False,
        mma_type=mma_type,
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((1024, 1024), dtype=torch_dtype, device=inputs.device)

    outputs = ops.launch_kernel(
        configs=[humming_kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        global_scale=global_scale,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.5)


@pytest.mark.parametrize("a_dtype", ["float16", "bfloat16", "float8e4m3", "int8", "int4"])
@pytest.mark.parametrize("b_dtype", ["uint3", "int8", "float4e1m2", "float8e1m6"])
@pytest.mark.parametrize("c_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("block_shape", [(128, 128), (256, 64), (64, 32), (256, 512), (64, 64)])
@pytest.mark.parametrize("is_channel_input_scale", [True, False])
@pytest.mark.parametrize("use_f16_accum", [True, False])
@pytest.mark.parametrize("mma_type", ["mma", "wgmma"])
def test_block_scale(
    a_dtype,
    b_dtype,
    c_dtype,
    block_shape,
    is_channel_input_scale,
    use_f16_accum,
    mma_type,
):
    skip_if_unsupported(a_dtype=a_dtype, mma_type=mma_type)
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.DataType.from_str(b_dtype)
    c_dtype = dtypes.DataType.from_str(c_dtype)

    if use_f16_accum:
        if c_dtype != dtypes.float16:
            return
        if a_dtype not in [dtypes.float16, dtypes.float8e4m3]:
            return

    if b_dtype.is_integer_type and a_dtype.is_integer_type:
        if a_dtype.num_bits < b_dtype.num_bits:
            return
        elif a_dtype.num_bits == b_dtype.num_bits and not b_dtype.is_signed:
            return
        elif a_dtype.num_bits > b_dtype.num_bits and b_dtype.is_signed:
            return
    elif b_dtype.is_integer_type and a_dtype.is_floating_point_type:
        if b_dtype.is_signed or b_dtype.num_bits > a_dtype.mantissa_bits + 2:
            return
    elif b_dtype.is_floating_point_type and a_dtype.is_floating_point_type:
        if b_dtype.exponent_bits > a_dtype.exponent_bits:
            return
        if b_dtype.mantissa_bits > a_dtype.mantissa_bits:
            return
    elif b_dtype.is_floating_point_type and a_dtype.is_integer_type:
        return

    if a_dtype in [dtypes.float16, dtypes.bfloat16] and a_dtype != c_dtype:
        return

    if mma_type == "wgmma" and a_dtype == dtypes.int4:
        return
    input_scale_group_size = 0 if is_channel_input_scale else block_shape[1]
    weight_scale_group_size = block_shape[1]
    if weight_scale_group_size > 0 and weight_scale_group_size < 256 // a_dtype.num_bits:
        return

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=weight_scale_group_size,
        group_size_n=block_shape[0],
        dtype=b_dtype,
        scale_dtype=dtypes.float32,
    )

    _, weight_ref, weight, weight_scale, _, _ = random_weight_data
    if a_dtype == dtypes.int8 and b_dtype == dtypes.int8:
        weight = (weight.view(torch.int8) + 128).view(torch.int32)

    weight = prepare_humming_weight(weight, b_dtype, a_dtype, use_wgmma=mma_type == "wgmma")
    weight_scale = weight_scale.transpose(-1, -2).contiguous()

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=128,
        k=1024,
        group_size=input_scale_group_size,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=(16, a_dtype.num_bits * 16, 1024 // a_dtype.num_bits),
        warp_shape=(16, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=dtypes.float32,
        num_stages=3,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        weight_scale_group_size_n=block_shape[0],
        use_f16_accum=use_f16_accum,
        weight_scale_type="block",
        use_warp_spec=False,
        use_tma=False,
        use_cp_async=False,
        mma_type=mma_type,
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((128, 1024), dtype=torch_dtype, device=inputs.device)
    outputs = ops.launch_kernel(
        configs=[humming_kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    if a_dtype.num_bits == 16 and weight_scale.size(-2) > 1:
        ws_full = (
            weight_scale.transpose(-1, -2)
            .repeat_interleave(block_shape[0], -2)
            .repeat_interleave(weight_scale_group_size, -1)
            .float()
        )
        weight_ref_no_scale = weight_ref / ws_full
        weight_ref = weight_ref_no_scale.to(torch_dtype) * ws_full.to(torch_dtype)
        weight_ref = weight_ref.float()

    atol = 0.2
    if a_dtype.num_bits == 16 and use_f16_accum:
        prev_fp16_accum = torch.backends.cuda.matmul.allow_fp16_accumulation
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        try:
            outputs_ref = inputs_ref.to(torch_dtype).matmul(weight_ref.to(torch_dtype).T).to(torch_dtype)
        finally:
            torch.backends.cuda.matmul.allow_fp16_accumulation = prev_fp16_accum
        atol = 0.5
    else:
        outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=atol)
