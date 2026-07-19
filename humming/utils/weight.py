import torch

from humming import dtypes, ops


def quantize_weight(
    weight: torch.Tensor,
    dtype: dtypes.DataType,
    scale_dtype: dtypes.DataType | None,
    group_size: int,
    group_size_n: int | None = None,
    has_zero_point: bool = False,
    weight_scale_2_type: str | None = None,
    is_fp_zero_point: bool = False,
    pack: bool = False,
    allow_negative_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    assert weight.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert weight.ndim in [2, 3]
    assert not has_zero_point or scale_dtype is not None

    assert weight_scale_2_type in [None, "none", "tensor", "channel"]
    has_tensor_scale = weight_scale_2_type == "tensor"
    has_channel_scale_2 = weight_scale_2_type == "channel"
    assert not has_channel_scale_2 or scale_dtype is not None, (
        "channel weight_scale_2 requires a groupwise weight_scale"
    )
    assert not has_channel_scale_2 or group_size_n is None

    weight = weight.cuda()
    origin_ndim = weight.ndim
    weight = weight.unsqueeze(0) if weight.ndim == 2 else weight
    origin_dtype = dtypes.DataType.from_torch_dtype(weight.dtype)
    e, n, k = weight.shape
    group_size = group_size if group_size > 0 else k

    if group_size_n is not None:
        assert n % group_size_n == 0
        weight = weight.view(e, n // group_size_n, group_size_n, k // group_size, group_size)
        weight = weight.permute(0, 1, 3, 2, 4).contiguous()
        weight = weight.view(e, n * k // group_size_n // group_size, -1)
        group_size = group_size_n * group_size

    quant_group_size = 0
    if scale_dtype is not None:
        quant_group_size = group_size
    elif has_tensor_scale:
        quant_group_size = weight.nelement() // e
    flatten_weight = weight.view(e, 1, -1)
    use_flatten_weight = scale_dtype is None and has_tensor_scale
    weight_scale: torch.Tensor | None
    quanted_weight, weight_scale, zero_point = ops.quant_weight(
        flatten_weight if use_flatten_weight else weight,
        source_dtype_str=str(origin_dtype),
        target_dtype_str=str(dtype),
        group_size=quant_group_size,
        use_e8m0_scale=scale_dtype == dtypes.float8e8m0,
        has_scale=scale_dtype is not None or has_tensor_scale,
        has_zero_point=has_zero_point,
        is_fp_zero_point=is_fp_zero_point,
        allow_negative_scale=allow_negative_scale,
    )

    if zero_point.dtype == torch.float32:
        torch_dtype = torch.float16 if scale_dtype == dtypes.float16 else torch.bfloat16
        zero_point = zero_point.to(torch_dtype)

    weight_scale_2 = None
    if has_channel_scale_2:
        assert weight_scale is not None
        ws_f32 = weight_scale.float()
        if scale_dtype == dtypes.float8e8m0:
            weight_scale_2 = ws_f32.log2().mean(-1).exp2()
        elif scale_dtype in [dtypes.float8e4m3, dtypes.float8e5m2]:
            max_value = 448 if scale_dtype == dtypes.float8e4m3 else 57344
            weight_scale_2 = torch.maximum(
                ws_f32.abs().amax(-1) / max_value,
                ws_f32.abs().mean(-1),
            )
        else:
            weight_scale_2 = ws_f32.abs().mean(-1)
        weight_scale = (ws_f32 / weight_scale_2.unsqueeze(-1)).to(weight_scale.dtype)

    tensor_scale = None
    if scale_dtype is None and has_tensor_scale:
        tensor_scale = weight_scale.view(-1)
        weight_scale = None
        quanted_weight = quanted_weight.view(e, n, k)
    elif has_tensor_scale and scale_dtype == dtypes.float8e8m0:
        tensor_scale = weight_scale.float().view(e, -1).log2().mean(1).exp2()
        weight_scale = (weight_scale.float() / tensor_scale.view(e, 1, 1)).to(torch.float8_e8m0fnu)
    elif scale_dtype in [dtypes.float16, dtypes.bfloat16]:
        if has_tensor_scale:
            tensor_scale = weight_scale.view(e, -1).abs().mean(1)
            weight_scale_view = weight_scale.view(e, -1)
            weight_scale_view = weight_scale_view / tensor_scale.unsqueeze(1)
            weight_scale = weight_scale_view.view(weight_scale.shape)
        torch_dtype = torch.float16 if scale_dtype == dtypes.float16 else torch.bfloat16
        weight_scale = weight_scale.to(torch_dtype)
    elif scale_dtype in [dtypes.float8e4m3, dtypes.float8e5m2]:
        max_value = 448 if scale_dtype == dtypes.float8e4m3 else 57344
        torch_dtype = torch.float8_e4m3fn if scale_dtype == dtypes.float8e4m3 else torch.float8_e5m2
        if has_tensor_scale:
            tensor_scale1 = weight_scale.view(e, -1).max(1)[0] / max_value
            tensor_scale2 = weight_scale.view(e, -1).abs().mean(1)
            use_scale1 = (tensor_scale1 > tensor_scale2).any()
            tensor_scale = tensor_scale1 if use_scale1 else tensor_scale2
            weight_scale = weight_scale / tensor_scale.view(-1, 1, 1)
        weight_scale = weight_scale.to(torch_dtype)

    if group_size_n is not None:
        group_size = group_size // group_size_n
        quanted_weight = quanted_weight.view(
            e,
            n // group_size_n,
            k // group_size,
            group_size_n,
            group_size,
        )
        quanted_weight = quanted_weight.permute(0, 1, 3, 2, 4).contiguous()
        quanted_weight = quanted_weight.view(e, n, k)
        assert weight_scale is not None
        weight_scale = weight_scale.view(e, n // group_size_n, k // group_size)

    if origin_ndim == 2:
        quanted_weight = quanted_weight.squeeze(0)
        if weight_scale is not None and weight_scale.nelement() > 0:
            weight_scale = weight_scale.squeeze(0)
        if zero_point is not None and zero_point.nelement() > 0:
            zero_point = zero_point.squeeze(0)
        if tensor_scale is not None and tensor_scale.nelement() > 0:
            tensor_scale = tensor_scale.squeeze(0)
        if weight_scale_2 is not None:
            weight_scale_2 = weight_scale_2.squeeze(0)

    if pack:
        quanted_weight = ops.pack_weight(quanted_weight, dtype.num_bits)
        if has_zero_point and not is_fp_zero_point:
            zero_point = zero_point.transpose(-1, -2).contiguous()
            zero_point = zero_point.view(*zero_point.shape)
            zero_point = ops.pack_weight(zero_point, dtype.num_bits)
            zero_point = zero_point.transpose(-1, -2).contiguous()
            zero_point = zero_point.view(*zero_point.shape)

    final_zero_point = zero_point if zero_point.nelement() > 0 else None
    final_scale_2 = weight_scale_2 if weight_scale_2 is not None else tensor_scale

    return quanted_weight, weight_scale, final_zero_point, final_scale_2


def dequantize_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    zero_point: torch.Tensor | None,
    weight_scale_2: torch.Tensor | None,
    dtype: dtypes.DataType,
    packed: bool = False,
) -> torch.Tensor:
    assert weight.dtype == torch.int32
    weight = weight.cuda()

    if packed:
        weight = ops.unpack_weight(weight, dtype.num_bits)
        if zero_point is not None and zero_point.dtype == torch.int32:
            zero_point = zero_point.transpose(-1, -2).contiguous().cuda()
            zero_point = zero_point.view(*zero_point.shape)
            zero_point = ops.unpack_weight(zero_point, dtype.num_bits)
            zero_point = zero_point.transpose(-1, -2).contiguous()
            zero_point = zero_point.view(*zero_point.shape).float()

    if isinstance(dtype, dtypes.FloatingPointType):
        weight = ops.dequant_weight(weight, dtype.exponent_bits, dtype.mantissa_bits, True)
    else:
        assert isinstance(dtype, dtypes.IntegerType)
        assert not dtype.is_signed
        weight = weight.float()

    if zero_point is not None:
        assert weight.size(-1) % zero_point.size(-1) == 0
        group_size = weight.size(-1) // zero_point.size(-1)
        zero_point = zero_point.repeat_interleave(group_size, -1)
        weight = weight - zero_point
    elif isinstance(dtype, dtypes.IntegerType):
        assert not dtype.is_signed
        weight = weight - (1 << (dtype.num_bits - 1))

    if weight_scale is not None:
        assert weight.size(-1) % weight_scale.size(-1) == 0
        group_size = weight.size(-1) // weight_scale.size(-1)
        weight_scale = weight_scale.float()
        weight_scale = weight_scale.repeat_interleave(group_size, -1)
        if weight_scale.size(-2) != weight.size(-2):
            assert weight.size(-2) % weight_scale.size(-2) == 0
            group_size_n = weight.size(-2) // weight_scale.size(-2)
            weight_scale = weight_scale.repeat_interleave(group_size_n, -2)
        weight = weight * weight_scale

    if weight_scale_2 is not None:
        ws2 = weight_scale_2.float()
        num_experts = weight.size(0) if weight.ndim == 3 else 1
        if ws2.nelement() == num_experts:
            ws2 = ws2.view(-1, 1, 1)
            if weight.ndim == 2:
                ws2 = ws2.squeeze(0)
        else:
            ws2 = ws2.reshape(*weight.shape[:-1], 1)
        weight = weight * ws2

    return weight


def prepare_humming_weight(
    weight: torch.Tensor,
    b_dtype: dtypes.DataType,
    a_dtype: dtypes.DataType,
    zero_point: torch.Tensor | None = None,
    use_wgmma: bool = False,
    use_fused_e8m0_scale: bool = False,
    packed: bool = False,
    padded_shape_n: int | None = None,
    padded_shape_k: int | None = None,
    interleave_mode: int = 3,
    use_packed_k_layout: bool = False,
) -> torch.Tensor:
    is_moe = weight.ndim == 3
    weight = weight.unsqueeze(0) if not is_moe else weight
    if zero_point is not None:
        zero_point = zero_point.unsqueeze(0) if zero_point.ndim == 2 else zero_point
    shape_n = weight.size(-2)
    if packed:
        assert weight.size(-1) * 32 % b_dtype.num_bits == 0
        shape_k = weight.size(-1) * 32 // b_dtype.num_bits
    else:
        shape_k = weight.size(-1)

    padded_shape_n = shape_n if padded_shape_n is None else padded_shape_n
    padded_shape_k = shape_k if padded_shape_k is None else padded_shape_k
    packed_block_size_k = 256 // a_dtype.num_bits

    assert padded_shape_n % 64 == 0
    assert padded_shape_k % (2 * packed_block_size_k) == 0

    should_preprocess_for_int2fp = False
    has_zero_point = zero_point is not None and zero_point.nelement() > 0
    if b_dtype.is_integer_type and a_dtype.is_floating_point_type:
        if a_dtype.num_bits < 16:
            should_preprocess_for_int2fp = True
        elif a_dtype == dtypes.bfloat16 and has_zero_point:
            should_preprocess_for_int2fp = b_dtype.num_bits > 6
        elif a_dtype == dtypes.bfloat16 and not has_zero_point:
            should_preprocess_for_int2fp = b_dtype.num_bits > 7

    if a_dtype == dtypes.int8 and b_dtype in [dtypes.int8, dtypes.uint8]:
        weight = (weight.view(torch.int8) - 128).view(torch.int32)

    if a_dtype == dtypes.int4 and b_dtype in [dtypes.int4, dtypes.uint4]:
        weight = weight.view(torch.uint8)
        weight1 = (weight & 0xF) - 8
        weight1 = weight1 & 0xF
        weight2 = (weight & 0xF0) - 8 * 16
        weight = (weight1 | weight2).view(torch.int32)

    if not should_preprocess_for_int2fp and has_zero_point:
        has_zero_point = False

    should_preprocess_with_zp = has_zero_point
    if zero_point is not None and zero_point.dtype.is_floating_point:
        should_preprocess_with_zp = False
        should_preprocess_for_int2fp = False

    if not has_zero_point:
        group_size_zp = 0
    else:
        assert zero_point is not None
        group_size_zp = shape_k // zero_point.size(-1)

    if use_packed_k_layout:
        assert use_wgmma, "use_packed_k_layout requires wgmma"
        assert a_dtype.num_bits == 8, "use_packed_k_layout requires 8-bit (fp8/int8) activation"
        assert not use_fused_e8m0_scale, "use_packed_k_layout is incompatible with fused-e8m0 scale"

    repacked_weight = ops.repack_weight(
        inputs=weight,
        zero_point=zero_point,
        weight_bits=b_dtype.num_bits,
        activation_bits=a_dtype.num_bits,
        is_weight_packed=packed,
        should_preprocess_for_int2fp=should_preprocess_for_int2fp,
        should_preprocess_with_zp=should_preprocess_with_zp,
        use_wgmma=use_wgmma,
        interleave_mode=interleave_mode,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
        group_size_zp=group_size_zp,
        use_packed_k_layout=use_packed_k_layout,
    )

    return repacked_weight if is_moe else repacked_weight.squeeze(0)


def prepare_humming_weight_scale(
    weight_scale: torch.Tensor,
    to_apply_on_c: bool = False,
    is_blockwise: bool = False,
    is_mxmma: bool = False,
    mxmma_scale_vec: int = 4,
) -> torch.Tensor:
    if is_blockwise:
        return weight_scale.transpose(-1, -2).contiguous()

    if is_mxmma:
        if mxmma_scale_vec == 1:
            ws = weight_scale.view(torch.uint8)
            lead = ws.shape[:-2]
            n, g = ws.shape[-2:]
            ws = ws.reshape(*lead, n // 16, 2, 8, g // 2, 2)
            ndim = len(lead)
            perm = (*range(ndim), ndim + 3, ndim + 0, ndim + 2, ndim + 1, ndim + 4)
            ws = ws.permute(*perm).contiguous()
            ws = ws.reshape(*lead, g // 2, n // 2, 4)
            return ws.view(torch.int32).squeeze(-1).contiguous()

        return weight_scale.view(torch.int32).transpose(-1, -2).contiguous()

    if to_apply_on_c:
        perm = [0, 1, 8, 9, 16, 17, 24, 25]
    else:
        perm = [0, 8, 16, 24, 32, 40, 48, 56]

    count = sum(x < 8 for x in perm)
    perm_new = []
    for i in range(8 // count):
        perm_new += [x + count * i for x in perm]

    perm_tensor = torch.tensor(perm_new, dtype=torch.int32, device=weight_scale.device)
    weight_scale = weight_scale.transpose(-1, -2).contiguous()
    orig_shape = weight_scale.shape
    weight_scale = weight_scale.view(-1, len(perm_tensor))[:, perm_tensor]
    weight_scale = weight_scale.contiguous().view(orig_shape)

    return weight_scale


def prepare_humming_zero_point(
    zero_point: torch.Tensor,
    dtype: dtypes.DataType,
    packed: bool = False,
) -> torch.Tensor | None:
    num_experts = None if zero_point.ndim == 2 else zero_point.size(0)
    if zero_point.dtype.is_floating_point:
        return prepare_humming_weight_scale(zero_point, False)

    if packed:
        zero_point = zero_point.transpose(-1, -2).contiguous()
        zero_point = zero_point.squeeze().view(*zero_point.shape)
        zero_point = ops.unpack_weight(zero_point, dtype.num_bits)
        zero_point = zero_point.transpose(-1, -2).contiguous()

    assert zero_point is not None
    num_zp_bits = 4 if dtype.num_bits <= 4 else 8
    shape_n = zero_point.size(-2)
    zero_point = prepare_humming_weight_scale(zero_point)
    assert zero_point is not None
    zero_point = zero_point.to(torch.uint8)
    zero_point = zero_point.view(-1)
    if num_zp_bits == 4:
        zero_point = zero_point[..., 1::2] * 16 + zero_point[..., ::2]
    zero_point = zero_point.view(torch.int32).view(-1, shape_n * num_zp_bits // 32)
    if num_experts is not None:
        zero_point = zero_point.view(num_experts, -1, zero_point.size(-1))
    return zero_point


def prepare_humming_bias(bias: torch.Tensor) -> torch.Tensor:
    bias = prepare_humming_weight_scale(bias.unsqueeze(-1), True)
    assert bias is not None
    return bias.squeeze(-2)
