import torch

from humming import dtypes
from humming.kernel.repack_weight import WeightRepackKernel
from humming.kernel.quant_weight import QuantWeightKernel
from humming.kernel.unpack_weight import UnpackWeightKernel


def quantize_weight(
    weight, dtype, scale_dtype, group_size, has_dynamic_zp=False, has_global_scale=False
):
    assert weight.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert weight.ndim in [2, 3]
    assert not has_dynamic_zp or scale_dtype is not None

    origin_ndim = weight.ndim
    weight = weight.unsqueeze(0) if weight.ndim == 2 else weight

    origin_dtype = dtypes.DataType.from_torch_dtype(weight.dtype)
    e, n, k = weight.shape
    group_size = group_size if group_size > 0 else k
    device = weight.device

    quanted_weight = torch.empty_like(weight, dtype=torch.int32)
    scale_shape = (e, n, k // group_size)
    if scale_dtype == dtypes.float8e8m0 and not has_global_scale:
        weight_scale = torch.empty(scale_shape, dtype=torch.float8_e8m0fnu, device=device)
    elif scale_dtype is not None:
        weight_scale = torch.empty(scale_shape, dtype=torch.float32, device=device)
    elif has_global_scale:
        weight_scale = torch.empty((e, 1, 1), dtype=torch.float32, device=device)
    else:
        weight_scale = None

    if has_dynamic_zp:
        assert scale_dtype is not None
        zero_point = torch.empty(scale_shape, dtype=torch.int32, device=device)
    else:
        zero_point = None

    quant_group_size = 0
    if weight_scale is not None:
        quant_group_size = weight.nelement() // weight_scale.nelement()

    quant_kernel = QuantWeightKernel(
        source_dtype=origin_dtype,
        target_dtype=dtype,
        group_size=quant_group_size,
        use_e8m0_scale=weight_scale.dtype == torch.float8_e8m0fnu,
        has_scale=weight_scale is not None,
        has_dynamic_zero_point=has_dynamic_zp,
    )

    if scale_dtype is None and has_global_scale:
        quant_kernel(weight.view(e, 1, -1), quanted_weight.view(e, 1, -1), weight_scale, zero_point)
    else:
        quant_kernel(weight, quanted_weight, weight_scale, zero_point)

    global_scale = None
    if scale_dtype is None and has_global_scale:
        global_scale = weight_scale.view(-1)
        weight_scale = None
    elif has_global_scale and scale_dtype == dtypes.float8e8m0:
        assert (weight_scale > 0).all()
        global_scale = weight_scale.view(e, -1).log2().mean(1).exp2()
        weight_scale = (weight_scale / global_scale.view(e, 1, 1)).to(torch.float8_e8m0fnu)
    elif scale_dtype in [dtypes.float16, dtypes.bfloat16]:
        if has_global_scale:
            global_scale = weight_scale.view(e, -1).abs().mean(1)
            weight_scale_view = weight_scale.view(e, -1)
            weight_scale_view = weight_scale_view / global_scale.unsqueeze(1)
            weight_scale = weight_scale_view.view(weight_scale.shape)
        torch_dtype = torch.float16 if scale_dtype == dtypes.float16 else torch.bfloat16
        weight_scale = weight_scale.to(torch_dtype)
    elif scale_dtype in [dtypes.float8e4m3, dtypes.float8e5m2]:
        max_value = 448 if scale_dtype == dtypes.float8e4m3 else 57344
        torch_dtype = torch.float8_e4m3fn if scale_dtype == dtypes.float8e4m3 else torch.float8_e5m2
        if has_global_scale:
            global_scale1 = weight_scale.view(e, -1).max(1)[0] / max_value
            global_scale2 = weight_scale.view(e, -1).abs().mean(1)
            use_scale1 = (global_scale1 > global_scale2).any()
            global_scale = global_scale1 if use_scale1 else global_scale2
            weight_scale = weight_scale / global_scale.view(-1, 1, 1)
        weight_scale = weight_scale.to(torch_dtype)

    if origin_ndim == 2:
        quanted_weight = quanted_weight.squeeze(0)
        if weight_scale is not None:
            weight_scale = weight_scale.squeeze(0)
        if zero_point is not None:
            zero_point = zero_point.squeeze(0)
        if global_scale is not None:
            global_scale = global_scale.squeeze(0)

    return quanted_weight, weight_scale, zero_point, global_scale


def prepare_humming_weight(
    weight,
    b_dtype,
    a_dtype,
    zero_point=None,
    packed=False,
    padded_shape_n=None,
    padded_shape_k=None,
):
    is_moe = weight.ndim == 3
    weight = weight.unsqueeze(0) if not is_moe else weight
    if zero_point is not None:
        zero_point = zero_point.unsqueeze(0) if zero_point.ndim == 2 else zero_point
    num_experts = 1 if not is_moe else weight.size(0)
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
    has_dynamic_zero_point = zero_point is not None and zero_point.nelement()
    if b_dtype.is_integer_type and a_dtype.is_floating_point_type:
        if a_dtype.num_bits < 16:
            should_preprocess_for_int2fp = True
        elif a_dtype == dtypes.bfloat16 and has_dynamic_zero_point:
            should_preprocess_for_int2fp = b_dtype.num_bits > 6
        elif a_dtype == dtypes.bfloat16 and not has_dynamic_zero_point:
            should_preprocess_for_int2fp = b_dtype.num_bits > 7

    if not should_preprocess_for_int2fp and has_dynamic_zero_point:
        has_dynamic_zero_point = False

    group_size_zp = 0 if not has_dynamic_zero_point else (shape_k // zero_point.size(-1))
    kernel = WeightRepackKernel(
        weight_bits=b_dtype.num_bits,
        activation_bits=a_dtype.num_bits,
        is_weight_pakced=packed,
        should_preprocess_for_int2fp=should_preprocess_for_int2fp,
        should_preprocess_with_zp=has_dynamic_zero_point,
        group_size_zp=group_size_zp,
        sm_version=None,
    )

    shape_k_new = padded_shape_k // packed_block_size_k
    shape_n_new = padded_shape_n * packed_block_size_k * b_dtype.num_bits // 32

    repacked_weight = torch.empty(
        (num_experts, shape_k_new, shape_n_new), dtype=torch.int32, device=weight.device
    )
    kernel(
        inputs=weight,
        outputs=repacked_weight,
        zero_point=zero_point,
        padded_shape_n=padded_shape_n,
        padded_shape_k=padded_shape_k,
    )

    return repacked_weight if is_moe else repacked_weight.squeeze(0)


def prepare_humming_weight_scale(weight_scale, to_apply_on_c=False):
    if to_apply_on_c:
        perm = [0, 1, 8, 9, 16, 17, 24, 25]
    else:
        perm = [0, 8, 16, 24, 32, 40, 48, 56]

    count = sum(x < 8 for x in perm)
    perm_new = []
    for i in range(8 // count):
        perm_new += [x + count * i for x in perm]

    perm_new = torch.tensor(perm_new, dtype=torch.int32, device=weight_scale.device)
    weight_scale = weight_scale.transpose(-1, -2).contiguous()
    orig_shape = weight_scale.shape
    weight_scale = weight_scale.view(-1, len(perm_new))[:, perm_new]
    weight_scale = weight_scale.contiguous().view(orig_shape)

    return weight_scale


def prepare_humming_zero_point(zero_point, dtype, packed=False):
    if zero_point is None:
        return zero_point

    if packed:
        zero_point = zero_point.transpose(-1, -2).contiguous()
        zero_point = zero_point.squeeze().view(*zero_point.shape)
        kernel = UnpackWeightKernel(dtype.num_bits)
        zero_point = kernel(zero_point)
        zero_point = zero_point.transpose(-1, -2).contiguous()

    num_zp_bits = 4 if dtype.num_bits <= 4 else 8
    shape_n = zero_point.size(-2)
    zero_point = prepare_humming_weight_scale(zero_point).to(torch.uint8)
    zero_point = zero_point.view(-1)
    if num_zp_bits == 4:
        zero_point = zero_point[..., 1::2] * 16 + zero_point[..., ::2]
    return zero_point.view(torch.int32).view(-1, shape_n * num_zp_bits // 32)


def prepare_humming_bias(bias):
    if bias is None:
        return
    return prepare_humming_weight_scale(bias.unsqueeze(-1), True).squeeze(0)
