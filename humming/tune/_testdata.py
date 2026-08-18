"""Random quantized-tensor generators for the tune measurement worker.

Ported from the pre-refactor humming.utils.test generators: the tune worker
needs the unpacked weight + reference pair to drive the golden-output gate,
which the KernelTestRunner fixtures do not expose.
"""

import torch

from humming import dtypes, ops
from humming.utils.weight import quantize_weight


def generate_random_inputs(
    m: int,
    k: int,
    group_size: int = 0,
    dtype: dtypes.DataType = dtypes.float32,
):
    group_size = group_size if group_size > 0 else k
    input_scale: torch.Tensor | None = torch.randn(
        (m, k // group_size),
        dtype=torch.float32,
        device="cuda:0",
    )
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
            has_zero_point=False,
        )

        inputs_ref = inputs.float()
        if isinstance(dtype, dtypes.FloatingPointType):
            inputs_ref = ops.dequant_weight(
                inputs,
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

        assert input_scale is not None
        inputs_ref = inputs_ref * input_scale.repeat_interleave(group_size, 1)

    return inputs_orig, inputs_ref, inputs, input_scale


def generate_random_weight(
    n,
    k,
    group_size,
    dtype,
    scale_dtype,
    group_size_n=None,
    num_experts=None,
    has_weight_scale_2=False,
    has_zero_point=False,
    is_fp_zero_point=False,
    allow_negative_scale=True,
):
    e = 1 if num_experts is None else num_experts
    dtype_orig = dtype
    group_size = group_size if group_size > 0 else k
    if has_zero_point:
        assert dtype.is_integer_type and not dtype.is_signed, (
            "dynamic zero point only supports for uint dtype"
        )

    if dtype.is_integer_type and dtype.is_signed:
        dtype = dtypes.IntegerType(is_signed=False, num_bits=dtype.num_bits)

    weight_orig = torch.randn((e, n, k), dtype=torch.float32, device="cuda:0")
    init_weight_scale = torch.rand((e, n, k // group_size), dtype=torch.float32, device="cuda:0")
    init_weight_scale = init_weight_scale + 0.01
    init_weight_bias = torch.randn((e, n, k // group_size), dtype=torch.float32, device="cuda:0")

    weight_orig = weight_orig + init_weight_bias.repeat_interleave(group_size, -1)
    weight_orig = weight_orig * init_weight_scale.repeat_interleave(group_size, -1)
    weight_orig = weight_orig / weight_orig.std()

    quanted_weight, weight_scale, zero_point, weight_scale_2 = quantize_weight(
        weight_orig,
        dtype=dtype,
        scale_dtype=scale_dtype,
        group_size=group_size,
        group_size_n=group_size_n,
        has_zero_point=has_zero_point,
        weight_scale_2_type="tensor" if has_weight_scale_2 else None,
        is_fp_zero_point=is_fp_zero_point,
        allow_negative_scale=allow_negative_scale,
    )

    if dtype.is_integer_type and has_zero_point:
        assert zero_point is not None
        weight_ref = quanted_weight.to(zero_point.dtype)
        weight_ref = weight_ref - zero_point.repeat_interleave(group_size, -1)
        weight_ref = weight_ref.float()
    elif dtype.is_integer_type and not has_zero_point:
        weight_ref = quanted_weight.float() - 2 ** (dtype.num_bits - 1)
    else:
        weight_ref = ops.dequant_weight(
            quanted_weight,
            exponent_bits=dtype.exponent_bits,
            mantissa_bits=dtype.mantissa_bits,
            is_signed=dtype.is_signed,
        )

    if weight_scale is not None:
        weight_scale_tmp = weight_scale.float().repeat_interleave(group_size, -1)
        if group_size_n is not None:
            weight_scale_tmp = weight_scale_tmp.repeat_interleave(group_size_n, -2)
        weight_ref = weight_ref * weight_scale_tmp

    if has_weight_scale_2:
        weight_ref = weight_ref * weight_scale_2.view(-1, 1, 1)

    if dtype_orig.is_integer_type and dtype_orig.is_signed:
        quanted_weight = quanted_weight - 2 ** (dtype.num_bits - 1)

    if num_experts is None:
        weight_orig = weight_orig.squeeze(0)
        weight_ref = weight_ref.squeeze(0)
        quanted_weight = quanted_weight.squeeze(0)
        if weight_scale is not None:
            weight_scale = weight_scale.squeeze(0)
        if weight_scale_2 is not None:
            weight_scale_2 = weight_scale_2.squeeze(0)
        if zero_point is not None:
            zero_point = zero_point.squeeze(0)

    return weight_orig, weight_ref, quanted_weight, weight_scale, zero_point, weight_scale_2
