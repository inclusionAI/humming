import pytest
from humming import dtypes
import torch
from humming.utils.test import generate_random_weight, generate_random_inputs
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
)
from humming.kernel.humming import HummingKernel


@pytest.mark.parametrize("a_dtype", ["float16", "bfloat16", "float8e4m3", "int8", "int4"])
@pytest.mark.parametrize("b_dtype", ["uint3"])
@pytest.mark.parametrize("c_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("a_swizzle_bytes", [64, 128])
def test_datatype(a_dtype, b_dtype, c_dtype, a_swizzle_bytes):
    assert a_swizzle_bytes in [64, 128]
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.DataType.from_str(b_dtype)
    c_dtype = dtypes.DataType.from_str(c_dtype)
    bs_dtype = c_dtype

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

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=0,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
    )

    _, weight_ref, weight, weight_scale, _, _ = random_weight_data

    weight = prepare_humming_weight(weight, b_dtype, a_dtype)
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=True)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=128,
        k=1024 - 32,
        group_size=0,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        problem_shape=(0, 1024, 1024),
        block_shape=(
            16,
            a_dtype.num_bits * 16,
            a_swizzle_bytes * 8 // a_dtype.num_bits,
        ),
        warp_shape=(16, a_dtype.num_bits * 4, a_swizzle_bytes * 8 // a_dtype.num_bits),
        pad_shape=(0, 8, 32),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_stages=3,
        use_warp_spec=False,
        use_tma=False,
        use_cp_async=False,
        mma_type="mma",
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((128, 1024 - 8), dtype=torch_dtype, device=inputs.device)

    outputs = humming_kernel(
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    outputs_ref = inputs_ref.matmul(weight_ref[:-8, :-32].contiguous().T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.03, atol=0.1)
