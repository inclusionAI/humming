import pytest
from humming import dtypes
import torch
from humming.utils.test import generate_random_weight, generate_random_inputs
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
)
from humming.kernel.humming import HummingKernel


@pytest.mark.parametrize("a_dtype", ["float16", "bfloat16", "float8e4m3", "int8", "int4"])
@pytest.mark.parametrize(
    "b_dtype",
    [
        "uint1",
        "uint2",
        "uint3",
        "uint4",
        "uint5",
        "uint6",
        "uint7",
        "uint8",
        "int4",
        "int8",
        "float3e1m1",
        "float3e2m0",
        "float4e1m2",
        "float4e2m1",
        "float4e3m0",
        "float5e2m2",
        "float5e3m1",
        "float5e4m0",
        "float6e2m3",
        "float6e3m2",
        "float6e4m1",
        "float7e2m4",
        "float7e4m2",
        "float7e6m0",
        "float8e1m6",
        "float8e2m5",
        "float8e4m3",
        # "float8e5m2",
    ],
)
@pytest.mark.parametrize("c_dtype", ["float16", "bfloat16"])
def test_datatype(a_dtype, b_dtype, c_dtype):
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
        k=1024,
        group_size=0,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        problem_shape=(0, 1024, 1024),
        block_shape=(16, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(16, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
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
    outputs = torch.zeros((128, 1024), dtype=torch_dtype, device=inputs.device)

    outputs = humming_kernel(
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.03, atol=0.1)


@pytest.mark.parametrize(
    "a_dtype, b_dtype",
    [
        ["float16", "uint3"],
        ["float16", "uint4"],
        ["float16", "uint5"],
        ["float16", "uint8"],
        ["bfloat16", "uint3"],
        ["bfloat16", "uint4"],
        ["bfloat16", "uint5"],
        ["bfloat16", "uint8"],
        ["float8e4m3", "uint3"],
        ["float8e4m3", "uint4"],
        ["int8", "uint4"],
        ["int8", "uint5"],
        ["int4", "uint1"],
        ["int4", "uint2"],
    ],
)
@pytest.mark.parametrize("c_dtype", ["float16", "bfloat16"])
def test_zeropoint(a_dtype, b_dtype, c_dtype):
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.DataType.from_str(b_dtype)
    c_dtype = dtypes.DataType.from_str(c_dtype)
    bs_dtype = c_dtype

    if a_dtype in [dtypes.float16, dtypes.bfloat16] and a_dtype != c_dtype:
        return

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=0,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
        has_dynamic_zp=True,
    )

    _, weight_ref, weight, weight_scale, zero_point, _ = random_weight_data

    weight = prepare_humming_weight(weight, b_dtype, a_dtype, zero_point)
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=True)
    zero_point = prepare_humming_zero_point(zero_point, b_dtype)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=128,
        k=1024,
        group_size=0,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        problem_shape=(0, 1024, 1024),
        block_shape=(16, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(16, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_stages=3,
        use_warp_spec=False,
        use_tma=False,
        use_cp_async=False,
        has_dynamic_zero_point=True,
        mma_type="mma",
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((128, 1024), dtype=torch_dtype, device=inputs.device)

    outputs = humming_kernel(
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        zero_point=zero_point,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.03, atol=0.1)
