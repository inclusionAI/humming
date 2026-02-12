import pytest
from humming import dtypes
import torch
from humming.utils.test import (
    generate_random_weight,
    generate_random_inputs,
)
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
)
from humming.kernel.humming import HummingKernel


@pytest.mark.parametrize("a_dtype", ["float16", "bfloat16", "float8e4m3", "int8", "int4"])
@pytest.mark.parametrize(
    "b_dtype",
    [
        "uint3",
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
@pytest.mark.parametrize("input_scale_group_size", [8, 16, 32, 64, 256, 0])
@pytest.mark.parametrize("weight_scale_group_size", [8, 16, 32, 64, 512, 0])
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

    if mma_type == "wgmma" and a_dtype == "int4":
        return
    if weight_scale_group_size > 0:
        if mma_type == "mma" and weight_scale_group_size < 128 // a_dtype.num_bits:
            return
        if mma_type == "wgmma" and weight_scale_group_size < 256 // a_dtype.num_bits:
            return
    if input_scale_group_size > 0:
        if a_dtype.num_bits == 16:
            return
        if mma_type == "mma" and input_scale_group_size < 128 // a_dtype.num_bits:
            return
        if mma_type == "wgmma" and input_scale_group_size < 256 // a_dtype.num_bits:
            return

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=weight_scale_group_size,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
    )

    _, weight_ref, weight, weight_scale, _, _ = random_weight_data

    if mma_type == "mma":
        to_apply_on_c = weight_scale_group_size == 0 or a_dtype.num_bits != 16
    elif mma_type == "wgmma":
        to_apply_on_c = weight_scale_group_size == 0

    weight = prepare_humming_weight(weight, b_dtype, a_dtype)
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=to_apply_on_c)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=128,
        k=1024,
        group_size=input_scale_group_size,
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
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        use_tma=False,
        use_cp_async=False,
        mma_type=mma_type,
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

    if a_dtype.num_bits == 16 and weight_scale.size(-2) > 1:
        weight_ref = weight_ref.to(torch_dtype).float()

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    try:
        torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.1)
    except BaseException:
        torch.save(
            (inputs, inputs_ref, weight_scale, weight_ref, outputs, outputs_ref), "/root/aa.pt"
        )
        raise
