import pytest
import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.transform import (
    transform_humming_weight,
    transform_humming_weight_scale,
)
from humming.utils.test import (
    generate_random_inputs,
    generate_random_weight,
    skip_if_unsupported,
)


@pytest.mark.parametrize("a_dtype", ["float16", "float8e4m3", "int8"])
@pytest.mark.parametrize("b_dtype", ["uint4", "uint8", "float4e1m2"])
@pytest.mark.parametrize("c_dtype", ["float16"])
@pytest.mark.parametrize("bs_dtype", ["float16", "float8e4m3"])
@pytest.mark.parametrize("input_scale_group_size", [64, 0])
@pytest.mark.parametrize("weight_scale_group_size", [64, 0])
@pytest.mark.parametrize("mma_type", ["mma", "wgmma"])
@pytest.mark.parametrize("shape_m", [123, 64, 950])
@pytest.mark.parametrize("cast_type", ["a", "b"])
def test_multi_cast(
    a_dtype,
    b_dtype,
    c_dtype,
    bs_dtype,
    input_scale_group_size,
    weight_scale_group_size,
    mma_type,
    shape_m,
    cast_type,
):
    skip_if_unsupported(
        a_dtype=a_dtype,
        mma_type=mma_type,
        use_cp_async=True,
        use_tma=True,
        use_warp_spec=True,
        use_mbarrier=True,
    )
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

    if input_scale_group_size > 0 and a_dtype.num_bits == 16:
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

    weight = transform_humming_weight(weight, b_dtype, a_dtype, use_wgmma=mma_type == "wgmma")
    weight_scale = transform_humming_weight_scale(
        weight_scale,
        to_apply_on_c=to_apply_on_c,
    )

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=shape_m,
        k=1024,
        group_size=input_scale_group_size,
        dtype=a_dtype,
    )

    locks = torch.zeros((1024,), dtype=torch.int32, device=inputs.device)
    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=(128, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(128, a_dtype.num_bits * 4, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_stages=3,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        use_cp_async=True,
        mma_type=mma_type,
        use_mbarrier=True,
        use_tma=True,
        use_warp_spec=True,
        use_stream_k=True,
        multi_cast_size_a=2 if cast_type == "a" else 1,
        multi_cast_size_b=2 if cast_type == "b" else 1,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((shape_m, 1024), dtype=torch_dtype, device=inputs.device)

    outputs = ops.launch_kernel(
        configs=[humming_kernel.kernel_id],
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        locks=locks,
    )

    if a_dtype.num_bits == 16 and weight_scale.size(-2) > 1:
        weight_ref = weight_ref.to(torch_dtype).float()

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.2)
