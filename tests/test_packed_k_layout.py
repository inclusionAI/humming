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


@pytest.mark.parametrize("a_dtype", ["float8e4m3", "int8"])
@pytest.mark.parametrize("b_dtype", ["uint4", "uint8", "int8", "float8e4m3", "float4e1m2"])
@pytest.mark.parametrize("c_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("bs_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("warp_k", [64, 128])
@pytest.mark.parametrize("warp_n", [32, 64])
@pytest.mark.parametrize("input_scale_group_size", [0, 64, 128])
@pytest.mark.parametrize("weight_scale_group_size", [0, 64, 128])
@pytest.mark.parametrize("use_tma", [False, True])
def test_packed_k_layout(
    a_dtype,
    b_dtype,
    c_dtype,
    bs_dtype,
    warp_k,
    warp_n,
    input_scale_group_size,
    weight_scale_group_size,
    use_tma,
):
    # packed-K layout (ITER_N) is a wgmma + 8-bit-activation feature.
    mma_type = "wgmma"
    skip_if_unsupported(a_dtype=a_dtype, mma_type=mma_type)
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.DataType.from_str(b_dtype)
    c_dtype = dtypes.DataType.from_str(c_dtype)
    bs_dtype = dtypes.DataType.from_str(bs_dtype)

    # weight/activation dtype compatibility (mirrors test_scale).
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

    if bs_dtype in [dtypes.float16, dtypes.bfloat16] and bs_dtype != c_dtype:
        return

    # packed-K constraints: each scale group is 0 (channel) or a multiple of
    # 256//a_bits AND >= warp_shape_k; if both group scales are set they must match.
    for gs in (input_scale_group_size, weight_scale_group_size):
        if gs > 0 and (gs < 256 // a_dtype.num_bits or gs < warp_k):
            return
    if input_scale_group_size > 0 and weight_scale_group_size > 0:
        if input_scale_group_size != weight_scale_group_size:
            return

    # 4 N-warps -> one full wgmma warpgroup (128 threads).
    block_shape = (16, 4 * warp_n, warp_k)
    warp_shape = (16, warp_n, warp_k)

    _, weight_ref, weight, weight_scale, _, _ = generate_random_weight(
        n=1024,
        k=1024,
        group_size=weight_scale_group_size,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
    )
    if a_dtype == dtypes.int8 and b_dtype == dtypes.int8:
        weight = (weight.view(torch.int8) + 128).view(torch.int32)

    to_apply_on_c = weight_scale_group_size == 0

    weight = transform_humming_weight(weight, b_dtype, a_dtype, use_wgmma=True, use_packed_k_layout=True)
    weight_scale = transform_humming_weight_scale(weight_scale, to_apply_on_c=to_apply_on_c)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=128,
        k=1024,
        group_size=input_scale_group_size,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=block_shape,
        warp_shape=warp_shape,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_stages=3,
        use_warp_spec=use_tma,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        weight_scale_type="group" if weight_scale_group_size else "channel",
        use_tma=use_tma,
        use_cp_async=not use_tma,
        mma_type=mma_type,
        use_stream_k=False,
        use_packed_k_layout=True,
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

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)
    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.5)
