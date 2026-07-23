import pytest
import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.transform import (
    transform_humming_weight,
    transform_humming_weight_scale,
    transform_humming_zero_point,
)
from humming.utils.test import (
    generate_random_inputs,
    generate_random_weight,
    skip_if_unsupported,
)


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
@pytest.mark.parametrize("is_fp_zero_point", [True])
@pytest.mark.parametrize("warp_shape_n_splits", [2, 1])
def test_zeropoint(a_dtype, b_dtype, c_dtype, is_fp_zero_point, warp_shape_n_splits):
    skip_if_unsupported(a_dtype=a_dtype)
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.DataType.from_str(b_dtype)
    c_dtype = dtypes.DataType.from_str(c_dtype)
    bs_dtype = c_dtype

    if a_dtype in [dtypes.float16, dtypes.bfloat16] and a_dtype != c_dtype:
        return

    if a_dtype not in [dtypes.float16, dtypes.bfloat16] and is_fp_zero_point:
        return

    if warp_shape_n_splits == 2 and a_dtype.num_bits != 16:
        return

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=0,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
        has_zero_point=True,
        is_fp_zero_point=is_fp_zero_point,
    )

    _, weight_ref, weight, weight_scale, zero_point, _ = random_weight_data

    weight = transform_humming_weight(weight, b_dtype, a_dtype, zero_point)
    weight_scale = transform_humming_weight_scale(weight_scale, to_apply_on_c=True)
    zero_point = transform_humming_zero_point(zero_point, b_dtype)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=128,
        k=1024,
        group_size=0,
        dtype=a_dtype,
    )

    warp_shape_n = a_dtype.num_bits * 4 // warp_shape_n_splits
    if warp_shape_n < 16:
        return

    humming_kernel = HummingKernel(
        shape_n=1024,
        shape_k=1024,
        block_shape=(16, a_dtype.num_bits * 16, 512 // a_dtype.num_bits),
        warp_shape=(16, warp_shape_n, 512 // a_dtype.num_bits),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_stages=3,
        use_warp_spec=False,
        use_tma=False,
        use_cp_async=False,
        has_zero_point=True,
        is_fp_zero_point=is_fp_zero_point,
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
        zero_point=zero_point,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch_dtype)

    torch.testing.assert_close(outputs, outputs_ref, rtol=0.03, atol=0.1)
