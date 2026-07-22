import random

import pytest

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


@pytest.mark.parametrize("a_dtype", ["bfloat16", "float8e4m3", "int8", "int4"])
@pytest.mark.parametrize("b_dtype", ["uint3"])
@pytest.mark.parametrize("input_scale_group_size", [64, 0])
@pytest.mark.parametrize("weight_scale_group_size", [64, 0])
@pytest.mark.parametrize("mma_type", ["mma", "wgmma"])
def test_batch_invariance(
    a_dtype,
    b_dtype,
    input_scale_group_size,
    weight_scale_group_size,
    mma_type,
):
    skip_if_unsupported(a_dtype=a_dtype, mma_type=mma_type)
    c_dtype = dtypes.bfloat16
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.DataType.from_str(b_dtype)
    bs_dtype = dtypes.bfloat16
    if b_dtype.num_bits >= a_dtype.num_bits:
        return

    if a_dtype.num_bits == 16 and mma_type == "mma":
        shapes = [
            ((16, 128, 64), (16, 64, 64)),
            ((64, 512, 64), (64, 64, 64)),
            ((64, 128, 64), (64, 64, 64)),
            ((64, 64, 64), (32, 64, 64)),
            ((16, 128, 32), (16, 64, 32)),
            ((48, 256, 32), (48, 64, 32)),
            ((64, 64, 32), (32, 64, 32)),
        ]
    elif a_dtype.num_bits == 16 and mma_type == "wgmma":
        shapes = [
            ((8, 256, 64), (8, 64, 64)),
            ((16, 256, 64), (16, 64, 64)),
            ((64, 512, 64), (64, 64, 64)),
            ((8, 256, 32), (8, 64, 32)),
            ((16, 256, 32), (16, 64, 32)),
            ((16, 512, 32), (8, 64, 32)),
            ((48, 256, 32), (48, 64, 32)),
        ]
    elif a_dtype.num_bits == 8:
        shapes = [
            ((8, 256, 64), (8, 32, 64)),
            ((16, 256, 64), (16, 32, 64)),
            ((16, 256, 64), (8, 32, 64)),
            ((64, 256, 64), (64, 32, 64)),
            ((48, 128, 64), (24, 64, 64)),
            ((64, 128, 64), (64, 64, 64)),
            ((48, 256, 64), (48, 64, 64)),
            ((48, 128, 64), (48, 32, 64)),
            ((64, 64, 64), (32, 32, 64)),
        ]
        if mma_type == "wgmma":
            shapes = [(x, y) for x, y in shapes if x[1] // y[1] >= 4]
        if mma_type == "mma":
            shapes = [(x, y) for x, y in shapes if y[0] % 16 == 0]
    elif a_dtype.num_bits == 4:
        if mma_type == "wgmma":
            return
        shapes = [
            ((16, 256, 128), (16, 64, 128)),
            ((64, 256, 128), (64, 64, 128)),
            ((64, 256, 128), (64, 16, 128)),
            ((64, 128, 256), (64, 32, 256)),
            ((48, 512, 128), (48, 64, 128)),
            ((48, 128, 128), (48, 64, 128)),
            ((64, 64, 128), (32, 16, 128)),
        ]

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
    weight = transform_humming_weight(weight, b_dtype, a_dtype, use_wgmma=mma_type == "wgmma")

    if mma_type == "mma":
        to_apply_on_c = weight_scale_group_size == 0 or a_dtype.num_bits != 16
    elif mma_type == "wgmma":
        to_apply_on_c = weight_scale_group_size == 0

    weight_scale = transform_humming_weight_scale(weight_scale, to_apply_on_c=to_apply_on_c)

    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=1234,
        k=1024,
        group_size=input_scale_group_size,
        dtype=a_dtype,
    )

    old_outputs = None

    for block_shape, warp_shape in shapes:
        humming_kernel = HummingKernel(
            shape_n=1024,
            shape_k=1024,
            block_shape=block_shape,
            warp_shape=warp_shape,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            bs_dtype=bs_dtype,
            input_scale_group_size=input_scale_group_size,
            weight_scale_group_size=weight_scale_group_size,
            num_stages=3,
            use_warp_spec=False,
            use_tma=False,
            use_cp_async=False,
            has_bias=False,
            mma_type=mma_type,
            use_stream_k=False,
        )

        for seed in range(10):
            new_shape_m = inputs.size(0)
            if seed > 1:
                random.seed(seed)
                new_shape_m = int(new_shape_m * random.random())
                new_shape_m = max(new_shape_m, 1)

            print(new_shape_m, block_shape, warp_shape)

            outputs_new = ops.launch_kernel(
                configs=[humming_kernel.kernel_id],
                inputs=inputs[:new_shape_m],
                weight=weight,
                outputs=None,
                input_scale=None if input_scale is None else input_scale[:new_shape_m],
                weight_scale=weight_scale,
            )

            if old_outputs is not None:
                assert (outputs_new == old_outputs[:new_shape_m]).all()
            else:
                old_outputs = outputs_new
