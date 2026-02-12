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


@pytest.mark.parametrize(
    "block_shape, warp_shape, a_dtype",
    [
        # 16bit activation
        [(16, 256, 32), (16, 64, 32), dtypes.bfloat16],
        [(32, 128, 64), (16, 64, 32), dtypes.bfloat16],
        [(32, 128, 64), (32, 64, 64), dtypes.bfloat16],
        [(48, 512, 64), (48, 64, 32), dtypes.bfloat16],
        [(64, 64, 64), (32, 64, 32), dtypes.bfloat16],
        [(128, 64, 128), (32, 64, 32), dtypes.bfloat16],
        [(96, 256, 128), (48, 64, 64), dtypes.bfloat16],
        # 8bit activation
        [(16, 256, 128), (16, 64, 64), dtypes.int8],
        [(32, 256, 64), (16, 32, 64), dtypes.int8],
        [(64, 128, 128), (64, 32, 128), dtypes.int8],
        [(48, 512, 64), (48, 64, 64), dtypes.int8],
        [(64, 64, 64), (32, 32, 64), dtypes.int8],
        [(128, 64, 128), (32, 32, 64), dtypes.int8],
        [(128, 64, 256), (32, 64, 64), dtypes.int8],
        [(96, 256, 256), (48, 64, 128), dtypes.int8],
        # 4bit activation
        [(16, 256, 512), (16, 64, 256), dtypes.int4],
        [(32, 256, 256), (16, 32, 256), dtypes.int4],
        [(64, 128, 128), (64, 32, 128), dtypes.int4],
        [(48, 512, 256), (48, 64, 128), dtypes.int4],
        [(64, 64, 128), (32, 32, 128), dtypes.int4],
        [(128, 64, 128), (32, 32, 128), dtypes.int4],
        [(128, 64, 256), (32, 16, 128), dtypes.int4],
        [(96, 128, 128), (48, 16, 128), dtypes.int4],
    ],
)
@pytest.mark.parametrize("b_dtype", ["uint3"])
def test_moe(block_shape, warp_shape, a_dtype, b_dtype):
    c_dtype = dtypes.bfloat16
    b_dtype = dtypes.DataType.from_str(b_dtype)
    bs_dtype = dtypes.bfloat16
    if b_dtype.num_bits >= a_dtype.num_bits:
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
        m=234,
        k=1024,
        group_size=0,
        dtype=a_dtype,
    )

    humming_kernel = HummingKernel(
        problem_shape=(0, 1024, 1024),
        block_shape=block_shape,
        warp_shape=warp_shape,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        num_stages=2,
        use_warp_spec=False,
        use_tma=False,
        use_cp_async=False,
        has_bias=False,
        mma_type="mma",
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.empty((234, 1024), dtype=torch_dtype, device=inputs.device)

    outputs = humming_kernel(
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T)
    outputs_ref = outputs_ref.to(torch_dtype)

    torch.testing.assert_close(outputs, outputs_ref, rtol=0.05, atol=0.1)
