import pytest
from humming import dtypes
import torch
from humming.utils.test import generate_random_weight, generate_random_inputs, generate_random_bias
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_bias,
)
from humming.kernel.humming import HummingKernel


@pytest.mark.parametrize("a_dtype", ["float16", "bfloat16", "float8e4m3", "int8", "int4"])
@pytest.mark.parametrize("c_dtype", ["float16", "bfloat16"])
def test_bias(a_dtype, c_dtype):
    c_dtype = dtypes.DataType.from_str(c_dtype)
    a_dtype = dtypes.DataType.from_str(a_dtype)
    b_dtype = dtypes.uint3
    bs_dtype = c_dtype

    if a_dtype.num_bits == 16 and a_dtype != c_dtype:
        return

    random_weight_data = generate_random_weight(
        n=1024,
        k=1024,
        group_size=0,
        dtype=b_dtype,
        scale_dtype=bs_dtype,
    )

    _, weight_ref, weight, weight_scale, _, _ = random_weight_data
    bias_ref = generate_random_bias(1024, c_dtype) * 8

    weight = prepare_humming_weight(weight, b_dtype, a_dtype)
    weight_scale = prepare_humming_weight_scale(weight_scale, to_apply_on_c=True)
    bias = prepare_humming_bias(bias_ref)

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
        has_bias=True,
        mma_type="mma",
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((128, 1024), dtype=torch_dtype, device=inputs.device)

    outputs = humming_kernel(
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        bias=bias,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T) + bias_ref.float().view(1, -1)
    outputs_ref = outputs_ref.to(torch_dtype)

    torch.testing.assert_close(outputs, outputs_ref, rtol=0.01, atol=0.1)


@pytest.mark.parametrize(
    "activation",
    [
        "sigmoid",
        "tanh",
        "relu",
        "gelu",
        "fastgelu",
        "quickgelu",
        "silu",
        "custom",
        "silu_glu",
        "custom_glu",
    ],
)
def test_activation(activation):
    c_dtype = dtypes.float16
    a_dtype = c_dtype
    b_dtype = dtypes.uint4
    bs_dtype = c_dtype

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

    custom_activation_func_impl = None
    if activation == "custom":
        custom_activation_func_impl = "return tanhf(a) * 2.33 + 0.666;"
    elif activation == "custom_glu":
        custom_activation_func_impl = "return (fabsf(a.x) - fabsf(a.y)) * 0.123;"

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
        use_stream_k=True,
        activation_type=activation,
        custom_activation_func_impl=custom_activation_func_impl,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    outputs = torch.zeros((128, 1024), dtype=torch_dtype, device=inputs.device)

    locks = torch.zeros((1024,), dtype=torch.int32, device=inputs.device)
    outputs = humming_kernel(
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        locks=locks,
    )

    outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch.float16).float()
    if activation == "sigmoid":
        outputs_ref = torch.sigmoid(outputs_ref)
    elif activation == "tanh":
        outputs_ref = torch.tanh(outputs_ref)
    elif activation == "relu":
        outputs_ref = torch.relu(outputs_ref)
    elif activation == "gelu":
        outputs_ref = torch.nn.GELU()(outputs_ref)
    elif activation == "fastgelu":
        outputs_ref = torch.nn.GELU("tanh")(outputs_ref)
    elif activation == "quickgelu":
        outputs_ref = outputs_ref * torch.sigmoid(1.702 * outputs_ref)
    elif activation == "silu":
        outputs_ref = torch.nn.SiLU()(outputs_ref)
    elif activation == "custom":
        outputs_ref = outputs_ref.tanh() * 2.33 + 0.666
    elif activation == "silu_glu":
        outputs_ref = outputs_ref[:, ::2] * torch.nn.SiLU()(outputs_ref[:, 1::2])
    elif activation == "custom_glu":
        outputs_ref = outputs_ref.abs()
        outputs_ref = (outputs_ref[:, ::2] - outputs_ref[:, 1::2]) * 0.123

    outputs_ref = outputs_ref.to(torch.float16)

    torch.testing.assert_close(outputs, outputs_ref, rtol=0.1, atol=0.1)
