import pytest
import torch

from humming.ops.input import process_input

from ._reference import (
    _assert_quantized,
    _hadamard_reference,
    _make_activated_input,
    _require_quant_capability,
)

SOURCE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
ROW_COUNTS = (1, 129)
HADAMARD_BLOCK_SIZES = (1, 32, 128, 512)
GROUP_SIZES = (64, 512)
ACTIVATIONS = ("none", "unary", "binary_split", "binary_interleaved")
MATRIX_QUANT_DTYPES = (None, "int8", "int4", "float8e4m3", "float4e2m1")
ALL_QUANT_DTYPES = (
    "int8",
    "int4",
    "float8e4m3",
    "float8e3m4",
    "float8e5m2",
    "float4e2m1",
    "float4e0m3",
)


@pytest.mark.parametrize("dtype", SOURCE_DTYPES)
@pytest.mark.parametrize("rows", ROW_COUNTS)
@pytest.mark.parametrize("block_size", HADAMARD_BLOCK_SIZES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("quant_dtype", MATRIX_QUANT_DTYPES)
def test_process_input_cartesian(dtype, rows, block_size, group_size, activation, quant_dtype):
    """Cross every core transform/quantization parameter and schedule regime."""
    _require_quant_capability(quant_dtype)
    torch.manual_seed(1)
    hidden_size = 2048
    inputs, activated, activation_impl = _make_activated_input(dtype, rows, hidden_size, activation)
    transformed = _hadamard_reference(activated, block_size)
    hadamard_block_size = block_size if block_size > 1 else None
    activation_type = activation
    activation_args = {"activation_type": activation_type, "activation_impl": activation_impl}

    if quant_dtype is None:
        result = process_input(
            inputs,
            quant_group_size=group_size,
            hadamard_block_size=hadamard_block_size,
            **activation_args,
        )
        tolerance = {
            torch.float16: dict(rtol=5e-3, atol=5e-3),
            torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
            torch.float32: dict(rtol=1e-5, atol=1e-5),
        }[dtype]
        torch.testing.assert_close(result[0], transformed.to(dtype), **tolerance)
        assert result[1] is None and result[2] is None
        return

    result = process_input(
        inputs,
        quant_mode="dynamic_group",
        quant_dtype=quant_dtype,
        quant_group_size=group_size,
        hadamard_block_size=hadamard_block_size,
        **activation_args,
    )
    reference = process_input(
        transformed,
        quant_mode="dynamic_group",
        quant_dtype=quant_dtype,
        quant_group_size=group_size,
    )
    _assert_quantized(result, reference, quant_dtype, group_size)


@pytest.mark.parametrize("dtype", SOURCE_DTYPES)
@pytest.mark.parametrize("quant_dtype", ALL_QUANT_DTYPES)
def test_quant_dtype_codecs(dtype, quant_dtype):
    """Cover every output codec without multiplying codecs into the large matrix."""
    _require_quant_capability(quant_dtype)
    torch.manual_seed(2)
    group_size = 128
    inputs = torch.randn((3, 512), device="cuda", dtype=dtype) * 0.5
    transformed = _hadamard_reference(inputs, group_size)
    result = process_input(
        inputs,
        quant_mode="dynamic_group",
        quant_dtype=quant_dtype,
        quant_group_size=group_size,
        hadamard_block_size=group_size,
    )
    reference = process_input(
        transformed,
        quant_mode="dynamic_group",
        quant_dtype=quant_dtype,
        quant_group_size=group_size,
    )
    _assert_quantized(result, reference, quant_dtype, group_size)
