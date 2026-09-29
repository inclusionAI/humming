import pytest
import torch

from humming.config import ActivationType
from humming.ops.input import process_input
from humming.testing.process_input import (
    ACTIVATION_TYPE_IMPL_TEST_MAP,
    assert_process_input_close,
    process_input_ref,
    skip_if_process_input_unsupported,
)


@pytest.mark.parametrize("quant_dtype", ["float8e3m4", "float4e0m3", "float4e2m1"])
@pytest.mark.parametrize(
    "capability,supported_dtypes",
    [
        ((8, 9), ()),
        ((10, 0), ("float8e3m4", "float4e0m3", "float4e2m1")),
        ((10, 3), ("float8e3m4", "float4e0m3", "float4e2m1")),
        ((12, 0), ("float8e3m4", "float4e0m3", "float4e2m1")),
        ((12, 1), ("float8e3m4", "float4e0m3", "float4e2m1")),
    ],
)
def test_quantization_support_matches_cubin_patcher(monkeypatch, quant_dtype, capability, supported_dtypes):
    """Patched conversion formats are available on SM100/103 and SM120/121."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)

    if quant_dtype in supported_dtypes:
        skip_if_process_input_unsupported(quant_dtype)
    else:
        with pytest.raises(pytest.skip.Exception, match="requires SM"):
            skip_if_process_input_unsupported(quant_dtype)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape_m,hidden_size,quant_group_size,hadamard_block_size,activation_type,quant_dtype",
    [
        (1, 384, 64, 1, "none", None),
        (3, 768, 128, 32, "unary", None),
        (4, 2048, 512, 128, "binary_split", None),
        (5, 7168, 128, 512, "binary_interleaved", None),
        (2, 11008, 128, 1, "unary", "int8"),
        (129, 2048, 64, 128, "binary_interleaved", "float8e4m3"),
        (3, 512, 128, 16, "binary_split", "int8"),
        (3, 512, 128, 256, "binary_interleaved", "float8e4m3"),
        (4, 768, 128, 128, "none", "int8"),
        (4, 768, 128, 128, "none", "int4"),
        (4, 768, 128, 128, "none", "float8e4m3"),
        (4, 768, 128, 128, "none", "float8e5m2"),
        (4, 768, 128, 128, "none", "float8e3m4"),
        (4, 768, 128, 128, "none", "float4e2m1"),
        (4, 768, 128, 128, "none", "float4e0m3"),
    ],
)
def test_transform_quantization(
    input_dtype,
    shape_m,
    hidden_size,
    quant_group_size,
    hadamard_block_size,
    activation_type,
    quant_dtype,
):
    skip_if_process_input_unsupported(quant_dtype)
    torch.manual_seed(1)

    activation_type = ActivationType(activation_type)
    input_width = hidden_size * (2 if activation_type.is_binary else 1)
    activation_impl = ACTIVATION_TYPE_IMPL_TEST_MAP[activation_type]["impl"]
    inputs = 0.5 * torch.randn(shape_m, input_width, device="cuda", dtype=input_dtype)
    quant_mode = "none" if quant_dtype is None else "dynamic_group"

    options = dict(
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        quant_group_size=quant_group_size,
        activation_type=activation_type.value,
        hadamard_block_size=hadamard_block_size,
    )

    expected = process_input_ref(inputs, **options)
    actual = process_input(inputs, activation_impl=activation_impl, **options)

    assert_process_input_close(
        actual,
        expected,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        quant_group_size=quant_group_size,
    )
