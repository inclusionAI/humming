import math

import pytest
import torch

from humming import ops
from humming.ops.input import hadamard_transform

SCALE_DTYPES = {
    "float32": torch.float32,
    "float8e4m3": torch.float8_e4m3fn,
    "float8e8m0": torch.float8_e8m0fnu,
}


def _empty_group_scales(
    rows: int,
    groups: int,
    scale_dtype: str,
    scale_layout: str = "row_major",
) -> torch.Tensor:
    stride = (rows + 3) // 4 * 4
    if scale_layout == "row_major":
        shape = (rows, groups)
    elif scale_layout == "m_major":
        shape = (groups, stride)
    else:
        shape = ((groups + 3) // 4, stride, 4)
    return torch.empty(shape, device="cuda", dtype=SCALE_DTYPES[scale_dtype])


def _source_after_transform(x: torch.Tensor, block_size: int | None) -> torch.Tensor:
    return x if block_size is None else hadamard_transform(x, block_size)


def _quantize_int8_reference(
    x: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    grouped = x.float().reshape(-1, x.size(-1) // group_size, group_size)
    grouped = grouped / scales.reshape(-1, x.size(-1) // group_size, 1)
    return torch.round(grouped).clamp(-128, 127).to(torch.int8).reshape(x.shape)


def _hadamard_reference(inputs: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size == 1:
        return inputs.float()

    values = inputs.float().reshape(-1, block_size)
    width = 1
    while width < block_size:
        pairs = values.reshape(-1, block_size // (2 * width), 2, width)
        low = pairs[:, :, 0]
        high = pairs[:, :, 1]
        values = torch.stack((low + high, low - high), dim=2).reshape(-1, block_size)
        width *= 2
    return (values * (1.0 / math.sqrt(block_size))).reshape(inputs.shape)


def _unpack_int4(values: torch.Tensor) -> torch.Tensor:
    low = (values & 0xF).to(torch.int8)
    high = ((values >> 4) & 0xF).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.empty(
        values.shape[:-1] + (values.size(-1) * 2,),
        dtype=torch.int8,
        device=values.device,
    )
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high
    return unpacked


def _decode_quantized(values: torch.Tensor, quant_dtype: str) -> torch.Tensor:
    if quant_dtype == "int4":
        return _unpack_int4(values).float()
    if quant_dtype == "float8e3m4":
        codes = values.view(torch.uint8).to(torch.int32).contiguous()
        return ops.dequant_weight(codes, 3, 4, True)
    if quant_dtype == "float4e0m3":
        codes = ops.unpack_weight(values.view(torch.int32), 4)
        magnitude = (codes & 0x7).float()
        return torch.where((codes & 0x8) != 0, -magnitude, magnitude)
    if quant_dtype == "float4e2m1":
        codes = ops.unpack_weight(values.view(torch.int32), 4)
        return ops.dequant_weight(codes, 2, 1, True)
    return values.float()


def _require_quant_capability(quant_dtype: str | None) -> None:
    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    if quant_dtype in ("float8e4m3", "float8e5m2") and capability < 89:
        pytest.skip(f"{quant_dtype} requires SM89+")
    if quant_dtype in ("float8e3m4", "float4e2m1", "float4e0m3") and capability < 100:
        pytest.skip(f"{quant_dtype} requires SM100+")


def _make_activated_input(dtype, rows: int, hidden_size: int, activation: str):
    a = torch.randn((rows, hidden_size), device="cuda", dtype=dtype) * 0.5
    if activation == "none":
        return a, a.float(), None
    if activation == "unary":
        return a, torch.relu(a.float()), "a > 0.f ? a : 0.f"

    b = torch.randn_like(a) * 0.5
    activated = a.float() * a.float() + b.float()
    if activation == "binary_split":
        inputs = torch.cat((a, b), dim=-1)
    else:
        inputs = torch.stack((a, b), dim=-1).flatten(-2)
    return inputs, activated, "a * a + b"


def _assert_quantized(result, reference, quant_dtype: str, group_size: int) -> None:
    assert result[0].shape == reference[0].shape
    assert result[0].dtype == reference[0].dtype
    assert result[1] is not None and reference[1] is not None
    torch.testing.assert_close(result[1], reference[1], rtol=1e-5, atol=1e-7)

    actual = _decode_quantized(result[0], quant_dtype)
    expected = _decode_quantized(reference[0], quant_dtype)
    if quant_dtype in ("int8", "int4"):
        difference = (actual.to(torch.int32) - expected.to(torch.int32)).abs()
        bad = torch.count_nonzero(difference > 1).item()
        assert bad / difference.numel() < 1e-3
        return

    scales = reference[1].repeat_interleave(group_size, dim=-1)
    dequantized_actual = actual * scales
    dequantized_expected = expected * scales
    difference = (dequantized_actual - dequantized_expected).abs()
    denominator = dequantized_expected.abs().clamp_min(scales.abs() * 0.5)
    relative = difference / denominator
    bad = torch.count_nonzero(relative > 0.30).item()
    assert bad / relative.numel() < 0.02, (
        f"{bad}/{relative.numel()} elements differ by more than one quantization step; "
        f"maximum relative difference is {relative.max().item()}"
    )


def _round_positive_m3_rne(value: torch.Tensor) -> torch.Tensor:
    bits = value.contiguous().view(torch.int32)
    retained_lsb = (bits >> 20) & 1
    rounded = (bits + 0x0007FFFF + retained_lsb) & -0x00100000
    return rounded.view(torch.float32)
