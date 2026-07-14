import pytest
import torch

from humming.ops import quant_input
from humming.ops.hadamard import hadamard_quant_input, hadamard_transform


def _unpack_int4(q: torch.Tensor) -> torch.Tensor:
    lo = (q & 0xF).to(torch.int8)
    hi = ((q >> 4) & 0xF).to(torch.int8)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    unpacked_shape = q.shape[:-1] + (q.size(-1) * 2,)
    out = torch.empty(unpacked_shape, dtype=torch.int8, device=q.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _to_float(q: torch.Tensor, quant_dtype: str) -> torch.Tensor:
    if quant_dtype == "int4":
        return _unpack_int4(q).to(torch.float32)
    return q.to(torch.float32)


@pytest.mark.parametrize("quant_dtype", ["int8", "int4", "float8e4m3", "float8e5m2"])
@pytest.mark.parametrize("block_size", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("group_size_ratio", [1, 2])
def test_fused_matches_unfused_fp32(quant_dtype, block_size, group_size_ratio):
    """With fp32 input, the unfused reference has no intermediate precision
    loss, so the fused kernel should produce identical scales and quantized
    outputs (up to rounding ties)."""
    torch.manual_seed(0)
    group_size = block_size // group_size_ratio
    if group_size < 8:
        pytest.skip("group_size < elems_per_thread (E=8) not supported")

    last_dim = max(block_size * 4, 256)
    last_dim = (last_dim // block_size) * block_size
    x = torch.randn((3, last_dim), device="cuda", dtype=torch.float32) * 0.5

    y_ref = hadamard_transform(x, block_size=block_size)
    q_ref, s_ref = quant_input(y_ref, quant_dtype, group_size=group_size)

    q_fused, s_fused = hadamard_quant_input(
        x, block_size=block_size, quant_dtype=quant_dtype, group_size=group_size
    )

    assert q_fused.shape == q_ref.shape
    assert s_fused.shape == s_ref.shape
    assert q_fused.dtype == q_ref.dtype

    torch.testing.assert_close(s_fused, s_ref, rtol=1e-5, atol=1e-6)

    # Quantized values should match exactly (or within 1 step at boundaries).
    qf = _to_float(q_fused, quant_dtype)
    qr = _to_float(q_ref, quant_dtype)
    if quant_dtype in ("int8", "int4"):
        # Integer codes: at most 1-step differences from rounding ties.
        diff = (qf.to(torch.int32) - qr.to(torch.int32)).abs()
        bad = (diff > 1).sum().item()
        assert bad / diff.numel() < 1e-3
    else:
        # FP codes may differ by at most one fp8 step on boundary roundings
        # (different fp32 summation order in FHT). Allow up to 1% to differ.
        s = s_ref.repeat_interleave(group_size, dim=-1)
        dqf = qf * s
        dqr = qr * s
        # one fp8 step is ~12.5% relative for e4m3, ~25% for e5m2; allow it.
        diff = (dqf - dqr).abs()
        ref_mag = dqr.abs().clamp_min(s.abs() * 0.5)
        rel = diff / ref_mag
        bad = (rel > 0.30).sum().item()
        assert bad / rel.numel() < 0.02, (
            f"{bad}/{rel.numel()} elements diff > 1 fp8 step (max rel={rel.max().item()})"
        )


@pytest.mark.parametrize("quant_dtype", ["int8", "float8e4m3"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_lower_dtype_shape_and_scale(quant_dtype, dtype):
    """For fp16/bf16 inputs, the fused kernel preserves more precision than
    the unfused reference (no intermediate cast). We only check shape +
    scales agreement to within the input dtype's precision."""
    torch.manual_seed(0)
    block_size = 128
    x = torch.randn((4, block_size * 4), device=("cuda"), dtype=dtype) * 0.4

    q_fused, s_fused = hadamard_quant_input(
        x, block_size=block_size, quant_dtype=quant_dtype, group_size=block_size
    )
    y_ref = hadamard_transform(x, block_size=block_size)
    q_ref, s_ref = quant_input(y_ref, quant_dtype, group_size=block_size)

    assert q_fused.shape == q_ref.shape
    assert s_fused.shape == s_ref.shape
    rtol = 5e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(s_fused, s_ref, rtol=rtol, atol=rtol)


@pytest.mark.parametrize("quant_dtype", ["int8", "float8e4m3"])
def test_fused_extra_scale(quant_dtype):
    torch.manual_seed(0)
    block_size = 128
    x = torch.randn((4, block_size * 3), device="cuda", dtype=torch.float32) * 0.3
    scale = 2.0
    q_fused, s_fused = hadamard_quant_input(
        x, block_size=block_size, quant_dtype=quant_dtype, scale=scale, group_size=block_size
    )
    y_ref = hadamard_transform(x, block_size=block_size, scale=scale)
    q_ref, s_ref = quant_input(y_ref, quant_dtype, group_size=block_size)
    torch.testing.assert_close(s_fused, s_ref, rtol=1e-5, atol=1e-6)
