import math

import pytest
import torch

from humming.ops.hadamard import hadamard_transform


def _hadamard_matrix(n: int, device, dtype=torch.float32) -> torch.Tensor:
    assert n > 0 and (n & (n - 1)) == 0
    h = torch.tensor([[1.0]], device=device, dtype=dtype)
    while h.size(0) < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
    return h


def _reference(x: torch.Tensor, n: int, scale: float) -> torch.Tensor:
    h = _hadamard_matrix(n, x.device, torch.float32)
    norm = scale / math.sqrt(n)
    orig_shape = x.shape
    x2 = x.to(torch.float32).reshape(-1, n)
    y = (x2 @ h) * norm
    return y.reshape(orig_shape).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block_size", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
def test_hadamard_correctness(dtype, block_size):
    torch.manual_seed(0)
    # Choose at least 8 tiles, last dim a multiple of block_size.
    last_dim = max(block_size * 4, 256)
    if last_dim % block_size != 0:
        last_dim = block_size * (last_dim // block_size)
    x = torch.randn((3, last_dim), device="cuda", dtype=dtype) * 0.5

    y = hadamard_transform(x, block_size=block_size)
    ref = _reference(x, block_size, scale=1.0)

    if dtype == torch.float32:
        tol = dict(rtol=1e-5, atol=1e-5)
    elif dtype == torch.float16:
        tol = dict(rtol=5e-3, atol=5e-3)
    else:  # bfloat16
        tol = dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(y, ref, **tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_hadamard_scale(dtype):
    torch.manual_seed(0)
    block_size = 128
    x = torch.randn((2, 8, block_size * 3), device="cuda", dtype=dtype) * 0.3
    scale = 2.5
    y = hadamard_transform(x, block_size=block_size, scale=scale)
    ref = _reference(x, block_size, scale=scale)
    tol = dict(rtol=5e-3, atol=5e-3) if dtype == torch.float16 else dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y, ref, **tol)


@pytest.mark.parametrize("block_size", [16, 256, 1024])
def test_hadamard_involution(block_size):
    """Applying H twice with scale = sqrt(N) recovers x (since H @ H = N * I)."""
    torch.manual_seed(0)
    x = torch.randn((4, block_size * 2), device="cuda", dtype=torch.float32)
    y = hadamard_transform(x, block_size=block_size)
    z = hadamard_transform(y, block_size=block_size)
    torch.testing.assert_close(z, x, rtol=1e-5, atol=1e-5)
