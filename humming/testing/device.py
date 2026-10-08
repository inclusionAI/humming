import torch

from humming import dtypes
from humming.config.mma import supports_undocumented_fp_dtypes, uses_undocumented_fp_operand
from humming.device import current_device

_A_DTYPE_MIN_SM = {
    dtypes.int4: 80,
    dtypes.int8: 75,
    dtypes.float4e0m3: 120,
    dtypes.float4e2m1: 120,
    dtypes.float8e3m4: 120,
    dtypes.float8e4m3: 89,
    dtypes.float8e5m2: 89,
    dtypes.bfloat16: 80,
    dtypes.float16: 75,
}


def skip_if_unsupported(
    a_dtype=None,
    b_dtype=None,
    mma_type=None,
    use_tma=None,
    use_warp_spec=None,
) -> None:
    """Skip a test whose hardware requirements aren't met by the current GPU."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    sm = current_device.sm_version

    if mma_type == "wgmma" and sm != 90:
        pytest.skip(f"wgmma requires SM90, current SM is {sm}")
    if mma_type == "mxmma" and sm // 10 != 12:
        pytest.skip(f"mxmma requires SM12x, current SM is {sm}")

    a_dtype = a_dtype and dtypes.DataType.from_any(a_dtype)
    if mma_type == "wgmma" and a_dtype == dtypes.int4:
        pytest.skip("wgmma does not support int4 activation")

    if a_dtype is not None and a_dtype in _A_DTYPE_MIN_SM:
        min_sm = _A_DTYPE_MIN_SM[a_dtype]
        if mma_type == "umma" and a_dtype in (dtypes.float4e2m1, dtypes.float4e0m3, dtypes.float8e3m4):
            min_sm = 100
        if sm < min_sm:
            pytest.skip(f"a_dtype {a_dtype} requires SM>={min_sm}, current SM is {sm}")

    b_dtype = b_dtype and dtypes.DataType.from_any(b_dtype)
    if a_dtype is not None and uses_undocumented_fp_operand(a_dtype, b_dtype):
        if not supports_undocumented_fp_dtypes(sm):
            pytest.skip(f"SM{sm} does not support E3M4/E0M3 operands")

    if current_device.is_ppu and a_dtype == dtypes.int4:
        pytest.skip("PPU does not support int4 mma")

    if use_tma and sm < 90:
        pytest.skip(f"TMA requires SM>=90, current SM is {sm}")

    if use_warp_spec and sm < 90:
        pytest.skip(f"warp specialization requires SM>=90, current SM is {sm}")
