import pytest
import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.utils.test import _current_sm_version, generate_random_weight
from humming.utils.weight import prepare_humming_weight, prepare_humming_weight_scale


def _skip_if_no_mxmma():
    sm = _current_sm_version()
    if sm // 10 != 12:
        pytest.skip(f"mxmma requires SM12x (Blackwell), current SM is {sm}")


def _dequant_act(xq, xs, a_dtype, group_size):
    scale = xs.float().repeat_interleave(group_size, dim=1)
    if a_dtype == dtypes.float4e2m1:
        x_un = ops.unpack_weight(xq.view(torch.int32), 4)
        x_fp = ops.dequant_weight(x_un, 2, 1, True)
        return x_fp.float() * scale
    return xq.float() * scale


def _run_mxmma(
    a_dtype,
    b_dtype,
    c_dtype,
    bs_dtype,
    group_size,
    m=256,
    n=1024,
    k=1024,
):
    # microscale block-scaled mma: scale_vec = (mma K-tile) / scale group.
    part_k = 256 // a_dtype.num_bits
    assert part_k % group_size == 0
    scale_vec = part_k // group_size

    # mxmma block-scale formats (ue8m0 / ue4m3) are unsigned, so the weight must
    # be quantized with a non-negative microscale.
    _, weight_ref, weight, weight_scale, _, _ = generate_random_weight(
        n=n, k=k, group_size=group_size, dtype=b_dtype, scale_dtype=bs_dtype,
        allow_negative_scale=False,
    )
    weight_ref = weight_ref.squeeze(0) if weight_ref.ndim == 3 else weight_ref
    weight_k = prepare_humming_weight(weight, b_dtype, a_dtype)
    weight_scale_k = prepare_humming_weight_scale(
        weight_scale, is_mxmma=True, mxmma_scale_vec=scale_vec
    )

    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.5
    a_name = "float4e2m1" if a_dtype == dtypes.float4e2m1 else "float8e4m3"
    xq, xs = ops.quant_input(
        x, a_name, group_size=group_size, scale_dtype=bs_dtype.to_str()
    )
    x_dq = _dequant_act(xq, xs, a_dtype, group_size)
    xs_packed = xs.view(torch.int32).contiguous()

    block_shape = (128, 128, part_k * 2)
    kernel = HummingKernel(
        shape_n=n,
        shape_k=k,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
        input_scale_group_size=group_size,
        weight_scale_group_size=group_size,
        weight_scale_type="group",
        mma_type="mxmma",
        block_shape=block_shape,
        warp_shape=(64, 64, part_k * 2),
        num_stages=2,
        use_tma=False,
        use_cp_async=True,
        use_warp_spec=True,
        use_stream_k=False,
    )

    torch_dtype = dtypes.torch_dtype_map[c_dtype]
    out = torch.zeros((m, n), dtype=torch_dtype, device="cuda")
    out = ops.launch_kernel(
        configs=[kernel.kernel_id],
        inputs=xq,
        weight=weight_k,
        outputs=out,
        input_scale=xs_packed,
        weight_scale=weight_scale_k,
    )
    torch.cuda.synchronize()

    ref = x_dq @ weight_ref.float().T
    assert torch.isfinite(out.float()).all(), "kernel produced non-finite output"
    torch.testing.assert_close(out.float(), ref, rtol=0.08, atol=0.6)


@pytest.mark.parametrize(
    "b_dtype",
    [
        "float8e4m3",
        "float6e3m2",
        "float7e3m3",
        "float6e2m3",
        "float4e2m1",
        "float3e1m1",
        "uint5",
        "uint4",
        "uint3",
        "uint2",
    ],
)
@pytest.mark.parametrize("c_dtype", ["bfloat16", "float16"])
def test_mxmma_fp8(b_dtype, c_dtype):
    _skip_if_no_mxmma()
    _run_mxmma(
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.DataType.from_str(b_dtype),
        c_dtype=dtypes.DataType.from_str(c_dtype),
        bs_dtype=dtypes.float8e8m0,
        group_size=32,
    )


@pytest.mark.parametrize(
    "bs_dtype,group_size",
    [
        ("float8e8m0", 32),
        ("float8e8m0", 16),
        ("float8e4m3", 16),
    ],
)
@pytest.mark.parametrize("c_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "b_dtype",
    [
        "float4e2m1",
        "float3e1m1",
        "uint3",
        "uint2",
    ],
)
def test_mxmma_fp4(b_dtype, bs_dtype, group_size, c_dtype):
    _skip_if_no_mxmma()
    _run_mxmma(
        a_dtype=dtypes.float4e2m1,
        b_dtype=dtypes.DataType.from_str(b_dtype),
        c_dtype=dtypes.DataType.from_str(c_dtype),
        bs_dtype=dtypes.DataType.from_str(bs_dtype),
        group_size=group_size,
    )
