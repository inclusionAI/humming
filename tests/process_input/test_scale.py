import pytest
import torch

from humming.ops.input import hadamard_quant_input, process_input

from ._reference import (
    _empty_group_scales,
    _quantize_int8_reference,
    _source_after_transform,
)


def test_combined_static_tensor_and_group_scales():
    x = torch.randn(3, 512, device="cuda", dtype=torch.float32)
    tensor_scale = torch.tensor([0.5], device="cuda")
    group_scale = torch.tensor([0.25, 0.5, 1.0, 2.0], device="cuda")
    result = process_input(
        x,
        quant_mode="static_tensor_group",
        quant_dtype="int8",
        quant_group_size=128,
        token_scales=tensor_scale,
        group_scales=group_scale,
    )
    scales = tensor_scale * group_scale
    expected = torch.round(x.reshape(3, 4, 128) / scales[None, :, None])
    expected = expected.clamp(-127, 127).to(torch.int8).reshape_as(x)
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("quant_dtype", ["float8e4m3", "int8"])
def test_normal_four_group_subwarp_schedule(quant_dtype):
    """Cover the four-groups-per-warp identity fast path."""
    torch.manual_seed(16)
    group_size = 128
    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)

    result = process_input(
        x,
        quant_mode="dynamic_group",
        quant_dtype=quant_dtype,
        quant_group_size=group_size,
    )
    grouped = x.float().reshape(32, 4, group_size)
    if quant_dtype == "int8":
        expected_scales = grouped.abs().amax(-1) / 127.0
        expected = _quantize_int8_reference(x, expected_scales, group_size)
    else:
        expected_scales = grouped.abs().amax(-1) / 448.0
        expected = (grouped / expected_scales[:, :, None]).to(torch.float8_e4m3fn)

    torch.testing.assert_close(result[1], expected_scales, rtol=0, atol=0)
    output_atol = 1 if quant_dtype == "int8" else 0
    torch.testing.assert_close(
        result[0].float().reshape_as(expected),
        expected.float(),
        rtol=0,
        atol=output_atol,
    )


@pytest.mark.parametrize("block_size", [None, 128])
@pytest.mark.parametrize("granularity", ["tensor", "group"])
def test_static_fp32_scale(block_size, granularity):
    torch.manual_seed(10)
    x = torch.randn(3, 512, device="cuda", dtype=torch.float32)
    group_size = 128
    if granularity == "tensor":
        static_scale = torch.tensor([0.025], device="cuda")
        scale_matrix = static_scale.expand(3, 4)
    else:
        static_scale = torch.tensor([0.01, 0.02, 0.04, 0.08], device="cuda")
        scale_matrix = static_scale.expand(3, 4)
    quant_mode = "static_tensor" if granularity == "tensor" else "static_group"
    static_args = (
        {"token_scales": static_scale} if granularity == "tensor" else {"group_scales": static_scale}
    )

    result = process_input(
        x,
        quant_mode=quant_mode,
        quant_dtype="int8",
        quant_group_size=group_size,
        hadamard_block_size=block_size,
        **static_args,
    )
    source = _source_after_transform(x, block_size)
    expected = _quantize_int8_reference(source, scale_matrix, group_size)

    if granularity == "tensor":
        assert result[2] is static_scale and result[1] is None
    else:
        assert result[1] is static_scale and result[2] is None
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale_dtype", ["float8e4m3", "float8e8m0"])
def test_static_group_scale_dtype_is_inferred(scale_dtype):
    x = torch.randn(2, 512, device="cuda", dtype=torch.float32)
    decoded_scales = torch.tensor([0.5, 1.0, 2.0, 4.0], device="cuda")
    if scale_dtype == "float8e4m3":
        group_scales = decoded_scales.to(torch.float8_e4m3fn)
    elif hasattr(torch, "float8_e8m0fnu"):
        group_scales = decoded_scales.to(torch.float8_e8m0fnu)
    else:
        group_scales = (torch.log2(decoded_scales) + 127).to(torch.uint8)

    result = process_input(
        x,
        quant_mode="static_group",
        quant_dtype="int8",
        quant_group_size=128,
        group_scales=group_scales,
    )
    expected = _quantize_int8_reference(x, decoded_scales.expand(2, 4), 128)
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("quant_dtype", ["float8e4m3", "int8"])
def test_static_tensor_non_power_of_two_hidden_size(quant_dtype):
    x = torch.randn((4, 7168), device="cuda", dtype=torch.float32)
    scale = torch.tensor([0.025], device="cuda")
    result = process_input(
        x,
        quant_mode="static_tensor",
        quant_dtype=quant_dtype,
        token_scales=scale,
    )

    if quant_dtype == "float8e4m3":
        expected = (x / scale).to(torch.float8_e4m3fn)
        torch.testing.assert_close(result[0].float(), expected.float(), rtol=0, atol=0)
    else:
        expected = _quantize_int8_reference(x, scale, x.size(-1))
        torch.testing.assert_close(result[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [None, 128])
@pytest.mark.parametrize("scale_dtype", ["float8e4m3", "float8e8m0"])
def test_static_tensor_combines_with_dynamic_group(block_size, scale_dtype):
    torch.manual_seed(15)
    x = torch.randn(3, 512, device="cuda", dtype=torch.float32)
    source = _source_after_transform(x, block_size)
    static_scale = torch.tensor([0.5], device="cuda")
    dynamic_scales = _empty_group_scales(3, 4, scale_dtype)

    result = process_input(
        x,
        quant_mode="static_tensor_dynamic_group",
        quant_dtype="float8e4m3",
        quant_group_size=128,
        group_scales=dynamic_scales,
        token_scales=static_scale,
        hadamard_block_size=block_size,
    )

    normalized = source.reshape(3, 4, 128) / static_scale
    raw = normalized.abs().amax(-1) / 448.0
    if scale_dtype == "float8e4m3":
        encoded = raw.to(torch.float8_e4m3fn).float()
    else:
        encoded = torch.exp2(torch.ceil(torch.log2(raw)))
    expected_q = (normalized / encoded[:, :, None]).to(torch.float8_e4m3fn)

    assert result[1] is not None
    torch.testing.assert_close(result[1].float(), encoded, rtol=0, atol=0)
    torch.testing.assert_close(result[0].float().reshape_as(expected_q), expected_q.float(), rtol=0, atol=0)


@pytest.mark.parametrize("quant_dtype", ["float8e4m3", "int8"])
@pytest.mark.parametrize(
    ("hadamard_block_size", "group_size"),
    [(None, 128), (128, 128), (128, 512)],
)
def test_dynamic_group_e4_token_centric_schedules(quant_dtype, hadamard_block_size, group_size):
    """Rows >= 512 select the token-centric pure per-group E4 schedule."""
    torch.manual_seed(23)
    rows = 512
    hidden_size = 512
    x = torch.randn((rows, hidden_size), device="cuda", dtype=torch.float32)
    static_scale = torch.tensor([0.5], device="cuda")
    group_scales = _empty_group_scales(rows, hidden_size // group_size, "float8e4m3")

    result = process_input(
        x,
        quant_mode="static_tensor_dynamic_group",
        quant_dtype=quant_dtype,
        quant_group_size=group_size,
        token_scales=static_scale,
        group_scales=group_scales,
        hadamard_block_size=hadamard_block_size,
    )

    source = _source_after_transform(x, hadamard_block_size)
    grouped = source.reshape(rows, hidden_size // group_size, group_size)
    if quant_dtype == "int8":
        raw = grouped.abs().amax(-1) / 127.0
    else:
        raw = grouped.abs().amax(-1) / 448.0
    expected_scale = (raw / static_scale).to(torch.float8_e4m3fn)
    if quant_dtype == "int8":
        expected_output = _quantize_int8_reference(source, expected_scale.float() * static_scale, group_size)
    else:
        expected_output = (grouped / (static_scale * expected_scale.float())[:, :, None]).to(
            torch.float8_e4m3fn
        )

    torch.testing.assert_close(result[1], expected_scale, rtol=0, atol=0)
    if quant_dtype == "int8":
        torch.testing.assert_close(result[0], expected_output, rtol=0, atol=1)
    else:
        actual = result[0].float().reshape_as(expected_output)
        expected = expected_output.float()
        mismatches = actual != expected
        assert mismatches.count_nonzero() / mismatches.numel() < 1e-5


@pytest.mark.parametrize("packed", [False, True])
def test_dynamic_group_e4_token_centric_m_major(packed):
    torch.manual_seed(24)
    x = torch.randn(512, 512, device="cuda")
    row_major = process_input(
        x,
        quant_mode="dynamic_group",
        quant_dtype="float8e4m3",
        quant_group_size=128,
        group_scales=_empty_group_scales(512, 4, "float8e4m3"),
    )
    m_major = process_input(
        x,
        quant_mode="dynamic_group",
        quant_dtype="float8e4m3",
        quant_group_size=128,
        group_scales=_empty_group_scales(
            512,
            4,
            "float8e4m3",
            "mx_packed" if packed else "m_major",
        ),
        group_scale_layout="mx_packed" if packed else "m_major",
    )
    if packed:
        unpacked = m_major[1].view(torch.uint8).reshape(1, 512, 4)[0]
        expected = row_major[1].view(torch.uint8)
    else:
        unpacked = m_major[1][:, :512].T
        expected = row_major[1]
    torch.testing.assert_close(unpacked, expected, rtol=0, atol=0)
    torch.testing.assert_close(m_major[0].float(), row_major[0].float(), rtol=0, atol=0)


@pytest.mark.parametrize("scale_dtype", ["float32", "float8e4m3", "float8e8m0"])
def test_legacy_hadamard_m_major_scale_layout(scale_dtype):
    torch.manual_seed(18)
    inputs = torch.randn((3, 512), device="cuda", dtype=torch.float32)
    _, row_major = hadamard_quant_input(
        inputs,
        block_size=128,
        quant_dtype="float8e4m3",
        group_size=128,
        scale_dtype=scale_dtype,
    )
    _, m_major = hadamard_quant_input(
        inputs,
        block_size=128,
        quant_dtype="float8e4m3",
        group_size=128,
        scale_dtype=scale_dtype,
        m_major_scale=True,
    )

    if scale_dtype == "float8e8m0":
        assert m_major.shape == (1, 4, 4)
        unpacked = m_major.view(torch.uint8).reshape(1, 4, 4)[0, :3]
        torch.testing.assert_close(unpacked, row_major.view(torch.uint8), rtol=0, atol=0)
    else:
        torch.testing.assert_close(m_major[:, :3].T.float(), row_major.float(), rtol=0, atol=0)
