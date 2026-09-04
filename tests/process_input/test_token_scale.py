import pytest
import torch

from humming.ops.input import process_input

from ._reference import (
    _quantize_int8_reference,
    _round_positive_m3_rne,
    _source_after_transform,
)


@pytest.mark.parametrize("block_size", [None, 128])
def test_dynamic_token_is_independent_mode(block_size):
    torch.manual_seed(11)
    x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
    source = _source_after_transform(x, block_size)
    result = process_input(
        x,
        quant_mode="dynamic_token",
        quant_dtype="float8e4m3",
        quant_group_size=128,
        hadamard_block_size=block_size,
    )
    expected_scale = source.abs().amax(dim=-1) / 448.0

    assert result[1] is None
    assert result[2] is not None
    torch.testing.assert_close(result[2], expected_scale, rtol=1e-6, atol=1e-7)
    expected_q = (source / expected_scale[:, None]).to(torch.float8_e4m3fn)
    torch.testing.assert_close(result[0].float(), expected_q.float(), rtol=0, atol=0)


@pytest.mark.parametrize("quant_dtype", ["float8e4m3", "int8"])
@pytest.mark.parametrize(
    ("hadamard_block_size", "group_size"),
    [(None, 128), (128, 128), (128, 512), (512, 128)],
)
def test_dynamic_token_register_schedules(quant_dtype, hadamard_block_size, group_size):
    torch.manual_seed(24)
    rows = 4
    hidden_size = 4096
    x = torch.randn((rows, hidden_size), device="cuda", dtype=torch.float32)
    num_groups = hidden_size // group_size
    static_scale = torch.ones(num_groups, device="cuda", dtype=torch.float32)

    result = process_input(
        x,
        quant_mode="dynamic_token",
        quant_dtype=quant_dtype,
        quant_group_size=group_size,
        hadamard_block_size=hadamard_block_size,
    )

    source = _source_after_transform(x, hadamard_block_size).float()
    grouped = source.reshape(rows, num_groups, group_size)
    normalized = grouped / static_scale[None, :, None]
    if quant_dtype == "int8":
        expected_scale = normalized.abs().amax(dim=(1, 2)) / 127.0
        expected_output = _quantize_int8_reference(
            source,
            static_scale[None, :] * expected_scale[:, None],
            group_size,
        )
        torch.testing.assert_close(result[0], expected_output, rtol=0, atol=1)
    else:
        expected_scale = normalized.abs().amax(dim=(1, 2)) / 448.0
        expected_output = (grouped / (static_scale[None, :, None] * expected_scale[:, None, None])).to(
            torch.float8_e4m3fn
        )
        torch.testing.assert_close(
            result[0].float().reshape_as(expected_output),
            expected_output.float(),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(result[2], expected_scale, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("hadamard_block_size", [None, 128])
def test_dynamic_token_long_k_register_schedule(hadamard_block_size):
    torch.manual_seed(31)
    x = torch.randn((2, 32768), device="cuda", dtype=torch.float32)
    result = process_input(
        x,
        quant_mode="dynamic_token",
        quant_dtype="float8e4m3",
        quant_group_size=128,
        hadamard_block_size=hadamard_block_size,
    )

    source = _source_after_transform(x, hadamard_block_size)
    expected_scale = source.abs().amax(-1) / 448.0
    expected_output = (source / expected_scale[:, None]).to(torch.float8_e4m3fn)
    torch.testing.assert_close(result[2], expected_scale, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(result[0].float(), expected_output.float(), rtol=0, atol=0)


@pytest.mark.parametrize("quant_dtype", ["float8e4m3", "int8"])
@pytest.mark.parametrize("hidden_size", [7168, 11008])
@pytest.mark.parametrize("hadamard_block_size", [None, 128])
def test_dynamic_token_non_power_of_two_hidden_size(quant_dtype, hidden_size, hadamard_block_size):
    torch.manual_seed(32)
    x = torch.randn((4, hidden_size), device="cuda", dtype=torch.float32)
    result = process_input(
        x,
        quant_mode="dynamic_token",
        quant_dtype=quant_dtype,
        quant_group_size=x.size(-1),
        hadamard_block_size=hadamard_block_size,
    )

    source = _source_after_transform(x, hadamard_block_size)
    target_maximum = 448.0 if quant_dtype == "float8e4m3" else 127.0
    expected_scale = source.abs().amax(-1) / target_maximum
    torch.testing.assert_close(result[2], expected_scale, rtol=1e-6, atol=1e-7)
    if quant_dtype == "float8e4m3":
        expected_output = (source / expected_scale[:, None]).to(torch.float8_e4m3fn)
        torch.testing.assert_close(result[0].float(), expected_output.float(), rtol=0, atol=0)
    else:
        expected_output = _quantize_int8_reference(source, expected_scale, x.size(-1))
        torch.testing.assert_close(result[0], expected_output, rtol=0, atol=1)


def test_dynamic_group_token_e4m3_staged_factorization():
    torch.manual_seed(12)
    group_size = 128
    exponents = torch.arange(10, device="cuda", dtype=torch.float32)
    x = torch.randn(2, 10, group_size, device="cuda") * torch.exp2(exponents)[None, :, None]
    x = x.reshape(2, -1)

    result = process_input(
        x,
        quant_mode="dynamic_group_token",
        quant_dtype="int8",
        quant_group_size=group_size,
    )

    grouped = x.reshape(2, 10, group_size)
    raw = grouped.abs().amax(-1) / 127.0
    m3 = _round_positive_m3_rne(raw)
    expected_token = torch.exp2(torch.ceil(torch.log2(m3.amax(-1) / 448.0)))
    expected_local = (m3 / expected_token[:, None]).to(torch.float8_e4m3fn)
    expected_q = _quantize_int8_reference(x, m3, group_size)

    assert result[1] is not None
    assert result[2] is not None
    torch.testing.assert_close(result[0], expected_q, rtol=0, atol=0)
    torch.testing.assert_close(result[1].float(), expected_local.float(), rtol=0, atol=0)
    torch.testing.assert_close(result[2], expected_token, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [4, 512, 2048])
@pytest.mark.parametrize(
    ("hadamard_block_size", "group_size"),
    [(None, 128), (128, 128), (128, 512)],
)
def test_group_token_register_schedules(rows, hadamard_block_size, group_size):
    """Exercise small, medium, and saturated-grid register plans."""
    torch.manual_seed(19)
    hidden_size = 512
    x = torch.randn((rows, hidden_size), device="cuda", dtype=torch.float32)
    result = process_input(
        x,
        quant_mode="dynamic_group_token",
        quant_dtype="float8e4m3",
        quant_group_size=group_size,
        hadamard_block_size=hadamard_block_size,
    )

    source = _source_after_transform(x, hadamard_block_size)
    grouped = source.reshape(rows, hidden_size // group_size, group_size)
    raw = grouped.abs().amax(-1) / 448.0
    m3 = _round_positive_m3_rne(raw)
    expected_token = torch.exp2(torch.ceil(torch.log2(m3.amax(-1) / 448.0)))
    expected_group = (m3 / expected_token[:, None]).to(torch.float8_e4m3fn)
    expected_output = (grouped / m3[:, :, None]).to(torch.float8_e4m3fn)

    torch.testing.assert_close(result[1], expected_group, rtol=0, atol=0)
    torch.testing.assert_close(result[2], expected_token, rtol=0, atol=0)
    actual_output = result[0].float().reshape_as(expected_output)
    expected_output = expected_output.float()
    mismatches = actual_output != expected_output
    assert mismatches.count_nonzero() / mismatches.numel() < 1e-5
    if mismatches.any():
        relative_error = (actual_output[mismatches] - expected_output[mismatches]).abs() / expected_output[
            mismatches
        ].abs().clamp_min(1)
        assert relative_error.max() <= 0.125


@pytest.mark.parametrize(
    ("hadamard_block_size", "group_size"),
    [(None, 512), (128, 128), (128, 512), (512, 128)],
)
def test_group_token_int8_register_schedules(hadamard_block_size, group_size):
    torch.manual_seed(20)
    rows = 512
    hidden_size = 4096
    x = torch.randn((rows, hidden_size), device="cuda", dtype=torch.float32)

    result = process_input(
        x,
        quant_mode="dynamic_group_token",
        quant_dtype="int8",
        quant_group_size=group_size,
        hadamard_block_size=hadamard_block_size,
    )

    source = _source_after_transform(x, hadamard_block_size)
    grouped = source.reshape(rows, hidden_size // group_size, group_size)
    raw = grouped.abs().amax(-1) / 127.0
    m3 = _round_positive_m3_rne(raw)
    expected_token = torch.exp2(torch.ceil(torch.log2(m3.amax(-1) / 448.0)))
    expected_group = (m3 / expected_token[:, None]).to(torch.float8_e4m3fn)
    expected_output = _quantize_int8_reference(source, m3, group_size)

    torch.testing.assert_close(result[1], expected_group, rtol=0, atol=0)
    torch.testing.assert_close(result[2], expected_token, rtol=0, atol=0)
    torch.testing.assert_close(result[0], expected_output, rtol=0, atol=1)


@pytest.mark.parametrize("layout", ["grouped", "permute"])
def test_group_token_moe_compact_layouts(layout):
    torch.manual_seed(21)
    x = torch.randn(5, 256, device="cuda")
    expert_offsets = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    permute_idx = None
    input_rows = torch.arange(5, device="cuda")
    if layout == "permute":
        permute_idx = torch.tensor([3, 1, 4, 0, 2], device="cuda", dtype=torch.int64)
        input_rows = permute_idx

    result = process_input(
        x,
        quant_mode="dynamic_group_token",
        quant_dtype="float8e4m3",
        quant_group_size=128,
        layout=layout,
        expert_layout=expert_offsets,
        indices=permute_idx,
    )

    grouped = x[input_rows].reshape(5, 2, 128)
    raw = grouped.abs().amax(-1) / 448.0
    m3 = _round_positive_m3_rne(raw)
    expected_token = torch.exp2(torch.ceil(torch.log2(m3.amax(-1) / 448.0)))
    expected_group = (m3 / expected_token[:, None]).to(torch.float8_e4m3fn)
    expected_output = (grouped / m3[:, :, None]).to(torch.float8_e4m3fn)

    torch.testing.assert_close(result[1], expected_group, rtol=0, atol=0)
    torch.testing.assert_close(result[2], expected_token, rtol=0, atol=0)
    torch.testing.assert_close(
        result[0].float().reshape_as(expected_output),
        expected_output.float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("packed", [False, True])
def test_group_token_m_major_scale_layout(packed):
    torch.manual_seed(22)
    x = torch.randn(5, 512, device="cuda")
    row_major = process_input(
        x,
        quant_mode="dynamic_group_token",
        quant_dtype="float8e4m3",
        quant_group_size=128,
        hadamard_block_size=128,
    )
    m_major = process_input(
        x,
        quant_mode="dynamic_group_token",
        quant_dtype="float8e4m3",
        quant_group_size=128,
        hadamard_block_size=128,
        group_scale_layout="mx_packed" if packed else "m_major",
    )

    if packed:
        assert m_major[1].shape == (1, 8, 4)
        assert m_major[1].dtype == torch.float8_e4m3fn
        unpacked = m_major[1][0, :5]
        expected_group_scales = row_major[1]
    else:
        assert m_major[1].shape == (4, 8)
        assert m_major[1].dtype == torch.float8_e4m3fn
        unpacked = m_major[1][:, :5].T
        expected_group_scales = row_major[1]
    torch.testing.assert_close(unpacked, expected_group_scales, rtol=0, atol=0)
    torch.testing.assert_close(m_major[2], row_major[2], rtol=0, atol=0)
    torch.testing.assert_close(m_major[0].float(), row_major[0].float(), rtol=0, atol=0)
