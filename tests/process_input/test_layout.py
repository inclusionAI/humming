import pytest
import torch

from humming.ops.input import hadamard_transform, process_input

from ._reference import (
    _empty_group_scales,
    _quantize_int8_reference,
    _round_positive_m3_rne,
)


def test_raw_permute_and_scatter():
    x = torch.randn(4, 384, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    permutation = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int64)
    permuted = process_input(
        x,
        layout="permute",
        expert_layout=offsets,
        indices=permutation,
    )[0]
    assert torch.equal(permuted.view(torch.uint8), x[permutation].view(torch.uint8))

    destinations = torch.tensor([[3, -1], [0, 2], [1, -1], [4, 5]], device="cuda")
    outputs = torch.zeros(8, 384, device="cuda", dtype=x.dtype)
    scattered = process_input(x, outputs=outputs, layout="scatter", indices=destinations)[0]
    for input_row, output_rows in enumerate(destinations.tolist()):
        for output_row in output_rows:
            if output_row >= 0:
                assert torch.equal(scattered[output_row].view(torch.uint8), x[input_row].view(torch.uint8))
    assert torch.count_nonzero(scattered[6:]) == 0


@pytest.mark.parametrize(
    ("quant_mode", "group_scale_dtype", "hadamard_block_size"),
    [
        ("group", "float32", None),
        ("group", "float8e4m3", 128),
        ("group", "float8e8m0", 128),
        ("token", None, None),
        ("token", None, 128),
        ("group_token", "float8e4m3", None),
        ("group_token", "float8e4m3", 128),
    ],
)
def test_scatter_layout(
    quant_mode,
    group_scale_dtype,
    hadamard_block_size,
):
    torch.manual_seed(31)
    x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    scatter_idx = torch.tensor(
        [[5, -1, 1], [3, 0, -1], [4, -1, 2]],
        device="cuda",
        dtype=torch.int64,
    )
    rows = scatter_idx.numel()
    common = dict(
        quant_mode={
            "group": "dynamic_group",
            "token": "dynamic_token",
            "group_token": "dynamic_group_token",
        }[quant_mode],
        quant_group_size=128,
        hadamard_block_size=hadamard_block_size,
    )
    reference_scales = None
    if group_scale_dtype is not None and quant_mode == "group":
        reference_scales = _empty_group_scales(3, 4, group_scale_dtype)
    reference = process_input(
        x,
        quant_dtype="float8e4m3",
        group_scales=reference_scales,
        **common,
    )

    outputs = torch.zeros((rows, x.size(1)), device=x.device, dtype=torch.float8_e4m3fn)
    group_scales = None
    if reference[1] is not None:
        group_scales = torch.zeros(
            (rows, x.size(1) // 128),
            device=x.device,
            dtype=reference[1].dtype,
        )
    token_scales = None
    if reference[2] is not None:
        token_scales = torch.zeros(rows, device=x.device, dtype=torch.float32)

    result = process_input(
        x,
        quant_dtype="float8e4m3",
        outputs=outputs,
        group_scales=group_scales,
        token_scales=token_scales,
        layout="scatter",
        indices=scatter_idx,
        **common,
    )

    for input_row, target_row in enumerate(scatter_idx.tolist()):
        for target in target_row:
            if target < 0:
                continue
            for index, (actual, expected) in enumerate(zip(result, reference, strict=True)):
                if actual is None:
                    continue
                actual, expected = actual[target], expected[input_row]
                if index < 2:
                    actual, expected = actual.float(), expected.float()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    for untouched in (6, 7, 8):
        assert torch.count_nonzero(result[0][untouched]) == 0
        if result[1] is not None:
            assert torch.count_nonzero(result[1][untouched]) == 0
        if result[2] is not None:
            assert result[2][untouched] == 0


def test_scatter_layout_static_scale_and_m_major_scale():
    torch.manual_seed(32)
    x = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
    scatter_idx = torch.tensor([[6, 0], [-1, 4]], device="cuda", dtype=torch.int64)
    static_scale = torch.tensor([0.5], device=x.device, dtype=torch.float32)

    static_reference = process_input(
        x,
        quant_mode="static_tensor",
        quant_dtype="int8",
        quant_group_size=128,
        token_scales=static_scale,
    )
    static_outputs = torch.zeros((7, 512), device=x.device, dtype=torch.int8)
    static_result = process_input(
        x,
        quant_mode="static_tensor",
        quant_dtype="int8",
        outputs=static_outputs,
        quant_group_size=128,
        token_scales=static_scale,
        layout="scatter",
        indices=scatter_idx,
    )
    torch.testing.assert_close(static_result[0][6], static_reference[0][0], rtol=0, atol=0)
    torch.testing.assert_close(static_result[0][0], static_reference[0][0], rtol=0, atol=0)
    torch.testing.assert_close(static_result[0][4], static_reference[0][1], rtol=0, atol=0)
    assert torch.count_nonzero(static_result[0][1]) == 0

    reference = process_input(
        x,
        quant_mode="dynamic_group",
        quant_dtype="float8e4m3",
        quant_group_size=128,
        group_scale_layout="m_major",
    )
    outputs = torch.zeros((7, 512), device=x.device, dtype=torch.float8_e4m3fn)
    group_scales = torch.zeros((4, 8), device=x.device, dtype=torch.float32)
    result = process_input(
        x,
        quant_mode="dynamic_group",
        quant_dtype="float8e4m3",
        outputs=outputs,
        group_scales=group_scales,
        quant_group_size=128,
        group_scale_layout="m_major",
        layout="scatter",
        indices=scatter_idx,
    )
    torch.testing.assert_close(result[1][:, 6], reference[1][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(result[1][:, 0], reference[1][:, 0], rtol=0, atol=0)
    torch.testing.assert_close(result[1][:, 4], reference[1][:, 1], rtol=0, atol=0)
    assert torch.count_nonzero(result[1][:, 1]) == 0


@pytest.mark.parametrize("layout", ["grouped", "permute"])
def test_moe_compact_layouts(layout):
    torch.manual_seed(13)
    x = torch.randn(5, 256, device="cuda")
    expert_offsets = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    static_scale = torch.tensor([0.02, 0.04], device="cuda")
    permute_idx = None
    input_rows = torch.arange(5, device="cuda")
    if layout == "permute":
        permute_idx = torch.tensor([3, 1, 4, 0, 2], device="cuda", dtype=torch.int64)
        input_rows = permute_idx

    result = process_input(
        x,
        quant_mode="static_tensor",
        quant_dtype="int8",
        quant_group_size=128,
        token_scales=static_scale,
        layout=layout,
        expert_layout=expert_offsets,
        indices=permute_idx,
    )

    row_scales = torch.tensor([0.02, 0.02, 0.04, 0.04, 0.04], device="cuda").expand(2, -1).T
    expected = _quantize_int8_reference(x[input_rows], row_scales, 128)
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("zero_invalid", [False, True])
def test_moe_grouped_padded_invalid_write_policy(zero_invalid):
    torch.manual_seed(14)
    x = torch.randn(2, 4, 256, device="cuda")
    valid_tokens = torch.tensor([3, 1], device="cuda", dtype=torch.int64)
    outputs = torch.full_like(x, 13, dtype=torch.int8)
    group_scales = torch.full((2, 4, 2), 7, device="cuda", dtype=torch.float8_e4m3fn)
    token_scales = torch.full((2, 4), 11, device="cuda", dtype=torch.float32)

    result = process_input(
        x,
        quant_mode="dynamic_group_token",
        quant_dtype="int8",
        outputs=outputs,
        group_scales=group_scales,
        token_scales=token_scales,
        quant_group_size=128,
        hadamard_block_size=128,
        layout="grouped_padded",
        expert_layout=valid_tokens,
        zero_invalid=zero_invalid,
    )
    invalid = torch.cat((result[0][0, 3:].flatten(), result[0][1, 1:].flatten()))
    invalid_groups = torch.cat((result[1][0, 3:].flatten(), result[1][1, 1:].flatten()))
    invalid_tokens = torch.cat((result[2][0, 3:].flatten(), result[2][1, 1:].flatten()))

    valid_mask = torch.arange(4, device="cuda")[None, :] < valid_tokens[:, None]
    transformed = hadamard_transform(x, 128)[valid_mask]
    grouped = transformed.reshape(-1, 2, 128)
    raw = grouped.abs().amax(-1) / 127.0
    m3 = _round_positive_m3_rne(raw)
    expected_token = torch.exp2(torch.ceil(torch.log2(m3.amax(-1) / 448.0)))
    expected_group = (m3 / expected_token[:, None]).to(torch.float8_e4m3fn)
    expected_output = _quantize_int8_reference(transformed, m3, 128)
    torch.testing.assert_close(result[1][valid_mask], expected_group, rtol=0, atol=0)
    torch.testing.assert_close(result[2][valid_mask], expected_token, rtol=0, atol=0)
    torch.testing.assert_close(result[0][valid_mask], expected_output, rtol=0, atol=1)

    if zero_invalid:
        assert torch.count_nonzero(invalid) == 0
        assert torch.count_nonzero(invalid_groups.view(torch.uint8)) == 0
        assert torch.count_nonzero(invalid_tokens) == 0
    else:
        assert torch.all(invalid == 13)
        assert torch.all(invalid_groups == 7)
        assert torch.all(invalid_tokens == 11)
