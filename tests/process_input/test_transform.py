import pytest
import torch
import torch.nn.functional as F

from humming.ops.input import process_input

from ._reference import _hadamard_reference, _quantize_int8_reference


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_raw_copy_is_bitwise(dtype):
    x = torch.randn(5, 384, device="cuda", dtype=dtype)
    x.view(torch.uint8).flatten()[17] = 0xFF
    result = process_input(x)
    assert torch.equal(result[0].view(torch.uint8), x.view(torch.uint8))
    assert result[1] is None and result[2] is None


@pytest.mark.parametrize(
    ("activation", "activation_impl"),
    [
        ("relu", "a > 0.f ? a : 0.f"),
        ("gelu", "0.5f * a * (1.f + erff(a * 0.7071067811865475f))"),
    ],
)
def test_unquantized_unary_activation(activation, activation_impl):
    x = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16)
    expected = getattr(F, activation)(x)
    result = process_input(
        x,
        activation_type="unary",
        activation_impl=activation_impl,
    )
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("input_layout", ["non_interleaved", "interleaved"])
def test_unquantized_silu_mul(input_layout):
    gate = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    if input_layout == "non_interleaved":
        inputs = torch.cat((gate, up), dim=-1)
    else:
        inputs = torch.stack((gate, up), dim=-1).flatten(-2)
    expected = (F.silu(gate.float()) * up.float()).to(gate.dtype)
    activation_type = "binary_split" if input_layout == "non_interleaved" else "binary_interleaved"
    result = process_input(
        inputs,
        activation_type=activation_type,
        activation_impl="a / (1.f + expf(-a)) * b",
    )
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)


def test_unquantized_hadamard_and_activation():
    x = torch.randn(3, 512, device="cuda", dtype=torch.float32)
    result = process_input(
        x,
        activation_type="unary",
        activation_impl="a > 0.f ? a : 0.f",
        hadamard_block_size=128,
    )
    expected = _hadamard_reference(F.relu(x), 128)
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["normal", "grouped", "grouped_padded"])
def test_unquantized_unary_inplace(layout):
    x = torch.randn((2, 4, 384) if layout == "grouped_padded" else (5, 384), device="cuda")
    original = x.clone()
    kwargs = {}
    if layout == "grouped":
        kwargs["expert_layout"] = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    elif layout == "grouped_padded":
        kwargs["expert_layout"] = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
        kwargs["zero_invalid"] = True

    result = process_input(
        x,
        outputs=x,
        activation_type="unary",
        activation_impl="a > 0.f ? a : 0.f",
        layout=layout,
        **kwargs,
    )

    assert result[0] is x
    expected = original.relu()
    if layout == "grouped_padded":
        expected[0, 3:] = 0
        expected[1, 1:] = 0
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


def test_unquantized_hadamard_inplace():
    x = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
    expected = _hadamard_reference(x, 128).to(x.dtype)
    result = process_input(x, outputs=x, hadamard_block_size=128)
    assert result[0] is x
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


def test_inplace_cache_hit_keeps_input_storage():
    first = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
    second = torch.randn_like(first)
    process_input(first, outputs=first)
    result = process_input(second, outputs=second)
    assert result[0] is second


def test_custom_binary_activation_hadamard_and_quantization():
    a = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    inputs = torch.cat((a, b), dim=-1)
    result = process_input(
        inputs,
        quant_mode="dynamic_group",
        quant_dtype="int8",
        quant_group_size=128,
        activation_type="binary_split",
        activation_impl="a * a + b",
        hadamard_block_size=128,
    )

    activated = a.float() * a.float() + b.float()
    source = _hadamard_reference(activated, 128)
    grouped = source.reshape(3, 2, 128)
    expected_scales = grouped.abs().amax(-1) / 127.0
    expected = _quantize_int8_reference(source, expected_scales, 128)
    torch.testing.assert_close(result[1], expected_scales, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(result[0], expected, rtol=0, atol=1)


def test_multiline_activation_impl():
    a = torch.randn(3, 256, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    inputs = torch.cat((a, b), dim=-1)
    activation_impl = """[](float a, float b) {
        float squared = a * a;
        return squared + b;
    }(a, b)"""

    result = process_input(
        inputs,
        activation_type="binary_split",
        activation_impl=activation_impl,
    )

    torch.testing.assert_close(result[0], a * a + b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("block_size", [16, 256, 512])
def test_hadamard_involution(block_size):
    torch.manual_seed(0)
    inputs = torch.randn((4, block_size * 2), device="cuda", dtype=torch.float32)
    transformed = process_input(inputs, hadamard_block_size=block_size)[0]
    restored = process_input(transformed, hadamard_block_size=block_size)[0]
    torch.testing.assert_close(restored, inputs, rtol=1e-5, atol=1e-5)


def test_precise_activation_keeps_subnormals_across_cached_variants():
    """Opting out of fast math must affect both compiled code and its cache key."""
    x = torch.full((1, 32), 2.0**-140, device="cuda", dtype=torch.float32)
    expected = x / 3.0
    for precise in [True, False, True]:
        result = process_input(
            x,
            activation_type="unary",
            activation_impl="a / 3.0f",
            disable_fast_math=precise,
        )[0]
        if precise:
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
        else:
            assert torch.count_nonzero(result) == 0
