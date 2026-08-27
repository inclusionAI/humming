import pytest
import torch

from humming.ops.input import QuantizationMode, hadamard_transform, process_input


def test_quantization_mode_accepts_enum_and_string():
    assert QuantizationMode("dynamic_group") is QuantizationMode.DynamicGroup
    assert QuantizationMode.DynamicGroup.value == "dynamic_group"


@pytest.mark.parametrize(
    ("dtype", "capability", "minimum"),
    [
        ("float8e4m3", (7, 5), "SM89"),
        ("float8e5m2", (7, 5), "SM89"),
        ("float8e3m4", (8, 9), "SM100"),
        ("float4e0m3", (8, 9), "SM100"),
        ("float4e2m1", (8, 9), "SM100"),
    ],
)
def test_target_dtype_capability_guard(monkeypatch, dtype, capability, minimum):
    x = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    with pytest.raises(RuntimeError, match=rf"{dtype} output requires {minimum}"):
        process_input(x, quant_mode="dynamic_group", quant_dtype=dtype)


def test_inplace_validation():
    x = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    indices = torch.arange(5, device="cuda", dtype=torch.int64)
    with pytest.raises(AssertionError, match="quantization"):
        process_input(x, inplace=True, quant_mode="dynamic_group", quant_dtype="int8")
    with pytest.raises(AssertionError, match="binary activation"):
        process_input(
            x,
            inplace=True,
            activation_type="binary_split",
            activation_impl="a * b",
        )
    with pytest.raises(AssertionError, match="permute layout"):
        process_input(
            x,
            inplace=True,
            layout="permute",
            expert_layout=offsets,
            indices=indices,
        )
    with pytest.raises(AssertionError, match="separate output"):
        process_input(x, outputs=torch.empty_like(x), inplace=True)


def test_inplace_alias_is_checked_after_cache_hit():
    x = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
    process_input(x, outputs=x, inplace=True)
    with pytest.raises(RuntimeError, match="must alias inputs"):
        process_input(x, outputs=torch.empty_like(x), inplace=True)


def test_scatter_layout_validation():
    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        process_input(
            x,
            quant_mode="dynamic_group",
            quant_dtype="float8e4m3",
            layout="scatter",
            indices=torch.zeros((2, 2), device=x.device, dtype=torch.int32),
        )


def test_non_power_of_two_group_size_is_rejected():
    x = torch.randn(2, 384, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="power of 2"):
        process_input(
            x,
            quant_mode="dynamic_group",
            quant_dtype="float8e4m3",
            quant_group_size=96,
        )


def test_group_size_above_limit_is_rejected():
    x = torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="per-group.*<= 512"):
        process_input(
            x,
            quant_mode="dynamic_group",
            quant_dtype="float8e4m3",
            quant_group_size=1024,
        )


def test_hadamard_block_above_limit_is_rejected():
    inputs = torch.randn((1, 1024), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match=r"\[2, 512\]"):
        hadamard_transform(inputs, block_size=1024)
