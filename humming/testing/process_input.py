import torch

from humming import dtypes, ops
from humming.config import ActivationType, InputQuantizationMode, ProcessInputLayoutType
from humming.config.mma import UNDOCUMENTED_FP_DTYPES, supports_undocumented_fp_dtypes
from humming.utils.math import is_pow_of_two, round_up

ACTIVATION_TYPE_IMPL_TEST_MAP = {
    ActivationType.None_: {
        "impl": "",
        "torch_impl": lambda a: a,
    },
    ActivationType.Unary: {
        "impl": "a > 0 ? a - 2 : a + 1",
        "torch_impl": lambda a: torch.where(a > 0, a - 2, a + 1),
    },
    ActivationType.BinarySplit: {
        "impl": "0.1 * a * b + 0.2 * a + 0.3 * b + 0.4",
        "torch_impl": lambda a, b: 0.1 * a * b + 0.2 * a + 0.3 * b + 0.4,
    },
    ActivationType.BinaryInterleaved: {
        "impl": "0.1 * a * b + 0.2 * a + 0.3 * b + 0.4",
        "torch_impl": lambda a, b: 0.1 * a * b + 0.2 * a + 0.3 * b + 0.4,
    },
}

QUANT_DTYPE_MAX_VAL_MAP = {
    dtypes.float8e3m4: 30,
    dtypes.float8e4m3: 448,
    dtypes.float8e5m2: 57344,
    dtypes.float4e2m1: 6,
    dtypes.float4e0m3: 7,
    dtypes.int4: 7,
    dtypes.int8: 127,
}


def apply_activation_ref(inputs: torch.Tensor, activation: ActivationType) -> torch.Tensor:
    assert inputs.dtype == torch.float32
    assert inputs.ndim == 2

    if activation == ActivationType.Unary:
        inputs = ACTIVATION_TYPE_IMPL_TEST_MAP[activation]["torch_impl"](inputs)
    elif activation in [ActivationType.BinarySplit, ActivationType.BinaryInterleaved]:
        assert inputs.size(1) % 2 == 0
        hidden_size = inputs.size(1) // 2
        if activation == ActivationType.BinarySplit:
            inputs1, inputs2 = inputs[:, :hidden_size], inputs[:, hidden_size:]
        else:
            inputs1, inputs2 = inputs[:, ::2], inputs[:, 1::2]

        inputs = ACTIVATION_TYPE_IMPL_TEST_MAP[activation]["torch_impl"](inputs1, inputs2)

    return inputs


def hadamard_transform_ref(inputs: torch.Tensor, hadamard_block_size: int) -> torch.Tensor:
    assert inputs.dtype == torch.float32
    assert inputs.ndim == 2
    assert hadamard_block_size > 0 and is_pow_of_two(hadamard_block_size)

    h = torch.ones((1, 1), device=inputs.device, dtype=torch.float32)
    while h.size(0) < hadamard_block_size:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)

    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        result = inputs.reshape(-1, hadamard_block_size).matmul(h).reshape(inputs.shape)
        result = result / (hadamard_block_size**0.5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    return result


def decode_quantized(values: torch.Tensor, quant_dtype: str) -> torch.Tensor:
    if quant_dtype == "int4":
        shifts = torch.tensor([0, 4], device=values.device, dtype=torch.int32)
        values = (values.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF
        values = values.flatten(-2)
        return torch.where(values >= 8, values - 16, values).float()
    if quant_dtype == "float8e3m4":
        values = values.view(torch.uint8).to(torch.int32).contiguous()
        return ops.dequant_weight(values, 3, 4, True)
    if quant_dtype == "float4e0m3":
        values = ops.unpack_weight(values.view(torch.int32), 4)
        magnitude = (values & 0x7).float()
        return torch.where((values & 0x8) != 0, -magnitude, magnitude)
    if quant_dtype == "float4e2m1":
        values = ops.unpack_weight(values.view(torch.int32), 4)
        return ops.dequant_weight(values, 2, 1, True)
    return values.float()


def tensor_round_to(tensor: torch.Tensor, dtype: dtypes.DataType, round_mode: str = "rn"):
    assert tensor.dtype == torch.float32
    if dtype == dtypes.float32:
        return tensor

    assert round_mode in ["rn", "ru", "rz"]
    assert dtype in QUANT_DTYPE_MAX_VAL_MAP or dtype == dtypes.float8e8m0

    step = 1
    if dtype.is_floating_point_type and dtype.exponent_bits > 0:
        bias = 2 ** (dtype.exponent_bits - 1) - 1
        _, exponent = torch.frexp(tensor.abs())
        min_exponent = 1 - bias if dtype.mantissa_bits == 0 else 2 - bias
        step = exponent.clamp_min(min_exponent) - 1 - dtype.mantissa_bits
        step = torch.ldexp(torch.ones_like(tensor), step)

    tensor = tensor / step
    if round_mode == "rn":
        tensor = tensor.round()
    elif round_mode == "rz":
        tensor = tensor.trunc()
    elif round_mode == "ru":
        tensor = tensor.ceil()

    tensor = tensor * step
    if dtype == dtypes.float8e8m0:
        tensor = tensor.clamp(2.0**-127, 2.0**127)
    else:
        max_val = QUANT_DTYPE_MAX_VAL_MAP[dtype]
        min_val = -max_val - 1 if dtype.is_integer_type else -max_val
        tensor = tensor.clamp(min_val, max_val)

    if dtype == dtypes.int8:
        tensor = tensor.to(torch.int8)
    elif dtype == dtypes.int4:
        tensor = tensor.to(torch.int8).view(torch.uint8) & 0xF
    elif dtype == dtypes.float8e4m3:
        tensor = tensor.to(torch.float8_e4m3fn)
    elif dtype == dtypes.float8e5m2:
        tensor = tensor.to(torch.float8_e5m2)
    elif dtype == dtypes.float8e8m0:
        tensor = tensor.to(torch.float8_e8m0fnu)
    elif dtype == dtypes.float4e0m3:
        tensor = tensor.abs().to(torch.uint8) | (torch.signbit(tensor).to(torch.uint8) << 3)
    elif dtype in (dtypes.float8e3m4, dtypes.float4e2m1):
        bias = 2 ** (dtype.exponent_bits - 1) - 1
        magnitude = tensor.abs().float() / (2.0 ** (127 - bias))
        codes = magnitude.view(torch.int32) >> (23 - dtype.mantissa_bits)
        codes = codes.to(torch.uint8)
        tensor = codes | (torch.signbit(tensor).to(torch.uint8) << (dtype.num_bits - 1))

    if dtype.num_bits == 4:
        assert tensor.size(-1) % 2 == 0
        tensor = tensor[..., ::2] | (tensor[..., 1::2] << 4)
    return tensor


def apply_scale_layout_ref(group_scales: torch.Tensor, use_m_major_input_scale: bool = False):
    rows, groups = group_scales.shape
    if group_scales.element_size() == 1:
        pad_columns = round_up(groups, 4) - groups
        group_scales = torch.nn.functional.pad(group_scales.view(torch.uint8), (0, pad_columns))
        group_scales = group_scales.view(torch.int32)

    if use_m_major_input_scale:
        pad_rows = round_up(rows, 4) - rows
        group_scales = torch.nn.functional.pad(group_scales, (0, 0, 0, pad_rows))
        group_scales = group_scales.T.contiguous()
    return group_scales


def quant_input_ref(
    inputs: torch.Tensor,
    quant_mode: InputQuantizationMode,
    quant_dtype: dtypes.DataType,
    quant_group_size: int | None = None,
    group_scale_dtype: dtypes.DataType | None = None,
    static_tensor_scale: torch.Tensor | None = None,
    use_m_major_input_scale: bool = False,
):
    assert inputs.dtype == torch.float32
    assert inputs.ndim == 2
    quant_mode = InputQuantizationMode(quant_mode)
    quant_dtype = dtypes.DataType.from_any(quant_dtype)
    assert quant_mode.should_quantize

    hidden_size = inputs.size(1)
    if quant_mode.has_group_scale:
        quant_group_size = quant_group_size or min(hidden_size & -hidden_size, 512)
    else:
        quant_group_size = hidden_size
    if group_scale_dtype is None:
        group_scale_dtype = (
            dtypes.float8e4m3 if quant_mode == InputQuantizationMode.DynamicGroupToken else dtypes.float32
        )
    else:
        group_scale_dtype = dtypes.DataType.from_any(group_scale_dtype)

    shape_m = inputs.size(0)
    if quant_mode.has_tensor_scale:
        assert static_tensor_scale is not None
        assert static_tensor_scale.nelement() == 1
    if quant_mode == InputQuantizationMode.StaticTensor:
        quantized_inputs = tensor_round_to(inputs * static_tensor_scale.reciprocal(), dtype=quant_dtype)
        return quantized_inputs, None, static_tensor_scale

    inputs = inputs.reshape(-1, quant_group_size)
    absmax_val = inputs.abs().max(1)[0]
    dtype_max_val = QUANT_DTYPE_MAX_VAL_MAP[quant_dtype]
    scale = (absmax_val / dtype_max_val).clamp_min(1e-30)

    if quant_mode == InputQuantizationMode.DynamicGroupToken:
        assert group_scale_dtype == dtypes.float8e4m3

        inv_scale = (1 / scale).log2().ceil().exp2()
        scale = tensor_round_to(scale * inv_scale, dtype=dtypes.float8e4m3, round_mode="ru")
        scale = scale.float() / inv_scale

        quantized_inputs = tensor_round_to(inputs / scale.float().unsqueeze(-1), dtype=quant_dtype)
        scale = scale.view(shape_m, -1)
        token_scales = tensor_round_to(scale.amax(dim=1) / 448, dtype=dtypes.float8e8m0, round_mode="ru")
        token_scales = token_scales.float()
        scale = scale / token_scales.unsqueeze(-1)
        group_scales = tensor_round_to(scale, dtype=dtypes.float8e4m3, round_mode="ru")

    elif quant_mode.has_group_scale:
        assert group_scale_dtype in [dtypes.float8e4m3, dtypes.float8e8m0, dtypes.float32]
        if quant_mode.has_tensor_scale:
            scale = scale / static_tensor_scale
        scale = tensor_round_to(scale, dtype=group_scale_dtype, round_mode="ru")
        input_scale = scale.float()
        if quant_mode.has_tensor_scale:
            input_scale = input_scale * static_tensor_scale
        quantized_inputs = tensor_round_to(inputs * input_scale.reciprocal().unsqueeze(-1), dtype=quant_dtype)
        group_scales = scale
        token_scales = static_tensor_scale

    elif quant_mode.has_token_scale:
        quantized_inputs = tensor_round_to(inputs / scale.unsqueeze(-1), dtype=quant_dtype)
        group_scales = None
        token_scales = scale

    quantized_inputs = quantized_inputs.view(shape_m, -1)
    if group_scales is not None:
        group_scales = group_scales.view(shape_m, -1)
        group_scales = apply_scale_layout_ref(group_scales, use_m_major_input_scale)

    if quant_mode.has_token_scale:
        token_scales = token_scales.view(shape_m, 1)
        if use_m_major_input_scale:
            token_scales = token_scales.view(1, shape_m)

    return quantized_inputs, group_scales, token_scales


def apply_layout_ref(
    inputs: torch.Tensor,
    layout: ProcessInputLayoutType = ProcessInputLayoutType.Normal,
    expert_tokens: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    num_valid_tokens: torch.Tensor | None = None,
    zero_invalid: bool = False,
):
    layout = ProcessInputLayoutType(layout)
    rows = inputs.size(0)
    if layout == ProcessInputLayoutType.Normal and num_valid_tokens is None:
        return inputs

    output_rows = rows * scatter_idx.size(1) if layout == ProcessInputLayoutType.Scatter else rows
    outputs = torch.empty((output_rows, *inputs.shape[1:]), dtype=inputs.dtype, device=inputs.device)

    source = inputs.contiguous().view(torch.uint8).reshape(rows, -1)
    destination = outputs.view(torch.uint8).reshape(output_rows, -1)
    input_rows = torch.arange(rows, device=inputs.device)
    if layout == ProcessInputLayoutType.Scatter:
        in_range = (scatter_idx >= 0) & (scatter_idx < scatter_idx.numel())
        valid = in_range
        if num_valid_tokens is not None:
            valid = in_range & (scatter_idx < num_valid_tokens.reshape(()))
            if zero_invalid:
                zero_rows = scatter_idx[in_range & ~valid].long()
                destination[zero_rows] = 0
        source_rows = input_rows[:, None].expand_as(scatter_idx)[valid]
        output_rows = scatter_idx[valid].long()
    else:
        if layout == ProcessInputLayoutType.Normal:
            valid = input_rows < num_valid_tokens.reshape(())
        else:
            assert rows % expert_tokens.numel() == 0
            rows_per_expert = rows // expert_tokens.numel()
            local_rows = torch.arange(rows_per_expert, device=inputs.device)
            valid = (local_rows[None, :] < expert_tokens[:, None]).flatten()
        source_rows = output_rows = input_rows[valid]
        if zero_invalid:
            destination[~valid] = 0
    destination.index_copy_(0, output_rows, source.index_select(0, source_rows))
    return outputs


def process_input_ref(
    inputs: torch.Tensor,
    quant_mode: str = "none",
    quant_dtype: str | None = None,
    quant_group_size: int | None = None,
    group_scale_dtype: str | None = None,
    static_tensor_scale: torch.Tensor | None = None,
    activation_type: str = "none",
    hadamard_block_size: int | None = None,
    layout: str = "normal",
    expert_tokens: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    num_valid_tokens: torch.Tensor | None = None,
    zero_invalid: bool = False,
    use_m_major_input_scale: bool = False,
):
    quant_mode = InputQuantizationMode(quant_mode)
    activation_type = ActivationType(activation_type)
    inputs = inputs.float()
    inputs = apply_activation_ref(inputs, activation_type)
    if hadamard_block_size is not None and hadamard_block_size > 1:
        inputs = hadamard_transform_ref(inputs, hadamard_block_size)

    group_scales = token_scales = None
    if quant_mode.should_quantize:
        inputs, group_scales, token_scales = quant_input_ref(
            inputs,
            quant_mode,
            quant_dtype,
            quant_group_size,
            group_scale_dtype,
            static_tensor_scale,
        )

    layout_args = dict(
        layout=layout,
        expert_tokens=expert_tokens,
        scatter_idx=scatter_idx,
        num_valid_tokens=num_valid_tokens,
        zero_invalid=zero_invalid,
    )
    outputs = apply_layout_ref(inputs, **layout_args)
    if quant_mode.has_group_scale:
        group_scales = apply_layout_ref(group_scales, **layout_args)
        group_scales = apply_scale_layout_ref(group_scales, use_m_major_input_scale)
    if quant_mode.has_token_scale:
        token_scales = apply_layout_ref(token_scales, **layout_args)
        if use_m_major_input_scale:
            token_scales = token_scales.T.contiguous()

    return outputs, group_scales, token_scales


def skip_if_process_input_unsupported(quant_dtype=None, group_scale_dtype=None):
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor

    for dtype in (quant_dtype, group_scale_dtype):
        if dtype is None:
            continue

        dtype = dtypes.DataType.from_any(dtype)
        if dtype in (dtypes.float8e4m3, dtypes.float8e5m2) and capability < 89:
            pytest.skip(f"{dtype} requires SM89+")

        if dtype in UNDOCUMENTED_FP_DTYPES and not supports_undocumented_fp_dtypes(capability):
            pytest.skip(f"{dtype} requires SM100/103/120/121 cubin patching")

        if dtype == dtypes.float4e2m1 and capability < 100:
            pytest.skip(f"{dtype} requires SM100+")


def unpack_group_scales(
    group_scales: torch.Tensor,
    num_rows: int,
    num_groups: int,
    group_scale_dtype: dtypes.DataType | str,
    use_m_major_input_scale: bool = False,
) -> torch.Tensor:
    if use_m_major_input_scale:
        group_scales = group_scales.T.contiguous()
    group_scale_dtype = dtypes.DataType.from_any(group_scale_dtype)
    group_scales = group_scales.view(dtypes.torch_dtype_map[group_scale_dtype])
    return group_scales[:num_rows, :num_groups]


def assert_process_input_close(
    actual: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None],
    *,
    quant_mode: InputQuantizationMode | str = "none",
    quant_dtype: dtypes.DataType | str | None = None,
    quant_group_size: int | None = None,
    group_scale_dtype: dtypes.DataType | str = dtypes.float32,
    use_m_major_input_scale: bool = False,
    valid_rows: torch.Tensor | None = None,
    zero_invalid: bool = False,
) -> None:
    quant_mode = InputQuantizationMode(quant_mode)
    actual_outputs, actual_group_scales, actual_token_scales = actual
    expected_outputs, expected_group_scales, expected_token_scales = expected

    assert actual_outputs.shape == expected_outputs.shape
    num_rows = actual_outputs.size(0)
    if valid_rows is None:
        valid_rows = torch.arange(num_rows, device=actual_outputs.device)

    if quant_mode.should_quantize:
        quant_dtype = dtypes.DataType.from_any(quant_dtype)
        assert actual_outputs.dtype == expected_outputs.dtype
        actual_values = decode_quantized(actual_outputs, str(quant_dtype))
        expected_values = decode_quantized(expected_outputs, str(quant_dtype))
    else:
        actual_values = actual_outputs.float()
        expected_values = expected_outputs.to(actual_outputs.dtype).float()

    hidden_size = expected_values.size(1)
    actual_values = actual_values[valid_rows]
    expected_values = expected_values[valid_rows]

    if quant_mode.should_quantize:
        step = torch.ones_like(expected_values)
        if quant_dtype.is_floating_point_type and quant_dtype.exponent_bits > 0:
            bias = 2 ** (quant_dtype.exponent_bits - 1) - 1
            _, exponent = torch.frexp(expected_values.abs())
            exponent = exponent.clamp_min(2 - bias) - 1 - quant_dtype.mantissa_bits
            step = torch.ldexp(step, exponent)

        difference = (actual_values - expected_values) / step
        torch.testing.assert_close(difference, torch.zeros_like(difference), rtol=0, atol=1.001)
    elif actual_outputs.dtype == torch.float16:
        torch.testing.assert_close(actual_values, expected_values, rtol=5e-3, atol=5e-3)
    elif actual_outputs.dtype == torch.bfloat16:
        torch.testing.assert_close(actual_values, expected_values, rtol=2e-2, atol=2e-2)
    elif actual_outputs.dtype == torch.float32:
        torch.testing.assert_close(actual_values, expected_values, rtol=1e-5, atol=1e-5)

    if quant_mode.has_group_scale:
        assert quant_group_size is not None
        num_groups = hidden_size // quant_group_size
        assert actual_group_scales.shape == expected_group_scales.shape
        assert actual_group_scales.dtype == expected_group_scales.dtype
        actual_group_scales = unpack_group_scales(
            actual_group_scales, num_rows, num_groups, group_scale_dtype, use_m_major_input_scale
        )
        expected_group_scales = unpack_group_scales(
            expected_group_scales, num_rows, num_groups, group_scale_dtype, use_m_major_input_scale
        )

        actual_values = actual_group_scales.float()[valid_rows]
        expected_values = expected_group_scales.float()[valid_rows]
        torch.testing.assert_close(actual_values, expected_values, rtol=1e-5, atol=1e-7)
    else:
        assert actual_group_scales is None
        assert expected_group_scales is None

    if quant_mode.has_token_scale:
        assert actual_token_scales.shape == expected_token_scales.shape
        actual_values = actual_token_scales.reshape(num_rows)[valid_rows]
        expected_values = expected_token_scales.reshape(num_rows)[valid_rows]
        torch.testing.assert_close(actual_values, expected_values, rtol=1e-5, atol=1e-7)
    else:
        torch.testing.assert_close(actual_token_scales, expected_token_scales)

    if zero_invalid:
        invalid_rows = torch.ones(num_rows, dtype=torch.bool, device=actual_outputs.device)
        invalid_rows[valid_rows] = False
        actual_values = actual_outputs.contiguous().view(torch.uint8)
        assert torch.count_nonzero(actual_values[invalid_rows]) == 0

        if quant_mode.has_group_scale:
            actual_values = actual_group_scales.contiguous().view(torch.uint8)
            assert torch.count_nonzero(actual_values[invalid_rows]) == 0

        if quant_mode.has_token_scale:
            actual_values = actual_token_scales.reshape(num_rows)
            assert torch.count_nonzero(actual_values[invalid_rows]) == 0
