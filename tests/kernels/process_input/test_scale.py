import pytest
import torch

from humming.config import InputQuantizationMode
from humming.ops.input import process_input
from humming.testing.process_input import (
    assert_process_input_close,
    process_input_ref,
    skip_if_process_input_unsupported,
)


@pytest.mark.parametrize("quant_dtype", ["int4", "int8", "float8e4m3"])
@pytest.mark.parametrize(
    "shape_m,hidden_size,quant_group_size,quant_mode,group_scale_dtype,use_m_major_input_scale",
    [
        (3, 768, 128, "static_tensor", "float32", False),
        (129, 32768, 512, "dynamic_token", "float32", False),
        (129, 32768, 512, "dynamic_token", "float32", True),
        (3, 768, 128, "dynamic_token", "float32", True),
        (4, 7168, 128, "dynamic_group", "float32", False),
        (3, 768, 128, "dynamic_group", "float32", True),
        (3, 768, 128, "dynamic_group", "float8e4m3", False),
        (3, 768, 128, "dynamic_group", "float8e4m3", True),
        (3, 768, 128, "dynamic_group", "float8e8m0", False),
        (3, 768, 128, "dynamic_group", "float8e8m0", True),
        (3, 768, 128, "static_tensor_dynamic_group", "float32", False),
        (3, 768, 128, "static_tensor_dynamic_group", "float8e4m3", True),
        (3, 768, 128, "static_tensor_dynamic_group", "float8e8m0", False),
        (3, 768, 128, "dynamic_group_token", "float8e4m3", False),
        (129, 32768, 512, "dynamic_group_token", "float8e4m3", True),
    ],
)
def test_quantization_scales(
    quant_dtype,
    shape_m,
    hidden_size,
    quant_group_size,
    quant_mode,
    group_scale_dtype,
    use_m_major_input_scale,
):
    skip_if_process_input_unsupported(quant_dtype, group_scale_dtype)
    torch.manual_seed(0)

    num_groups = hidden_size // quant_group_size
    inputs = torch.randn(shape_m, num_groups, quant_group_size, device="cuda")

    group_exponents = torch.arange(num_groups, device="cuda") % 5 - 2
    group_amplitudes = torch.exp2(group_exponents.float())
    inputs *= group_amplitudes[None, :, None]
    inputs = inputs.flatten(1)
    inputs[0] = 0

    quant_mode = InputQuantizationMode(quant_mode)
    static_tensor_scale = None
    if quant_mode.has_tensor_scale:
        static_tensor_scale = torch.tensor([0.5], device="cuda")

    options = dict(
        quant_mode=quant_mode.value,
        quant_dtype=quant_dtype,
        quant_group_size=quant_group_size,
        group_scale_dtype=group_scale_dtype,
        use_m_major_input_scale=use_m_major_input_scale,
    )

    expected = process_input_ref(inputs, static_tensor_scale=static_tensor_scale, **options)
    actual = process_input(inputs, token_scales=static_tensor_scale, **options)

    assert_process_input_close(
        actual,
        expected,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        group_scale_dtype=group_scale_dtype,
        quant_group_size=quant_group_size,
        use_m_major_input_scale=use_m_major_input_scale,
    )
