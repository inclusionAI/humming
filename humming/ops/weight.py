import math

import torch
from torch._subclasses.fake_tensor import FakeTensor

from humming import dtypes
from humming.kernel.dequant_weight import DequantKernel
from humming.kernel.pack_weight import PackWeightKernel
from humming.kernel.process_mxfp4 import ProcessMxfp4W4A8Kernel
from humming.kernel.quant_weight import QuantWeightKernel
from humming.kernel.repack_weight import RepackWeightKernel
from humming.kernel.unpack_weight import UnpackWeightKernel
from humming.ops.utils import (
    _prepare_output,
    _prepare_output_arg,
    _select_output,
    _should_use_torch_op,
    register_op,
)


@register_op("humming::dequant_weight", mutates_args=["outputs"])
def _dequant_weight_op(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    exponent_bits: int,
    mantissa_bits: int,
    is_signed: bool,
) -> torch.Tensor:
    assert inputs.dtype == torch.int32
    assert inputs.is_cuda
    assert inputs.is_contiguous()

    outputs, returned_outputs = _prepare_output(outputs, inputs.shape, torch.float32, inputs.device)

    if not isinstance(inputs, FakeTensor):
        kernel = DequantKernel()
        kernel(
            inputs=inputs,
            outputs=outputs,
            exponent_bits=exponent_bits,
            mantissa_bits=mantissa_bits,
            is_signed=is_signed,
        )

    return returned_outputs


@register_op("humming::pack_weight", mutates_args=["outputs"])
def _pack_weight_op(inputs: torch.Tensor, outputs: torch.Tensor, num_bits: int) -> torch.Tensor:
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert inputs.size(-1) % 32 == 0
    assert inputs.size(-1) * num_bits % 32 == 0
    assert inputs.dtype == torch.int32

    output_shape = inputs.shape[:-1] + (inputs.size(-1) * num_bits // 32,)
    outputs, returned_outputs = _prepare_output(outputs, output_shape, torch.int32, inputs.device)

    if not isinstance(inputs, FakeTensor):
        kernel = PackWeightKernel(num_bits=num_bits)
        kernel(inputs=inputs, outputs=outputs)

    return returned_outputs


@register_op("humming::quant_weight", mutates_args=["outputs", "scales", "zero_point"])
def _quant_weight_op(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    scales: torch.Tensor,
    zero_point: torch.Tensor,
    source_dtype_str: str,
    target_dtype_str: str,
    group_size: int,
    has_scale: bool,
    use_e8m0_scale: bool,
    has_zero_point: bool,
    is_fp_zero_point: bool,
    allow_negative_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_size = inputs.size(-1) if group_size <= 0 else group_size
    source_dtype = dtypes.DataType.from_str(source_dtype_str)
    target_dtype = dtypes.DataType.from_str(target_dtype_str)

    assert inputs.is_cuda
    assert inputs.is_contiguous()

    outputs, returned_outputs = _prepare_output(outputs, inputs.shape, torch.int32, inputs.device)

    if has_scale:
        scale_shape = inputs.shape[:-1] + (inputs.size(-1) // group_size,)
        scale_dtype = torch.float8_e8m0fnu if use_e8m0_scale else torch.float32
        scale_device = inputs.device
    else:
        scale_shape = (0,)
        scale_dtype = torch.float32
        scale_device = torch.device("cpu")
    scales, returned_scales = _prepare_output(scales, scale_shape, scale_dtype, scale_device)

    if has_scale and has_zero_point:
        zero_point_shape = scale_shape
        zero_point_dtype = torch.float32 if is_fp_zero_point else torch.int32
        zero_point_device = inputs.device
    else:
        zero_point_shape = (0,)
        zero_point_dtype = torch.float32
        zero_point_device = torch.device("cpu")
    zero_point, returned_zero_point = _prepare_output(
        zero_point,
        zero_point_shape,
        zero_point_dtype,
        zero_point_device,
    )

    if not isinstance(inputs, FakeTensor):
        kernel = QuantWeightKernel(
            source_dtype=source_dtype,
            target_dtype=target_dtype,
            group_size=group_size,
            has_scale=has_scale,
            has_zero_point=has_zero_point,
            use_e8m0_scale=use_e8m0_scale,
            is_fp_zero_point=is_fp_zero_point,
            allow_negative_scale=allow_negative_scale,
        )
        kernel(inputs=inputs, outputs=outputs, scales=scales, zero_point=zero_point)

    return returned_outputs, returned_scales, returned_zero_point


@register_op("humming::repack_weight", mutates_args=["outputs"])
def _repack_weight_op(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    weight_bits: int,
    activation_bits: int,
    is_weight_packed: bool,
    should_preprocess_for_int2fp: bool = False,
    should_preprocess_with_zp: bool = False,
    use_wgmma: bool = False,
    use_fused_e8m0_scale: bool = False,
    interleave_mode: int = 3,
    group_size_zp: int = 0,
    padded_shape_n: int | None = None,
    padded_shape_k: int | None = None,
    zero_point: torch.Tensor | None = None,
    use_packed_k_layout: bool = False,
    use_native_dequant: bool = False,
) -> torch.Tensor:
    assert inputs.ndim in [2, 3]
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert inputs.dtype == torch.int32
    device = inputs.device
    num_experts = 1 if inputs.ndim == 2 else inputs.size(0)
    shape_n = inputs.size(-2)
    shape_k = inputs.size(-1)
    if is_weight_packed:
        assert shape_k * 32 % weight_bits == 0
        shape_k = shape_k * 32 // weight_bits

    if should_preprocess_with_zp:
        assert zero_point is not None and zero_point.dtype == torch.int32
        group_size_zp = shape_k if group_size_zp == 0 else group_size_zp
        zero_point_shape = inputs.shape[:-1] + (math.ceil(shape_k / group_size_zp),)

        if is_weight_packed:
            assert shape_n * weight_bits % 32 == 0
            packed_shape_n = shape_n * weight_bits // 32
            zero_point_shape = zero_point_shape[:-2] + (packed_shape_n,) + zero_point_shape[-1:]

        assert zero_point.shape == zero_point_shape

    pack_size_k = 64 if use_packed_k_layout else 256 // activation_bits
    output_shape: tuple[int, ...] = (
        shape_k // pack_size_k,
        shape_n * pack_size_k * weight_bits // 32,
    )
    if inputs.ndim == 3:
        output_shape = (num_experts,) + output_shape

    outputs, returned_outputs = _prepare_output(outputs, output_shape, torch.int32, device)

    if not isinstance(inputs, FakeTensor):
        kernel = RepackWeightKernel(
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            is_weight_packed=is_weight_packed,
            should_preprocess_for_int2fp=should_preprocess_for_int2fp,
            should_preprocess_with_zp=should_preprocess_with_zp,
            use_wgmma=use_wgmma,
            use_fused_e8m0_scale=use_fused_e8m0_scale,
            group_size_zp=group_size_zp,
            use_packed_k_layout=use_packed_k_layout,
            use_native_dequant=use_native_dequant,
        )

        kernel(
            inputs=inputs,
            outputs=outputs,
            zero_point=zero_point,
            padded_shape_n=padded_shape_n,
            padded_shape_k=padded_shape_k,
            interleave_mode=interleave_mode,
        )

    return returned_outputs


@register_op("humming::unpack_weight", mutates_args=["outputs"])
def _unpack_weight_op(inputs: torch.Tensor, outputs: torch.Tensor, num_bits: int) -> torch.Tensor:
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert inputs.size(-1) % num_bits == 0
    assert inputs.dtype == torch.int32

    shape_k = inputs.size(-1) // num_bits * 32
    output_shape = inputs.shape[:-1] + (shape_k,)
    outputs, returned_outputs = _prepare_output(outputs, output_shape, torch.int32, inputs.device)

    if not isinstance(inputs, FakeTensor):
        kernel = UnpackWeightKernel(num_bits=num_bits)
        kernel(inputs=inputs, outputs=outputs)

    return returned_outputs


@register_op("humming::process_mxfp4_w4a8_weight", mutates_args=["outputs"])
def _process_mxfp4_w4a8_weight_op(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    delta_scale_offsets: torch.Tensor,
) -> torch.Tensor:
    assert inputs.dtype == torch.int32
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert delta_scale_offsets.dtype == torch.uint8
    assert delta_scale_offsets.device == inputs.device
    assert delta_scale_offsets.is_contiguous()
    assert inputs.nelement() >= delta_scale_offsets.nelement() * 4

    def is_power_of_two(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    repeat_count = inputs.nelement() // delta_scale_offsets.nelement() // 4
    assert is_power_of_two(repeat_count)
    delta_scale_offsets = delta_scale_offsets.repeat_interleave(repeat_count, -1)

    outputs, returned_outputs = _prepare_output(outputs, inputs.shape, inputs.dtype, inputs.device)

    if not isinstance(inputs, FakeTensor):
        kernel = ProcessMxfp4W4A8Kernel()
        kernel(inputs, outputs, delta_scale_offsets)

    return returned_outputs


def dequant_weight(
    inputs: torch.Tensor,
    exponent_bits: int,
    mantissa_bits: int,
    is_signed: bool,
    outputs: torch.Tensor | None = None,
) -> torch.Tensor:
    outputs = _prepare_output_arg(inputs, outputs, torch.float32)
    op = torch.ops.humming.dequant_weight if _should_use_torch_op(inputs) else _dequant_weight_op
    returned_outputs = op(
        inputs,
        outputs,
        exponent_bits,
        mantissa_bits,
        is_signed,
    )
    return _select_output(outputs, returned_outputs)


def pack_weight(inputs: torch.Tensor, num_bits: int, outputs: torch.Tensor | None = None) -> torch.Tensor:
    outputs = _prepare_output_arg(inputs, outputs, torch.int32)
    op = torch.ops.humming.pack_weight if _should_use_torch_op(inputs) else _pack_weight_op
    returned_outputs = op(inputs, outputs, num_bits)
    return _select_output(outputs, returned_outputs)


def quant_weight(
    inputs: torch.Tensor,
    source_dtype_str: str,
    target_dtype_str: str,
    group_size: int,
    has_scale: bool,
    use_e8m0_scale: bool,
    has_zero_point: bool,
    is_fp_zero_point: bool,
    allow_negative_scale: bool = True,
    outputs: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = _prepare_output_arg(inputs, outputs, torch.int32)
    scales = _prepare_output_arg(inputs, scales, torch.float32)
    zero_point = _prepare_output_arg(inputs, zero_point, torch.float32)

    op = torch.ops.humming.quant_weight if _should_use_torch_op(inputs) else _quant_weight_op
    returned_outputs, returned_scales, returned_zero_point = op(
        inputs,
        outputs,
        scales,
        zero_point,
        source_dtype_str,
        target_dtype_str,
        group_size,
        has_scale,
        use_e8m0_scale,
        has_zero_point,
        is_fp_zero_point,
        allow_negative_scale,
    )
    return (
        _select_output(outputs, returned_outputs),
        _select_output(scales, returned_scales),
        _select_output(zero_point, returned_zero_point),
    )


def repack_weight(
    inputs: torch.Tensor,
    weight_bits: int,
    activation_bits: int,
    is_weight_packed: bool,
    should_preprocess_for_int2fp: bool = False,
    should_preprocess_with_zp: bool = False,
    use_wgmma: bool = False,
    use_fused_e8m0_scale: bool = False,
    interleave_mode: int = 3,
    group_size_zp: int = 0,
    padded_shape_n: int | None = None,
    padded_shape_k: int | None = None,
    zero_point: torch.Tensor | None = None,
    use_packed_k_layout: bool = False,
    use_native_dequant: bool = False,
    outputs: torch.Tensor | None = None,
) -> torch.Tensor:
    outputs = _prepare_output_arg(inputs, outputs, torch.int32)
    op = torch.ops.humming.repack_weight if _should_use_torch_op(inputs) else _repack_weight_op
    returned_outputs = op(
        inputs,
        outputs,
        weight_bits,
        activation_bits,
        is_weight_packed,
        should_preprocess_for_int2fp,
        should_preprocess_with_zp,
        use_wgmma,
        use_fused_e8m0_scale,
        interleave_mode,
        group_size_zp,
        padded_shape_n,
        padded_shape_k,
        zero_point,
        use_packed_k_layout,
        use_native_dequant,
    )
    return _select_output(outputs, returned_outputs)


def unpack_weight(
    inputs: torch.Tensor,
    num_bits: int,
    outputs: torch.Tensor | None = None,
) -> torch.Tensor:
    outputs = _prepare_output_arg(inputs, outputs, torch.int32)
    op = torch.ops.humming.unpack_weight if _should_use_torch_op(inputs) else _unpack_weight_op
    returned_outputs = op(inputs, outputs, num_bits)
    return _select_output(outputs, returned_outputs)


def process_mxfp4_w4a8_weight(
    inputs: torch.Tensor,
    delta_scale_offsets: torch.Tensor,
    inplace: bool = False,
    outputs: torch.Tensor | None = None,
) -> torch.Tensor:
    assert not inplace or outputs is None
    if inplace:
        outputs = inputs
    else:
        outputs = _prepare_output_arg(inputs, outputs, inputs.dtype)

    op = torch.ops.humming.process_mxfp4_w4a8_weight
    if not _should_use_torch_op(inputs):
        op = _process_mxfp4_w4a8_weight_op
    returned_outputs = op(inputs, outputs, delta_scale_offsets)
    return _select_output(outputs, returned_outputs)
