import torch
from torch._subclasses.fake_tensor import FakeTensor

from .enums import GroupScaleLayout, QuantizationMode
from .process import process_input


def hadamard_transform(
    inputs: torch.Tensor,
    block_size: int,
    outputs: torch.Tensor | None = None,
    use_pdl: bool = False,
) -> torch.Tensor:
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert 2 <= block_size <= 512 and (block_size & (block_size - 1)) == 0, (
        f"block_size must be a power of 2 in [2, 512], got {block_size}"
    )
    assert inputs.size(-1) % block_size == 0, (
        f"last dim {inputs.size(-1)} must be divisible by block_size {block_size}"
    )
    assert inputs.dtype in (torch.float16, torch.bfloat16, torch.float32)

    if outputs is None:
        outputs = torch.empty_like(inputs)
    else:
        assert outputs.shape == inputs.shape
        assert outputs.dtype == inputs.dtype
        assert outputs.is_contiguous()

    if not isinstance(inputs, FakeTensor):
        process_input(
            inputs,
            outputs=outputs,
            hadamard_block_size=block_size,
            use_pdl=use_pdl,
        )

    return outputs


def hadamard_quant_input(
    inputs: torch.Tensor,
    block_size: int,
    quant_dtype: str,
    group_size: int | None = None,
    outputs: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
    m_major_scale: bool = False,
    scale_dtype: str = "float32",
    global_scale: torch.Tensor | None = None,
    use_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_size is None or group_size == 0:
        group_size = inputs.size(-1)

    group_scale_layout = GroupScaleLayout.RowMajor
    if m_major_scale:
        group_scale_layout = GroupScaleLayout.MMajor
        if scale_dtype == "float8e8m0":
            group_scale_layout = GroupScaleLayout.MxPacked

    quant_mode = (
        QuantizationMode.StaticTensorDynamicGroup
        if global_scale is not None
        else QuantizationMode.DynamicGroup
    )

    quantized, result_group_scales, _ = process_input(
        inputs,
        outputs=outputs,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        quant_group_size=group_size,
        group_scales=scales,
        group_scale_dtype=scale_dtype,
        token_scales=global_scale,
        hadamard_block_size=block_size,
        group_scale_layout=group_scale_layout,
        use_pdl=use_pdl,
    )
    assert result_group_scales is not None
    return quantized, result_group_scales
