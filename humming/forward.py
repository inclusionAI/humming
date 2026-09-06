import json

import torch

from humming import ops
from humming.config import LayerConfig, MmaType
from humming.tune import get_heuristics_class


def _resolve_use_pdl(
    config: LayerConfig,
    inputs: torch.Tensor,
    use_pdl: bool | None,
) -> bool:
    if use_pdl is not None:
        return use_pdl
    if not inputs.is_cuda:
        return False

    heuristics = get_heuristics_class(inputs.device)
    shape_m = inputs.numel() // inputs.size(-1)
    return heuristics.should_use_pdl_for_input(config, shape_m)


def _prepare_input_scale(config: LayerConfig, input_scale: torch.Tensor) -> torch.Tensor:
    mx_scale_dtype = str(config.as_dtype) in ("float8e4m3", "float8e8m0")
    grouped_mxmma = config.mma_type == MmaType.MXMMA and config.input_scale_group_size > 0
    if mx_scale_dtype and grouped_mxmma and input_scale.dtype != torch.int32:
        if config.use_mxumma and input_scale.ndim == 2:
            group_size = config.input_scale_group_size
            logical_groups = (config.shape_k - config.pad_shape_k) // group_size
            packed_groups = (config.shape_k // group_size + 3) // 4 * 4
            if input_scale.size(-1) == logical_groups and logical_groups < packed_groups:
                input_scale = torch.nn.functional.pad(
                    input_scale.view(torch.uint8), (0, packed_groups - logical_groups)
                )
        packed_scale = input_scale.view(torch.int32)
        if input_scale.ndim == 3:
            packed_scale = packed_scale.reshape(input_scale.size(0), input_scale.size(1))
        return packed_scale
    return input_scale


def _group_scale_layout(config: LayerConfig, m_major_scale: bool) -> str:
    if not m_major_scale or config.input_scale_group_size == 0:
        return "row_major"
    if str(config.as_dtype) in ("float8e4m3", "float8e8m0"):
        return "mx_packed"
    return "m_major"


def may_process_input(
    config: LayerConfig,
    inputs: torch.Tensor,
    *,
    outputs: torch.Tensor | None = None,
    group_scales: torch.Tensor | None = None,
    token_scales: torch.Tensor | None = None,
    activation_type: str = "none",
    activation_impl: str | None = None,
    hadamard_block_size: int | None = None,
    layout: str = "normal",
    expert_layout: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    zero_invalid: bool = False,
    m_major_scale: bool = False,
    use_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    config.check_device(inputs.device)
    should_quantize = config.a_dtype.num_bits != 16
    quant_mode = "none"
    quant_dtype = None
    quant_group_size = None
    group_scale_dtype = None
    if should_quantize:
        assert config.as_dtype is not None
        quant_mode = "dynamic_group" if config.input_scale_group_size > 0 else "dynamic_token"
        quant_dtype = str(config.a_dtype)
        quant_group_size = config.input_scale_group_size or None
        group_scale_dtype = str(config.as_dtype)

    no_transform = (
        activation_type == "none"
        and activation_impl in (None, "")
        and (hadamard_block_size is None or hadamard_block_size <= 1)
    )
    no_layout = layout == "normal" and expert_layout is None and indices is None and not zero_invalid
    no_buffers = outputs is None and group_scales is None and token_scales is None
    if not should_quantize and no_transform and no_layout and no_buffers:
        return inputs, None, None

    resolved_use_pdl = _resolve_use_pdl(config, inputs, use_pdl)
    return ops.process_input(
        inputs=inputs,
        outputs=outputs,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        quant_group_size=quant_group_size,
        group_scales=group_scales,
        group_scale_dtype=group_scale_dtype,
        token_scales=token_scales,
        activation_type=activation_type,
        activation_impl=activation_impl,
        hadamard_block_size=hadamard_block_size,
        layout=layout,
        expert_layout=expert_layout,
        indices=indices,
        zero_invalid=zero_invalid,
        group_scale_layout=_group_scale_layout(config, m_major_scale),
        use_pdl=resolved_use_pdl,
    )


def may_quant_input(
    config: LayerConfig,
    inputs: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    quanted_input: torch.Tensor | None = None,
    use_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if config.a_dtype.num_bits == 16:
        config.check_device(inputs.device)
        return inputs, None
    if input_scale is not None:
        config.check_device(inputs.device)
        return inputs, _prepare_input_scale(config, input_scale)
    outputs, group_scales, token_scales = may_process_input(
        config,
        inputs=inputs,
        outputs=quanted_input,
        m_major_scale=(
            config.mma_type == MmaType.MXMMA
            and config.input_scale_group_size > 0
            and not config.use_mxumma
        ),
        use_pdl=use_pdl,
    )
    if token_scales is not None:
        token_scales = token_scales.unsqueeze(-1)
    scale = group_scales if group_scales is not None else token_scales
    assert scale is not None
    return outputs, _prepare_input_scale(config, scale)


def humming_forward(
    config: LayerConfig,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    weight_scale_2: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
    compute_config: dict | str | None = None,
    tuning_config: dict | list | str | None = None,
    hadamard_block_size: int | None = None,
    use_pdl: bool | None = None,
) -> torch.Tensor:
    m_major_scale = False
    if config.input_scale_group_size > 0:
        parsed_compute_config = compute_config
        if isinstance(parsed_compute_config, str) and parsed_compute_config:
            parsed_compute_config = json.loads(parsed_compute_config)
        if isinstance(parsed_compute_config, dict):
            m_major_scale = bool(parsed_compute_config.get("use_m_major_input_scale", False))

    if input_scale is None:
        inputs, group_scales, token_scales = may_process_input(
            config,
            inputs=inputs,
            hadamard_block_size=hadamard_block_size,
            m_major_scale=m_major_scale,
            use_pdl=use_pdl,
        )
        if token_scales is not None:
            token_scales = token_scales.unsqueeze(-1)
        input_scale = group_scales if group_scales is not None else token_scales
    if input_scale is not None:
        input_scale = _prepare_input_scale(config, input_scale)

    if isinstance(compute_config, dict):
        compute_config = json.dumps(compute_config)
    if isinstance(tuning_config, (list, dict)):
        tuning_config = json.dumps(tuning_config)

    return ops.humming_gemm(
        layer_config=config.to_str(),
        compute_config=compute_config,
        tuning_config=tuning_config,
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        weight_scale=weight_scale,
        zero_point=zero_point,
        bias=bias,
        weight_scale_2=weight_scale_2,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        expert_layout=expert_layout,
        locks=locks,
        top_k=top_k,
        valid_shape_m=valid_shape_m,
    )
