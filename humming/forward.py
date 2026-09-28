import json

import torch

from humming import ops
from humming.config import LayerConfig
from humming.tune import get_heuristics_class


@torch.compiler.assume_constant_result
def _prepare_forward_configs(
    compute_config: dict | str | None,
    tuning_config: dict | list | str | None,
) -> tuple[str | None, str | None, bool]:
    """Constant-fold static config processing that Dynamo cannot trace.

    Keep JSON parsing/serialization and similar config-only work here so
    humming_forward supports full-graph capture. Return strings and scalar
    metadata to avoid Dynamo guards on parsed dictionaries. Tensor- or
    shape-dependent work must stay outside this helper.
    """
    parsed_compute_config = compute_config
    if isinstance(parsed_compute_config, str) and parsed_compute_config:
        parsed_compute_config = json.loads(parsed_compute_config)

    m_major_scale = False
    if isinstance(parsed_compute_config, dict):
        m_major_scale = bool(parsed_compute_config.get("use_m_major_input_scale", False))

    if isinstance(compute_config, dict):
        compute_config = json.dumps(compute_config)
    if isinstance(tuning_config, (list, dict)):
        tuning_config = json.dumps(tuning_config)
    return compute_config, tuning_config, m_major_scale


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
    expert_tokens: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    num_valid_tokens: torch.Tensor | None = None,
    zero_invalid: bool = False,
    m_major_scale: bool = False,
    use_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    config.check_device(inputs.device)
    should_quantize = config.input_quant_mode.should_quantize
    quant_mode = "none"
    quant_dtype = None
    quant_group_size = None
    group_scale_dtype = None
    if should_quantize:
        assert config.as_dtype is not None
        quant_mode = config.input_quant_mode.value
        quant_dtype = str(config.a_dtype)
        quant_group_size = config.input_scale_group_size or None
        group_scale_dtype = str(config.as_dtype)

    should_transform = hadamard_block_size is not None and hadamard_block_size > 1
    has_activation = activation_type != "none"
    should_scatter = layout == "scatter"
    should_process = should_quantize or should_transform or has_activation or should_scatter
    should_process = should_process or num_valid_tokens is not None
    if not should_process:
        if outputs is not None and outputs is not inputs:
            outputs.copy_(inputs)
            return outputs, None, None
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
        expert_tokens=expert_tokens,
        scatter_idx=scatter_idx,
        num_valid_tokens=num_valid_tokens,
        zero_invalid=zero_invalid,
        use_m_major_input_scale=m_major_scale,
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
        return inputs, input_scale
    outputs, group_scales, token_scales = may_process_input(
        config,
        inputs=inputs,
        outputs=quanted_input,
        m_major_scale=(config.use_block_scaled_mma and config.input_scale_group_size > 0),
        use_pdl=use_pdl,
    )
    scale = group_scales if group_scales is not None else token_scales
    assert scale is not None
    return outputs, scale


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
    input_scale_2: torch.Tensor | None = None,
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
    compute_config, tuning_config, m_major_scale = _prepare_forward_configs(compute_config, tuning_config)

    unquantized_dtype = [torch.bfloat16, torch.float16, torch.float32]
    should_quantize = config.input_quant_mode.should_quantize
    is_quantized_input = inputs.dtype not in unquantized_dtype
    should_transform = hadamard_block_size is not None and hadamard_block_size > 1
    should_process = not is_quantized_input and (should_quantize or should_transform)
    if should_process:
        group_scales = input_scale if config.input_quant_mode.has_group_scale else None
        token_scales = input_scale_2 if config.input_quant_mode.has_secondary_scale else input_scale
        inputs, group_scales, token_scales = may_process_input(
            config,
            inputs=inputs,
            group_scales=group_scales,
            token_scales=token_scales,
            hadamard_block_size=hadamard_block_size,
            m_major_scale=m_major_scale,
            use_pdl=use_pdl,
        )
        input_scale = group_scales if config.input_quant_mode.has_group_scale else token_scales
        input_scale_2 = token_scales if config.input_quant_mode.has_secondary_scale else None

    return ops.humming_gemm(
        layer_config=config.to_str(),
        compute_config=compute_config,
        tuning_config=tuning_config,
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        input_scale_2=input_scale_2,
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
