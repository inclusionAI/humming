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
    if str(config.as_dtype) == "float8e8m0" and input_scale.dtype != torch.int32:
        return input_scale.view(torch.int32)
    return input_scale


def may_quant_input(
    config: LayerConfig,
    inputs: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    quanted_input: torch.Tensor | None = None,
    use_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    config.check_device(inputs.device)
    if config.a_dtype.num_bits == 16:
        return inputs, None
    if input_scale is not None:
        return inputs, input_scale
    use_pdl = _resolve_use_pdl(config, inputs, use_pdl)
    assert config.as_dtype is not None
    quanted_input, input_scale = ops.quant_input(
        inputs=inputs,
        outputs=quanted_input,
        dtype=str(config.a_dtype),
        group_size=config.input_scale_group_size or None,
        m_major_scale=(config.mma_type == MmaType.MXMMA and config.input_scale_group_size > 0),
        scale_dtype=str(config.as_dtype),
        use_pdl=use_pdl,
    )
    return quanted_input, _prepare_input_scale(config, input_scale)


def may_hadamard_quant_input(
    config: LayerConfig,
    inputs: torch.Tensor,
    hadamard_block_size: int | None = None,
    input_scale: torch.Tensor | None = None,
    quanted_input: torch.Tensor | None = None,
    m_major_scale: bool = False,
    use_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    config.check_device(inputs.device)
    should_rotate = hadamard_block_size is not None and hadamard_block_size > 1
    should_quant = config.a_dtype.num_bits != 16

    if input_scale is not None:
        return inputs, input_scale
    if not should_rotate and not should_quant:
        return inputs, None
    use_pdl = _resolve_use_pdl(config, inputs, use_pdl)
    if should_rotate and not should_quant:
        outputs = ops.hadamard_transform(
            inputs=inputs,
            block_size=hadamard_block_size,
            outputs=quanted_input,
            use_pdl=use_pdl,
        )
        return outputs, None
    if not should_rotate:
        assert config.as_dtype is not None
        outputs, scales = ops.quant_input(
            inputs=inputs,
            dtype=str(config.a_dtype),
            outputs=quanted_input,
            group_size=config.input_scale_group_size,
            m_major_scale=m_major_scale,
            scale_dtype=str(config.as_dtype),
            use_pdl=use_pdl,
        )
        return outputs, _prepare_input_scale(config, scales)
    assert config.as_dtype is not None
    outputs, scales = ops.hadamard_quant_input(
        inputs=inputs,
        block_size=hadamard_block_size,
        quant_dtype=str(config.a_dtype),
        group_size=config.input_scale_group_size,
        outputs=quanted_input,
        m_major_scale=m_major_scale,
        scale_dtype=str(config.as_dtype),
        use_pdl=use_pdl,
    )
    return outputs, _prepare_input_scale(config, scales)


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

    inputs, input_scale = may_hadamard_quant_input(
        config,
        inputs=inputs,
        hadamard_block_size=hadamard_block_size,
        input_scale=input_scale,
        m_major_scale=m_major_scale,
        use_pdl=use_pdl,
    )

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
