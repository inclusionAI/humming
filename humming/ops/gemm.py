import json

import torch
from torch._subclasses.fake_tensor import FakeTensor

from humming import dtypes
from humming.kernel.humming import HummingKernel
from humming.ops.launcher import launch_kernel
from humming.ops.utils import _should_use_torch_op, init_humming_launcher, register_op


@torch.compiler.assume_constant_result
def _get_humming_gemm_output_meta(layer_config: str, compute_config: str | None):
    layer_obj = json.loads(layer_config)
    compute_obj = json.loads(compute_config or "{}")
    shape_n = int(layer_obj["shape_n"]) - int(layer_obj.get("pad_shape_n", 0))
    c_dtype = dtypes.DataType.from_str(layer_obj["c_dtype"])
    output_dtype = dtypes.torch_dtype_map[c_dtype]
    return shape_n, output_dtype, compute_obj.get("gemm_type") == "indexed"


@register_op("humming::prepare_kernels")
def _prepare_kernels_op(
    device_guard: torch.Tensor,
    layer_config: str,
    compute_config: str | None = None,
    tuning_config: str | None = None,
) -> torch.Tensor:
    if isinstance(device_guard, FakeTensor):
        init_humming_launcher()
        _, _, tuning_obj = HummingKernel._resolve_configs(layer_config, compute_config, tuning_config)
        num_configs = len(tuning_obj) if isinstance(tuning_obj, list) else 1
        return torch.empty((num_configs * 4,), dtype=torch.int64, device="cpu")

    return HummingKernel.prepare_kernels(layer_config, compute_config, tuning_config)


def humming_gemm(
    *,
    layer_config: str,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
    compute_config: str | None = None,
    tuning_config: str | None = None,
    input_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
) -> torch.Tensor:
    use_ops = _should_use_torch_op(inputs)
    if use_ops:
        if outputs is None:
            shape_n, output_dtype, is_indexed = _get_humming_gemm_output_meta(layer_config, compute_config)
            shape_m = inputs.size(0) * (top_k if is_indexed else 1)
            outputs = inputs.new_empty((shape_m, shape_n), dtype=output_dtype)
        configs = torch.ops.humming.prepare_kernels(inputs, layer_config, compute_config, tuning_config)
    else:
        configs = _prepare_kernels_op(inputs, layer_config, compute_config, tuning_config)

    return launch_kernel(
        configs=configs,
        inputs=inputs,
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        input_scale=input_scale,
        zero_point=zero_point,
        bias=bias,
        outputs=outputs,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        expert_layout=expert_layout,
        locks=locks,
        top_k=top_k,
        valid_shape_m=valid_shape_m,
    )
