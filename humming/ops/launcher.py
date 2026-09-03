import torch

from humming.ops.utils import _prepare_output_arg, _select_output, init_humming_launcher


def register_kernel(cubin_path: str) -> tuple[int, str]:
    init_humming_launcher()
    return torch.ops.humming.register_kernel(cubin_path)


def register_process_input_kernel(cubin_path: str) -> tuple[int, str]:
    init_humming_launcher()
    return torch.ops.humming.register_process_input_kernel(cubin_path)


def get_kernel_smem_size(kernel_id: int) -> int:
    init_humming_launcher()
    return torch.ops.humming.get_kernel_smem_size(kernel_id)


def launch_kernel(
    *,
    configs: torch.Tensor | list[int],
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
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
    assert weight_scale is not None, "weight_scale is required (a lone scale also rides this slot)"
    if isinstance(configs, list):
        configs = torch.tensor(configs, dtype=torch.int64, device="cpu")

    outputs = _prepare_output_arg(inputs, outputs, inputs.dtype)
    if outputs.nelement() > 0 and locks is None:
        locks = inputs.new_empty((0,), dtype=torch.int32)
    launch_args = (
        configs,
        inputs,
        weight,
        weight_scale,
        weight_scale_2,
        input_scale,
        zero_point,
        bias,
        outputs if outputs.nelement() > 0 else None,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        expert_layout,
        locks,
        top_k,
        valid_shape_m,
    )
    if outputs.nelement() == 0:
        returned_outputs = torch.ops.humming.launch_kernel.default(*launch_args)
    else:
        torch.ops.humming.launch_kernel.out(*launch_args)
        returned_outputs = outputs.new_empty((0,))

    return _select_output(outputs, returned_outputs)
