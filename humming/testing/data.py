import math

import torch
import torch.nn.functional as F

from humming.config import GemmType


def generate_random_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    std_scale: float = 1.0,
    group_size: int = 0,
    device: torch.device | str | int = "cuda:0",
) -> torch.Tensor:
    assert shape, "shape must not be empty"

    shape_k = shape[-1]
    assert std_scale > 0, "std_scale must be positive"

    group_size = group_size or shape_k
    assert group_size > 0 and shape_k % group_size == 0

    num_groups = shape_k // group_size
    grouped_shape = (*shape[:-1], num_groups, group_size)
    stat_shape = (*shape[:-1], num_groups, 1)

    values = torch.randn(grouped_shape, dtype=dtype, device=device)
    val = 256 / max(group_size, 256)
    group_mean = torch.empty(stat_shape, dtype=dtype, device=values.device).uniform_(-val, val)
    group_std = torch.empty(stat_shape, dtype=dtype, device=values.device).uniform_(0.9, 1.1)
    values = ((values + group_mean) * group_std).view(shape)

    target_std_center = std_scale / (shape_k**0.25)
    target_std = torch.empty((*shape[:-1], 1), dtype=dtype, device=values.device)
    target_std = target_std.uniform_(0.9 * target_std_center, 1.1 * target_std_center)
    values *= target_std / values.std(dim=-1, keepdim=True, correction=0)
    return values


_moe_tensors_type = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]


def generate_random_topk_ids(
    shape_m: int,
    num_experts: int,
    top_k: int,
    balanced: bool = False,
    device: torch.device | str | int = "cuda:0",
) -> torch.Tensor:
    if balanced:
        quotient, remainder = divmod(shape_m * top_k, num_experts)
        remaining = [quotient + (expert_id < remainder) for expert_id in range(num_experts)]
        rows = []
        for token_id in range(shape_m):
            order = sorted(
                range(num_experts),
                key=lambda expert_id: (-remaining[expert_id], (expert_id - token_id) % num_experts),
            )
            row = [expert_id for expert_id in order[:top_k] if remaining[expert_id] > 0]
            assert len(row) == top_k
            for expert_id in row:
                remaining[expert_id] -= 1
            rows.append(row)
        assert not any(remaining)
        return torch.tensor(rows, dtype=torch.int32, device=device)

    scores = torch.randn((shape_m, num_experts), dtype=torch.float32, device=device)
    return scores.topk(top_k, 1).indices.int()


def generate_moe_tensors(
    topk_ids: torch.Tensor,
    num_experts: int,
    gemm_type: GemmType | str,
    block_size_config: int | list[int] | None = None,
    expert_max_tokens: int | None = None,
    expert_alignment: int = 1,
) -> _moe_tensors_type:
    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)
    assert gemm_type != GemmType.DENSE

    shape_m, _ = topk_ids.shape

    if gemm_type in [GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED]:
        expert_num_tokens = topk_ids.view(-1).bincount(minlength=num_experts)
        if gemm_type == GemmType.GROUPED_MASKED:
            assert expert_max_tokens is not None
            assert (expert_num_tokens <= expert_max_tokens).all(), "expert_max_tokens"
            expert_layout = expert_num_tokens.int()
        else:
            expert_num_tokens = (
                (expert_num_tokens + expert_alignment - 1) // expert_alignment * expert_alignment
            )
            expert_first_token_offset = expert_num_tokens.cumsum(0)
            expert_first_token_offset = F.pad(expert_first_token_offset, pad=(1, 0), value=0)
            expert_layout = expert_first_token_offset.long()
        return topk_ids, expert_layout, None, None, None

    assert gemm_type == GemmType.INDEXED
    if isinstance(block_size_config, int):
        block_size = block_size_config
    else:
        assert isinstance(block_size_config, list)
        for i in range(len(block_size_config) // 3):
            if shape_m > block_size_config[i * 3] and shape_m <= block_size_config[i * 3 + 1]:
                block_size = block_size_config[i * 3 + 2]
                break

    # TODO: moe_align_block_size cuda kernel
    part_token_ids_list = []
    expert_id_list = []
    for expert_id in range(num_experts):
        part_token_ids = torch.where(topk_ids.view(-1) == expert_id)[0]
        num_blocks = math.ceil(part_token_ids.size(0) / block_size)
        padded_size = num_blocks * block_size
        pad_size = padded_size - part_token_ids.size(0)
        part_token_ids = F.pad(part_token_ids, pad=(0, pad_size), value=topk_ids.nelement())
        part_token_ids_list.append(part_token_ids)
        expert_id_list += [expert_id] * num_blocks

    sorted_token_ids = torch.cat(part_token_ids_list, dim=0).to(torch.int32)
    expert_ids = torch.tensor(expert_id_list, dtype=torch.int32, device=topk_ids.device)
    num_tokens_padded = torch.tensor(
        sorted_token_ids.size(0),
        dtype=torch.int32,
        device=topk_ids.device,
    )

    return topk_ids, None, sorted_token_ids, expert_ids, num_tokens_padded


def generate_random_moe_tensors(
    shape_m: int,
    num_experts: int,
    top_k: int,
    gemm_type: GemmType | str = "indexed",
    balanced: bool = False,
    block_size_config: int | list[int] | None = None,
    expert_max_tokens: int | None = None,
    expert_alignment: int = 1,
) -> _moe_tensors_type:
    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)
    if gemm_type == GemmType.DENSE:
        return (None,) * 5

    topk_ids = generate_random_topk_ids(shape_m, num_experts, top_k, balanced=balanced)
    return generate_moe_tensors(
        topk_ids,
        num_experts,
        gemm_type,
        block_size_config,
        expert_max_tokens,
        expert_alignment,
    )


def random_fill_tensor(tensor: torch.Tensor) -> None:
    if tensor.dtype == torch.int32:
        min_value = 2**31 * -1
        max_value = 2**31 - 1
        tensor.random_(min_value, max_value)
    elif tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        tensor.normal_(std=0.01)
    else:
        tensor.copy_(tensor.float().random_().to(tensor.dtype))
