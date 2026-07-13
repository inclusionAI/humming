import functools
import math

import torch


def raster_group_m(
    shape_m: int,
    shape_n: int,
    shape_k: int,
    block_m: int,
    block_n: int,
    a_dtype_bits: int,
    b_dtype_bits: int,
    l2_bytes: int,
    num_sms: int,
    *,
    multicast_a: int = 1,
) -> int:
    m_blocks = math.ceil(shape_m / block_m)
    n_blocks = math.ceil(shape_n / (block_n * multicast_a))
    if m_blocks <= 1 or n_blocks <= 1:
        return 1

    bytes_a = a_dtype_bits / 8.0
    bytes_b = b_dtype_bits / 8.0

    if shape_n * shape_k * bytes_b <= 0.7 * l2_bytes:
        return 1

    reserve = min(0.5, 0.12 + 0.28 * bytes_b / bytes_a)
    act_budget = (1.0 - reserve) * l2_bytes

    ub = int(act_budget / (block_m * shape_k * bytes_a))
    lb = math.ceil(num_sms / n_blocks)

    g = min(ub, m_blocks)
    if ub >= lb:
        g = min(max(g, lb), m_blocks)
    return max(1, g)


@functools.lru_cache(maxsize=8)
def _device_l2_sms(device_index: int) -> tuple[int, int]:
    p = torch.cuda.get_device_properties(device_index)
    l2 = getattr(p, "L2_cache_size", None) or getattr(p, "l2_cache_size", 40 * 1024 * 1024)
    return l2, p.multi_processor_count


def raster_group_m_for_config(meta, block_shape, multicast_a: int = 1) -> int:
    block_m, block_n = block_shape[0], block_shape[1]
    l2_bytes, num_sms = _device_l2_sms(torch.cuda.current_device())
    return raster_group_m(
        shape_m=block_m * 4096,
        shape_n=meta.shape_n,
        shape_k=meta.shape_k,
        block_m=block_m,
        block_n=block_n,
        a_dtype_bits=meta.a_dtype.num_bits,
        b_dtype_bits=meta.b_dtype.num_bits,
        l2_bytes=l2_bytes,
        num_sms=num_sms,
        multicast_a=multicast_a,
    )
