import math

from humming.device import current_device


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


def raster_group_m_for_config(layer_config, block_shape, multicast_a: int = 1) -> int:
    block_m, block_n = block_shape[0], block_shape[1]
    return raster_group_m(
        shape_m=block_m * 4096,
        shape_n=layer_config.shape_n,
        shape_k=layer_config.shape_k,
        block_m=block_m,
        block_n=block_n,
        a_dtype_bits=layer_config.a_dtype.num_bits,
        b_dtype_bits=layer_config.b_dtype.num_bits,
        l2_bytes=current_device.l2_cache_size,
        num_sms=current_device.sm_count,
        multicast_a=multicast_a,
    )
