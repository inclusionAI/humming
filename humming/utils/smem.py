import math

from humming.config import (
    ComputeConfig,
    GemmType,
    LayerConfig,
    MmaType,
    TuningConfig,
)

_INT4 = 16


def _align_up(size: int, alignment: int = 128) -> int:
    if size <= 0:
        return 0
    return math.ceil(size / alignment) * alignment


def _struct_size(fields: list[tuple[int, int]], struct_align: int) -> int:
    offset = 0
    for nbytes, align in fields:
        if nbytes <= 0:
            continue
        offset = math.ceil(offset / align) * align
        offset += nbytes
    if offset == 0:
        return 0
    return math.ceil(offset / struct_align) * struct_align


def _stage_storage_bytes(layer_config: LayerConfig, block_shape, is_mxmma: bool) -> int:
    block_m, block_n, block_k = block_shape
    a_bits = layer_config.a_dtype.num_bits
    b_bits = layer_config.b_dtype.num_bits
    bs_bits = (layer_config.bs_dtype or layer_config.c_dtype).num_bits

    has_input_scale = a_bits != 16
    is_group_input_scale = has_input_scale and layer_config.input_scale_group_size > 0
    is_group_or_block_ws = layer_config.is_group_weight_scale or layer_config.is_block_weight_scale
    has_stage_zp = layer_config.has_zero_point and not layer_config.is_channel_weight_scale
    zp_bits = 16 if layer_config.is_fp_zero_point else max(4, _next_pow2(b_bits))

    fields: list[tuple[int, int]] = []
    # a[]: alignas(1024); b[]: alignas(128)
    fields.append((block_m * block_k * a_bits // 8, 1024))
    fields.append((block_n * block_k * b_bits // 8, 128))

    if is_group_input_scale:
        num_groups_a = math.ceil(block_k / layer_config.input_scale_group_size)
        if is_mxmma:
            ng_storage = math.ceil(num_groups_a / 4) * 4
            as_bytes = math.ceil(ng_storage * block_m * bs_bits / 8 / _INT4) * _INT4
        else:
            as_bytes = (num_groups_a * block_m // 4) * _INT4
        fields.append((as_bytes, 128))

    if is_group_or_block_ws and layer_config.weight_scale_group_size > 0:
        num_groups_b = math.ceil(block_k / layer_config.weight_scale_group_size)
        fields.append((num_groups_b * block_n * bs_bits // 8, 128))
        if has_stage_zp:
            fields.append((num_groups_b * block_n * zp_bits // 8, 128))

    return _struct_size(fields, 1024)


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p <<= 1
    return p


def estimate_smem_size_layer(
    layer_config: LayerConfig,
    block_shape: tuple[int, int, int],
    gemm_type: GemmType,
    num_stages: int,
    *,
    warp_shape: tuple[int, int, int] | None = None,
    reduce_overlap_last_stage_only: bool = False,
    use_mbarrier: bool = False,
    use_warp_spec: bool = False,
    num_write_splits: int = 1,
    mma_accum_bits: int = 32,
) -> int:
    block_m, block_n, block_k = block_shape
    is_mxmma = layer_config.mma_type == MmaType.MXMMA
    a_bits = layer_config.a_dtype.num_bits
    bs_bits = (layer_config.bs_dtype or layer_config.c_dtype).num_bits
    zp_bits = 16 if layer_config.is_fp_zero_point else max(4, _next_pow2(layer_config.b_dtype.num_bits))

    stage_bytes = _stage_storage_bytes(layer_config, block_shape, is_mxmma)

    channel_zp = layer_config.has_zero_point and layer_config.is_channel_weight_scale
    channel_zp_bytes = (block_n * zp_bits // 8) if channel_zp else 0
    channel_bs_bytes = (block_n * bs_bits // 8) if layer_config.is_channel_weight_scale else 0
    channel_bs2_bytes = (block_n * 16 // 8) if layer_config.is_channel_weight_scale_2 else 0
    bias_bytes = (block_n * 2) if layer_config.has_bias else 0
    channel_as_bytes = (block_m * 4) if (a_bits != 16 and layer_config.input_scale_group_size == 0) else 0

    struct_a = _struct_size(
        [
            (channel_zp_bytes, 128),
            (channel_bs_bytes, 128),
            (channel_bs2_bytes, 128),
            (bias_bytes, 128),
            (channel_as_bytes, 128),
            (stage_bytes * num_stages, 1024),
        ],
        1024,
    )

    n_warps_k = (block_k // warp_shape[2]) if warp_shape else 1
    warp_reduce = 0
    if warp_shape and n_warps_k >= 2:
        m_warps = block_m // warp_shape[0]
        reduce_buffers = n_warps_k - 1 if n_warps_k <= 4 else n_warps_k // 2
        warp_reduce = m_warps * 16 * block_n * mma_accum_bits // 128 * reduce_buffers
    block_output = block_m * block_n // 2 // 4 // max(1, num_write_splits)
    reduce_bytes = max(warp_reduce, block_output) * _INT4

    struct_b_fields: list[tuple[int, int]] = []
    if reduce_overlap_last_stage_only:
        struct_b_fields.append((channel_zp_bytes, 128))
        struct_b_fields.append((channel_bs_bytes, 128))
        struct_b_fields.append((channel_bs2_bytes, 128))
        struct_b_fields.append((bias_bytes, 128))
        struct_b_fields.append((channel_as_bytes, 128))
        struct_b_fields.append((stage_bytes * (num_stages - 1), 1024))
    struct_b_fields.append((reduce_bytes, 128))
    struct_b = _struct_size(struct_b_fields, 1024)

    union_bytes = _align_up(max(struct_a, struct_b), 1024)

    offset = union_bytes

    def add(nbytes: int, align: int):
        nonlocal offset
        if nbytes <= 0:
            return
        offset = math.ceil(offset / align) * align
        offset += nbytes

    if gemm_type == GemmType.INDEXED:
        add(block_m * 4 * 2, 4)
    elif gemm_type in (GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED):
        add(128, 64)  # tensor_map_buffer[1] (CUtensorMap)
        add(layer_config.num_experts * 4, 4)  # expert_tokens
        add(4, 4)  # total_m_blocks
        if gemm_type == GemmType.GROUPED_CONTIGUOUS:
            add((layer_config.num_experts + 1) * 4, 4)  # expert_offset

    if use_mbarrier:
        add((num_stages + 2) * 8, 128)  # load_mbar
    if use_warp_spec:
        num_math_mbarriers = num_stages + 1
        if reduce_overlap_last_stage_only and num_stages == 2:
            num_math_mbarriers += 1
        add(num_math_mbarriers * 8, 8)  # math_mbar

    return _align_up(offset, 1024)


def estimate_smem_size_config(
    layer_config: LayerConfig,
    compute_config: ComputeConfig,
    tuning_config: TuningConfig,
) -> int:
    gemm_type = compute_config.gemm_type
    if gemm_type is None:
        if layer_config.num_experts:
            raise ValueError("gemm_type must be specified for MoE GEMM")
        gemm_type = GemmType.DENSE

    return estimate_smem_size_layer(
        layer_config,
        tuning_config.block_shape,
        gemm_type,
        tuning_config.num_stages,
        warp_shape=tuning_config.warp_shape,
        reduce_overlap_last_stage_only=tuning_config.reduce_overlap_last_stage_only,
        use_mbarrier=bool(tuning_config.use_mbarrier),
        use_warp_spec=bool(tuning_config.use_warp_spec),
        num_write_splits=tuning_config.num_write_splits,
        mma_accum_bits=16 if compute_config.use_f16_accum else 32,
    )
