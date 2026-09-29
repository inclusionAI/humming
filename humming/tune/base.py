import math

import numpy as np

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import current_device
from humming.utils.math import round_up
from humming.utils.smem import estimate_smem_size_layer


def _estimate_compute_bound_threshold(layer_config: LayerConfig, use_f16_accum: bool) -> float:
    info = current_device
    dtype = str(layer_config.a_dtype)
    if "float16" not in info.tensorcore_tops:
        raise RuntimeError(f"unknown FP16 Tensor Core throughput for sm{info.sm_version}")

    max_tops = info.tensorcore_tops[dtype]
    max_bandwidth = info.memory_bandwidth_gbps
    if info.sm_version in (75, 86, 89) and "float" in dtype and use_f16_accum:
        max_tops *= 2

    weight_nbytes = layer_config.weight_nbytes // (layer_config.num_experts or 1)
    shape_n = layer_config.shape_n
    shape_k = layer_config.shape_k
    left_bias = weight_nbytes / max_bandwidth
    left_factor = shape_k * layer_config.a_dtype.num_bits / 8 / max_bandwidth
    right_factor = shape_n * shape_k * 2 / max_tops
    return left_bias / (right_factor - left_factor) * 1e3


def _count_warps(block_shape: tuple[int, int, int], warp_shape: tuple[int, int, int]) -> int:
    return math.prod(block // warp for block, warp in zip(block_shape, warp_shape, strict=True))


class DeviceHeuristics:
    max_smem_size: int = 0
    b16_allowed_dtypes: list[dtypes.DataType] = []
    b8_allowed_dtypes: list[dtypes.DataType] = []
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 0
    # Resident warps per SM to aim for in memory-bound MoE GEMMs (0 disables).
    moe_occupancy_warps_per_sm: int = 0

    @classmethod
    def should_use_pdl_for_input(cls, layer_config: LayerConfig, shape_m: int) -> bool:
        return False

    @classmethod
    def get_base_config(
        cls,
        a_dtype: dtypes.DataType,
        b_dtype: dtypes.DataType,
        group_size: int,
        use_f16_accum: bool,
        use_fused_e8m0_scale: bool,
        gemm_type: GemmType,
        shape_k: int,
    ):
        raise NotImplementedError

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        compute_bound_min_shape_m = _estimate_compute_bound_threshold(layer_config, use_f16_accum)

        # 1. base config
        group_size = layer_config.input_scale_group_size or layer_config.weight_scale_group_size
        config = cls.get_base_config(
            layer_config.a_dtype,
            layer_config.b_dtype,
            group_size,
            use_f16_accum,
            layer_config.use_fused_e8m0_scale,
            gemm_type,
            layer_config.shape_k,
        )
        block_shape_m, block_shape_n, block_shape_k = config["block_shape"]
        warp_shape_m, warp_shape_n, warp_shape_k = config["warp_shape"]
        num_ctas_per_sm = config.get("num_ctas_per_sm", 1)
        num_stages = config.get("num_stages", 3 if cls.sm_version != 75 else 2)
        num_write_splits = config.get("num_write_splits", 1)
        num_warps_m = block_shape_m // warp_shape_m

        # 2. block_shape_m and warp_shape_m
        if not layer_config.num_experts:
            if shape_m <= block_shape_m:
                block_shape_m = round_up(shape_m, 16)
            else:
                blocks = [math.ceil(shape_m / ((i + 1) * 16)) for i in range(block_shape_m // 16)]
                block_shape_m = np.argmin(blocks).item() * 16 + 16
        else:
            for moe_block_size in [16, 32, 48, 64]:
                if shape_m / layer_config.num_experts / moe_block_size < 0.9:
                    break

            new_shape_m = int(shape_m / layer_config.num_experts / 0.9)
            new_shape_m = max(new_shape_m, 1)
            if block_shape_m == 128:
                if round_up(new_shape_m, 96) < round_up(new_shape_m, 64):
                    block_shape_m = 96
                elif round_up(new_shape_m, 128) < round_up(new_shape_m, 64) * 1.05:
                    block_shape_m = 128
                else:
                    block_shape_m = moe_block_size
            elif new_shape_m >= 64 and new_shape_m < 96:
                block_shape_m = 48
            else:
                block_shape_m = moe_block_size

        assert num_warps_m <= 2
        if num_warps_m == 2 and block_shape_m >= 64:
            block_shape_m = round_up(block_shape_m, 32)
            warp_shape_m = block_shape_m // 2
        elif num_warps_m == 2 and block_shape_m % 32 == 0:
            warp_shape_m = block_shape_m // 2
        else:
            warp_shape_m = block_shape_m
            num_warps_m = 1

        while layer_config.shape_n % block_shape_n != 0:
            assert block_shape_n > 64
            block_shape_n = block_shape_n // 2
            if warp_shape_n > layer_config.a_dtype.num_bits * 4:
                warp_shape_n = warp_shape_n // 2

        num_blocks_n = layer_config.shape_n // block_shape_n
        num_blocks_m = cls.estimate_num_blocks_m(layer_config, shape_m, block_shape_m)

        num_sms = current_device.sm_count
        while num_blocks_n * num_blocks_m * 2 < num_sms * num_ctas_per_sm:
            prefer_m_split = shape_m > block_shape_m >= block_shape_n and num_blocks_m < num_blocks_n
            fitted_block_m = cls._fit_dense_block_m_to_grid(
                (block_shape_m, block_shape_n),
                layer_config,
                shape_m,
                gemm_type,
                num_ctas_per_sm,
            )
            if prefer_m_split and fitted_block_m != block_shape_m:
                break
            if warp_shape_n > layer_config.a_dtype.num_bits * 4 and block_shape_n > 64:
                warp_shape_n = warp_shape_n // 2
                block_shape_n = block_shape_n // 2
                num_blocks_n = num_blocks_n * 2
                continue
            elif block_shape_n > 64:
                block_shape_n = block_shape_n // 2
                num_blocks_n = num_blocks_n * 2
            elif num_ctas_per_sm > 1:
                num_ctas_per_sm = num_ctas_per_sm - 1
                continue
            else:
                break

        if block_shape_n < 256 and warp_shape_k == 1024 // layer_config.a_dtype.num_bits:
            block_shape_k = block_shape_k // 2
            warp_shape_k = warp_shape_k // 2

        num_warps_m = block_shape_m // warp_shape_m
        num_warps_n = block_shape_n // warp_shape_n
        num_warps_k = block_shape_k // warp_shape_k
        num_warps = num_warps_m * num_warps_n * num_warps_k * num_ctas_per_sm

        if num_warps < 8:
            block_shape = (block_shape_m, block_shape_n, block_shape_k)
            smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, num_stages)
            while num_warps < 8:
                if layer_config.shape_k % (block_shape_k * 2) != 0:
                    break
                block_shape_new = (block_shape_m, block_shape_n, block_shape_k * 2)
                smem_size = estimate_smem_size_layer(
                    layer_config,
                    block_shape_new,
                    gemm_type,
                    num_stages,
                )
                if smem_size * num_ctas_per_sm > cls.max_smem_size:
                    break
                block_shape = block_shape_new
                block_shape_k = block_shape_k * 2
                num_warps = num_warps * 2

        if num_warps < 8 and warp_shape_m % 32 == 0:
            warp_shape_m = warp_shape_m // 2
            num_warps = num_warps * 2

        if num_warps < 8 and num_ctas_per_sm == 1 and num_blocks_n * num_blocks_m >= num_sms:
            smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, num_stages)
            if smem_size * 2 <= cls.max_smem_size:
                num_ctas_per_sm = 2

        if shape_m < compute_bound_min_shape_m:
            b_block_bits = block_shape_n * block_shape_k * layer_config.b_dtype.num_bits
            b_load_iters = b_block_bits / 128 / (num_warps * 32 / num_ctas_per_sm)
            if warp_shape_k % (1024 // layer_config.a_dtype.num_bits) == 0 and b_load_iters >= 4:
                warp_shape_k = warp_shape_k // 2
                block_shape_k = block_shape_k // 2

        dense_block_m = cls._fit_dense_block_m_to_grid(
            (block_shape_m, block_shape_n),
            layer_config,
            shape_m,
            gemm_type,
            num_ctas_per_sm,
        )
        use_dense_output_grid = False
        if dense_block_m != block_shape_m:
            num_warps_n = block_shape_n // warp_shape_n
            num_warps_m = 2 if dense_block_m > 32 and dense_block_m % 32 == 0 else 1
            target_k_warps = max(1, 4 // (num_warps_m * num_warps_n))
            min_warp_shape_k = 1024 // layer_config.a_dtype.num_bits
            dense_warp_shape_k = min(
                block_shape_k,
                max(min_warp_shape_k, block_shape_k // target_k_warps),
            )
            num_warps_k = block_shape_k // dense_warp_shape_k
            if num_warps_m * num_warps_n * num_warps_k < 4:
                dense_block_m = block_shape_m
            else:
                block_shape_m = dense_block_m
                num_blocks_m = math.ceil(shape_m / block_shape_m)
                warp_shape_m = block_shape_m // num_warps_m
                warp_shape_k = dense_warp_shape_k
                min_grid_blocks = math.ceil(num_sms * num_ctas_per_sm / 2)
                use_dense_output_grid = num_blocks_n * num_blocks_m >= min_grid_blocks

        max_num_stages = 5 if cls.sm_version == 80 else 3
        for num_stages_new in range(num_stages + 1, max_num_stages + 1):
            block_shape = (block_shape_m, block_shape_n, block_shape_k)
            smem_size = estimate_smem_size_layer(
                layer_config,
                block_shape,
                gemm_type,
                num_stages_new,
            )
            if smem_size * num_ctas_per_sm < cls.max_smem_size:
                num_stages = num_stages_new

        # The compute-bound threshold is per expert, so compare tokens per expert.
        # Wider expert blocks (48+ rows) sit near that threshold and keep their config.
        # FP16 accumulation is left out: dropping K-split warps lengthens each warp's
        # FP16 accumulation, and keeping them leaves too few stages at low token counts.
        # Grouped-masked shape_m counts expert capacity, not routed rows, so the tile
        # count below would include idle experts.
        num_experts = layer_config.num_experts or 0
        is_memory_bound_moe = num_experts > 0 and shape_m / num_experts < compute_bound_min_shape_m
        has_sparse_expert_blocks = block_shape_m <= 32
        uses_default_compute_mode = not (use_batch_invariant or use_f16_accum)
        has_routed_row_count = gemm_type != GemmType.GROUPED_MASKED
        is_supported_moe_case = layer_config.a_dtype.num_bits == 16 and uses_default_compute_mode
        is_supported_moe_case = is_supported_moe_case and has_routed_row_count
        use_moe_occupancy = cls.moe_occupancy_warps_per_sm > 0 and is_supported_moe_case
        use_moe_occupancy = use_moe_occupancy and is_memory_bound_moe and has_sparse_expert_blocks
        uses_moe_occupancy_config = False
        if use_moe_occupancy:
            block_shape = (block_shape_m, block_shape_n, block_shape_k)
            warp_shape = (warp_shape_m, warp_shape_n, warp_shape_k)
            moe_occupancy_config = cls._fit_moe_ctas_per_sm(
                layer_config,
                block_shape,
                warp_shape,
                gemm_type,
                num_tiles=num_blocks_n * num_blocks_m,
                num_sms=num_sms,
                max_num_stages=max_num_stages,
                current_warps_per_sm=_count_warps(block_shape, warp_shape) * num_ctas_per_sm,
            )
            if moe_occupancy_config is not None:
                fitted_block_shape, fitted_warp_shape, num_ctas_per_sm, num_stages = moe_occupancy_config
                block_shape_m, block_shape_n, block_shape_k = fitted_block_shape
                warp_shape_m, warp_shape_n, warp_shape_k = fitted_warp_shape
                num_blocks_n = layer_config.shape_n // block_shape_n
                uses_moe_occupancy_config = True

        use_stream_k = True
        if use_batch_invariant:
            warp_shape_k = 512 // layer_config.a_dtype.num_bits
            block_shape_k = 512 // layer_config.a_dtype.num_bits
            use_stream_k = False

            if cls.sm_version != 75:
                num_warps_m = block_shape_m // warp_shape_m
                warp_shape_m = round_up(warp_shape_m, 16)
                block_shape_m = num_warps_m * warp_shape_m

        while layer_config.shape_k % block_shape_k != 0:
            block_shape_k = block_shape_k // 2
            if use_batch_invariant:
                warp_shape_k = block_shape_k
            else:
                warp_shape_k = 512 // layer_config.a_dtype.num_bits
                assert block_shape_k >= warp_shape_k

        use_stream_k = layer_config.shape_k > 1024 and use_stream_k and not use_dense_output_grid
        # With several CTAs per SM there is enough parallel work; the stream-K fixup only adds cost.
        use_stream_k = use_stream_k and not uses_moe_occupancy_config
        if use_batch_invariant:
            assert not use_stream_k
            assert block_shape_k == warp_shape_k

        if num_ctas_per_sm == 1:
            factor = min(4.5, layer_config.shape_k / (3 * block_shape_k))
            num_sms = min(num_sms, math.ceil(num_blocks_n * num_blocks_m * factor))

        if num_write_splits > 1 and (block_shape_m != warp_shape_m or block_shape_m % 32):
            num_write_splits = 1

        return {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (warp_shape_m, warp_shape_n, warp_shape_k),
            "use_stream_k": use_stream_k,
            "use_f16_accum": use_f16_accum,
            "num_sms": num_sms,
            "num_stages": num_stages,
            "num_ctas_per_sm": num_ctas_per_sm,
            "num_write_splits": num_write_splits,
            "use_pdl": cls.sm_version >= 90,
        }

    @classmethod
    def _fit_moe_ctas_per_sm(
        cls,
        layer_config: LayerConfig,
        block_shape: tuple[int, int, int],
        warp_shape: tuple[int, int, int],
        gemm_type: GemmType,
        num_tiles: int,
        num_sms: int,
        max_num_stages: int,
        current_warps_per_sm: int,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int, int] | None:
        """Trade pipeline depth for resident CTAs in a memory-bound MoE GEMM.

        With few tokens per expert, weight loads are latency-bound and hidden by
        resident warps rather than by a deeper pipeline, so prefer several small
        CTAs per SM over one deep-pipelined CTA. With fewer than three tiles per SM
        the N tile is halved to get more of them. K-split warps are dropped until a
        CTA has at most 4 warps by halving the K tile, down to one 128-byte row of
        activations and then widening the warp K step, unless halving the K tile
        further (less shared memory per stage) fits more CTAs. Warps along M
        dequantize the same weight tile, so they are merged whenever the merged CTA
        still fits twice per SM; otherwise M/N warps are kept and the config must add
        resident warps. Merged CTAs keep 3 stages and the others at most a third of
        their K iterations, as deeper pipelines measured slower. Returns
        (block_shape, warp_shape, num_ctas_per_sm, num_stages) or None.
        """
        max_warps_per_cta = 4
        block_shape_m, block_shape_n, block_shape_k = block_shape
        warp_shape_m, warp_shape_n, warp_shape_k = warp_shape

        is_tile_starved = num_tiles < 3 * num_sms
        can_split_n = block_shape_n >= 256 and layer_config.shape_n % (block_shape_n // 2) == 0
        if is_tile_starved and can_split_n:
            block_shape_n, warp_shape_n = block_shape_n // 2, warp_shape_n // 2
            num_tiles = num_tiles * 2
        # Below two tiles per SM some SMs would hold a single small CTA.
        if num_tiles < 2 * num_sms:
            return None

        num_mn_warps = (block_shape_m // warp_shape_m) * (block_shape_n // warp_shape_n)

        def has_few_warps(block_k, warp_k):
            return num_mn_warps * block_k // warp_k <= max_warps_per_cta

        min_block_shape_k = 1024 // layer_config.a_dtype.num_bits
        fitted_block_shape_k, fitted_warp_shape_k = block_shape_k, warp_shape_k
        while fitted_block_shape_k > fitted_warp_shape_k and not has_few_warps(
            fitted_block_shape_k, fitted_warp_shape_k
        ):
            if fitted_block_shape_k > min_block_shape_k:
                fitted_block_shape_k = fitted_block_shape_k // 2
            else:
                fitted_warp_shape_k = fitted_warp_shape_k * 2
        halved_block_shape_k = block_shape_k
        while halved_block_shape_k > warp_shape_k and not has_few_warps(halved_block_shape_k, warp_shape_k):
            halved_block_shape_k = halved_block_shape_k // 2
        k_shapes = [(fitted_block_shape_k, fitted_warp_shape_k), (halved_block_shape_k, warp_shape_k)]

        def fit_best(merge_m_warps):
            best = None
            for fitted_block_shape_k, fitted_warp_shape_k in k_shapes:
                num_k_iters = layer_config.shape_k // fitted_block_shape_k
                fitted_max_num_stages = 3 if merge_m_warps else min(max_num_stages, max(3, num_k_iters // 3))
                fitted_block_shape = (block_shape_m, block_shape_n, fitted_block_shape_k)
                fitted_warp_shape_m = block_shape_m if merge_m_warps else warp_shape_m
                fitted_warp_shape = (fitted_warp_shape_m, warp_shape_n, fitted_warp_shape_k)
                fitted = cls._fit_moe_cta_count(
                    layer_config,
                    fitted_block_shape,
                    fitted_warp_shape,
                    gemm_type,
                    num_tiles,
                    num_sms,
                    fitted_max_num_stages,
                )
                if fitted is not None and (best is None or fitted[0] > best[2]):
                    best = (fitted_block_shape, fitted_warp_shape, *fitted)
            return best

        if block_shape_m > warp_shape_m:
            merged_config = fit_best(merge_m_warps=True)
            if merged_config is not None:
                return merged_config

        config = fit_best(merge_m_warps=False)
        # Only resident warps hide the latency; the same warps in smaller CTAs do not.
        if config is None or _count_warps(config[0], config[1]) * config[2] <= current_warps_per_sm:
            return None
        return config

    @classmethod
    def _fit_moe_cta_count(
        cls,
        layer_config: LayerConfig,
        block_shape: tuple[int, int, int],
        warp_shape: tuple[int, int, int],
        gemm_type: GemmType,
        num_tiles: int,
        num_sms: int,
        max_num_stages: int,
    ) -> tuple[int, int] | None:
        """Most CTAs per SM (at least 2), then most stages, that fit the warp target,
        the tiles, the register file and shared memory. `__launch_bounds__` makes the
        compiler spill rather than exceed its per-thread register share. Returns
        (num_ctas_per_sm, num_stages) or None.
        """
        num_warps = _count_warps(block_shape, warp_shape)
        # Roughly 96 registers per thread besides the FP32 accumulators of its warp tile.
        registers_per_thread = round_up(96 + warp_shape[0] * warp_shape[1] // 32, 8)
        max_ctas_by_registers = current_device.max_registers_per_sm // (num_warps * 32 * registers_per_thread)
        max_ctas_by_warps = cls.moe_occupancy_warps_per_sm // num_warps
        # Enough CTAs to take every tile at once avoids a tail of lone CTAs.
        max_ctas_by_tiles = math.ceil(num_tiles / num_sms)
        max_ctas_per_sm = min(max_ctas_by_warps, max_ctas_by_tiles, max_ctas_by_registers)
        for num_ctas_per_sm in range(max_ctas_per_sm, 1, -1):
            for num_stages in range(max_num_stages, 2, -1):
                smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, num_stages)
                if smem_size * num_ctas_per_sm <= cls.max_smem_size:
                    return num_ctas_per_sm, num_stages
        return None

    @classmethod
    def estimate_num_blocks_m(cls, layer_config: LayerConfig, shape_m: int, block_shape_m: int):
        if not layer_config.num_experts:
            estimated_num_blocks_m = math.ceil(shape_m / block_shape_m)
        elif shape_m < layer_config.num_experts:
            estimated_num_blocks_m = shape_m
        else:
            estimated_num_blocks_m = layer_config.num_experts

        return estimated_num_blocks_m

    @classmethod
    def _fit_dense_block_m_to_grid(
        cls,
        block_shape: tuple[int, int],
        layer_config: LayerConfig,
        shape_m: int,
        gemm_type: GemmType,
        num_ctas_per_sm: int,
    ) -> int:
        return block_shape[0]

    @classmethod
    def get_configs(
        cls,
        layer_config: LayerConfig,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        a_dtype = layer_config.a_dtype
        if a_dtype.num_bits == 16:
            assert a_dtype in cls.b16_allowed_dtypes
        elif a_dtype.num_bits == 8:
            assert a_dtype in cls.b8_allowed_dtypes
        elif a_dtype.num_bits == 4:
            assert a_dtype in cls.b4_allowed_dtypes
        else:
            raise AssertionError(f"unsupported a_dtype {a_dtype} on sm{cls.sm_version}")

        last_shape_m = 0
        configs: list[list[int | dict]] = []
        last_config_str: str = ""

        if not layer_config.num_experts:
            max_shape_m = 8192
        else:
            max_shape_m = 65536

        shape_m_candidates = [1, 2, 4, 8]
        if cls.sm_version == 90:
            shape_m_candidates += list(range(8, max_shape_m, 8))
        else:
            shape_m_candidates += list(range(16, max_shape_m, 16))

        for shape_m in shape_m_candidates:
            if shape_m > 1024 and shape_m % 16 != 0:
                continue
            if shape_m > 2048 and shape_m % 32 != 0:
                continue
            if shape_m > 4096 and shape_m % 64 != 0:
                continue
            if shape_m > 16384 and shape_m % 128 != 0:
                continue

            config = cls.get_config(
                layer_config=layer_config,
                shape_m=shape_m,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
            )
            config_str = str(config)

            if last_config_str == config_str:
                configs[-1][1] = shape_m
            else:
                configs.append([last_shape_m, shape_m, config])

            last_config_str = config_str
            last_shape_m = shape_m

        configs[-1][1] = 1 << 30

        return configs
