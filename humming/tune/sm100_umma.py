import dataclasses
import math

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType, SmemReuseMode
from humming.device import current_device
from humming.tune.base import DeviceHeuristics
from humming.utils.math import round_up
from humming.utils.smem import estimate_smem_size_layer


class Sm100UmmaHeuristics(DeviceHeuristics):
    max_smem_size = 227 * 1024
    sm_version = 100
    b16_allowed_dtypes = [dtypes.float16, dtypes.bfloat16]

    @classmethod
    def _fits_resources(cls, layer_config, block_shape, num_stages, num_ctas_per_sm):
        block_m, block_n, block_k = block_shape
        output_groups = math.ceil(block_n / 128)
        operand_columns = block_k // 2
        stage_columns = output_groups * (num_stages * operand_columns + block_m)
        tmem_columns = 1 << (stage_columns - 1).bit_length()
        if tmem_columns * num_ctas_per_sm > 512:
            # UMMA falls back to two operand buffers when stage-local operands
            # do not fit. Account for that allocation before checking residency.
            buffered_columns = output_groups * (2 * operand_columns + block_m)
            tmem_columns = 1 << (buffered_columns - 1).bit_length()
        if tmem_columns * num_ctas_per_sm > 512:
            return False

        smem_size = estimate_smem_size_layer(
            layer_config,
            block_shape,
            GemmType.DENSE,
            num_stages,
            warp_shape=(block_m, 32, block_k),
            smem_reuse_mode=SmemReuseMode.NONE,
        )
        return smem_size * num_ctas_per_sm <= cls.max_smem_size

    @staticmethod
    def _get_stream_k_work(layer_config, output_tiles, k_iters, block_k, num_stages, resident_ctas):
        # Mirror Scheduler's tail-wave selection and scale/stage alignment.
        stream_tiles = output_tiles
        if output_tiles > resident_ctas:
            stream_tiles = output_tiles % resident_ctas
            if stream_tiles and stream_tiles * 10 <= resident_ctas:
                stream_tiles += resident_ctas
        data_parallel_waves = (output_tiles - stream_tiles) // resident_ctas
        scale_group = max(layer_config.input_scale_group_size, layer_config.weight_scale_group_size, 1)
        blocks_per_group = max(scale_group // block_k, 1)
        alignment = math.lcm(blocks_per_group, num_stages)
        max_slices = 1 << (layer_config.c_dtype.mantissa_bits // 2)
        min_slice_iters = math.ceil((k_iters - 1) / (max_slices - 1))
        slice_iters = max(math.ceil(stream_tiles * k_iters / resident_ctas), min_slice_iters)
        slice_iters = round_up(slice_iters, alignment) if stream_tiles else 0
        return data_parallel_waves * k_iters + slice_iters

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        if gemm_type != GemmType.DENSE or layer_config.num_experts:
            raise ValueError("Sm100UmmaHeuristics currently supports dense GEMM only")
        if use_f16_accum:
            raise ValueError("UMMA requires FP32 accumulation")
        if shape_m <= 0:
            raise ValueError("shape_m must be positive")
        layer_config = dataclasses.replace(layer_config, mma_type=MmaType.UMMA)

        # Balance the final M tile instead of padding a nearly empty second tile
        # to the hardware maximum of 256 rows.
        m_blocks = math.ceil(shape_m / 256)
        block_m = round_up(math.ceil(shape_m / m_blocks), 8)
        shape_n, shape_k = layer_config.shape_n, layer_config.shape_k
        num_sms = current_device.sm_count

        for block_k in (64, 32):
            if shape_k % block_k:
                continue
            k_iters = shape_k // block_k
            for block_n in (256, 128, 64):
                if shape_n % block_n:
                    continue
                block_shape = (block_m, block_n, block_k)
                output_tiles = m_blocks * (shape_n // block_n)
                # Narrow short-K grids need more N tiles because splitting K
                # cannot provide enough work for all SMs.
                can_fill_grid = output_tiles * k_iters >= num_sms * 4
                if block_n > 64 and output_tiles < num_sms and not can_fill_grid:
                    continue

                for num_ctas_per_sm in (2, 1):
                    resident_ctas = num_sms * num_ctas_per_sm
                    # Two resident CTAs hide latency with a short pipeline.
                    # With one CTA, wider N provides more work per stage.
                    output_groups = math.ceil(block_n / 128)
                    target_stages = 5 - output_groups - (num_ctas_per_sm - 1)
                    target_stages = min(target_stages, max(2, k_iters))
                    for num_stages in range(target_stages, 1, -1):
                        if not cls._fits_resources(layer_config, block_shape, num_stages, num_ctas_per_sm):
                            continue
                        stream_k_work = cls._get_stream_k_work(
                            layer_config, output_tiles, k_iters, block_k, num_stages, resident_ctas
                        )
                        data_parallel_work = math.ceil(output_tiles / resident_ctas) * k_iters
                        use_stream_k = (
                            not use_batch_invariant and stream_k_work + 2 * num_stages < data_parallel_work
                        )
                        if num_ctas_per_sm == 2 and output_tiles < resident_ctas:
                            active_ctas = math.ceil(output_tiles * k_iters / max(stream_k_work, 1))
                            if not use_stream_k or active_ctas <= num_sms:
                                continue
                        # For a single short wave, cp.async avoids TMA setup
                        # without sacrificing overlap across persistent tiles.
                        is_short_wave = output_tiles <= resident_ctas and k_iters <= num_stages
                        use_tma = not is_short_wave
                        return {
                            "mma_type": MmaType.UMMA.value,
                            "block_shape": block_shape,
                            "warp_shape": (block_m, 32, block_k),
                            "num_stages": num_stages,
                            "num_ctas_per_sm": num_ctas_per_sm,
                            "umma_num_dequant_warpgroups": 1,
                            "smem_reuse_mode": SmemReuseMode.NONE.value,
                            "use_warp_spec": True,
                            "use_tma": use_tma,
                            "use_tma_a": use_tma,
                            "use_tma_c": use_tma,
                            "use_stream_k": use_stream_k,
                            "use_pdl": False,
                            "raster_group_m": 1,
                        }

        raise ValueError("no resource-feasible dense UMMA tile for this layer")
