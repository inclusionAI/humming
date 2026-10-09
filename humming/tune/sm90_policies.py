import dataclasses
import math
from typing import Literal

import numpy as np
import torch

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.config.mma import get_default_mma_type
from humming.device import current_device
from humming.tune.candidate import (
    CandidateAnalysis,
    ScheduleCandidate,
    TuningDecision,
    TuningProblem,
    analyze_candidate,
    fit_pipeline_stages,
)


@dataclasses.dataclass(frozen=True, slots=True)
class Sm90CandidatePolicy:
    max_indexed_threads_for_two_ctas: int = 256
    max_indexed_threads_for_three_ctas: int = 256


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class _IndexedOption:
    candidate: ScheduleCandidate
    transform: Literal["base", "half_k", "split_n_widen_k"]
    priority: int
    parent: "_IndexedOption | None" = None


def calc_sm90_num_block_list(
    layer_config: LayerConfig,
    shape_m: int,
    max_block_m: int,
) -> list[int]:
    num_blocks_list = []
    if not layer_config.num_experts:
        for block_m in range(8, max_block_m + 1, 8):
            num_blocks_list.append(math.ceil(shape_m / block_m))
    else:
        random_state = np.random.RandomState(seed=0)
        samples = random_state.randint(0, layer_config.num_experts, size=shape_m)
        counts = np.bincount(samples)
        for block_m in range(8, max_block_m + 1, 8):
            num_blocks = int(np.ceil(counts * 1.1 / block_m).sum().item())
            num_blocks_list.append(num_blocks)

    for index, block_m in enumerate(range(8, max_block_m + 1, 8)):
        if layer_config.a_dtype == dtypes.int8 and block_m % 16 == 8 and block_m > 32:
            num_blocks_list[index] = 1000000

    return num_blocks_list


def _select_sm90_block_m(
    layer_config: LayerConfig,
    shape_m: int,
    max_block_m: int,
) -> int:
    num_blocks_list = calc_sm90_num_block_list(
        layer_config,
        shape_m,
        max_block_m,
    )
    return np.argmin(num_blocks_list).item() * 8 + 8


def build_sm90_seed_config(problem: TuningProblem) -> dict:
    """Build the sparse seed config shared by legacy and indexed-A16 paths."""
    layer_config = problem.layer_config
    tune_indexed_a16 = (
        problem.gemm_type == GemmType.INDEXED
        and layer_config.a_dtype.num_bits == 16
        and not problem.use_batch_invariant
    )
    if layer_config.use_packed_k_layout:
        max_block_m = 128
    elif problem.use_f16_accum:
        max_block_m = 256
    else:
        max_block_m = 176

    if tune_indexed_a16:
        # Bound padding when only a few routed rows land on each expert.
        tokens_per_expert = problem.shape_m / layer_config.num_experts
        first_threshold = 1.01 if layer_config.b_dtype.num_bits == 4 else 0.7
        moe_block_size_configs = (
            (8, first_threshold),
            (16, 0.7),
            (24, 0.8),
            (32, 0.9),
            (48, 0.9),
            (64, 0.9),
        )
        for block_shape_m, threshold in moe_block_size_configs:
            if tokens_per_expert / block_shape_m < threshold:
                break
    else:
        block_shape_m = _select_sm90_block_m(
            layer_config,
            problem.shape_m,
            max_block_m,
        )
    warp_shape_n = 32
    warp_shape_k = 1024 // layer_config.a_dtype.num_bits

    # Long-K layers need more routed rows before wider N tiles pay off.
    wide_tile_min_shape_m = 64 if layer_config.shape_k > 4096 else 16
    use_wide_indexed_tile = (
        tune_indexed_a16 and block_shape_m <= 64 and problem.shape_m >= wide_tile_min_shape_m
    )
    if use_wide_indexed_tile:
        warp_shape_n = 64
        # N=512 spills its accumulator at two-CTA residency from M=48 onward.
        if layer_config.shape_k <= 512 and layer_config.shape_n >= 2048 and block_shape_m < 48:
            block_shape_n = 512
            block_shape_k = 64
        else:
            block_shape_n = 256
            block_shape_k = 128
    elif layer_config.shape_n <= 4096 and not problem.use_batch_invariant and block_shape_m <= 64:
        block_shape_n = 128
        block_shape_k = warp_shape_k * 2
        if block_shape_m <= 32:
            block_shape_k = block_shape_k * 2
        if block_shape_k > 256:
            block_shape_k = block_shape_k // 2
            warp_shape_k = warp_shape_k // 2

        while layer_config.shape_k % block_shape_k != 0:
            block_shape_k = block_shape_k // 2
    else:
        block_shape_n = 256
        block_shape_k = warp_shape_k
        if block_shape_m <= 32 and layer_config.b_dtype.num_bits <= 6:
            block_shape_k = block_shape_k * 2
        elif block_shape_m <= 32:
            warp_shape_k = warp_shape_k // 2

    min_warp_shape_n = 32 if layer_config.a_dtype.num_bits == 16 else 16
    # Keep a complete four-warp WGMMA group while fitting output width.
    while layer_config.shape_n % block_shape_n != 0:
        block_shape_n //= 2
        assert block_shape_n >= min_warp_shape_n * 4
    warp_shape_n = min(warp_shape_n, block_shape_n // 4)

    # Earlier shape fitting can reduce block K below the initial warp K.
    warp_shape_k = min(warp_shape_k, block_shape_k)
    while layer_config.shape_k % block_shape_k != 0:
        block_shape_k = block_shape_k // 2
        warp_shape_k = min(warp_shape_k, block_shape_k)
        assert block_shape_k >= warp_shape_k

    if layer_config.use_packed_k_layout:
        warp_shape_k = 128
        block_shape_k = max(block_shape_k, warp_shape_k)

    if problem.gemm_type == GemmType.INDEXED and layer_config.use_packed_k_layout:
        while block_shape_n // warp_shape_n * (block_shape_k // warp_shape_k) > 8:
            block_shape_k //= 2

    dense_small_fp4 = (
        problem.gemm_type == GemmType.DENSE
        and layer_config.a_dtype.num_bits == 16
        and layer_config.b_dtype.num_bits == 4
        and problem.shape_m <= 128
        and layer_config.shape_n % 128 == 0
        and layer_config.shape_k % 64 == 0
    )
    if dense_small_fp4:
        block_shape_n = 128
        block_shape_k = 64
        warp_shape_n = 32
        warp_shape_k = 64
    if problem.use_batch_invariant:
        # Keep one K partition and the same reduction tile for every batch size.
        batch_invariant_k = 128 if layer_config.use_packed_k_layout else 1024 // layer_config.a_dtype.num_bits
        block_shape_k = min(batch_invariant_k, layer_config.shape_k & -layer_config.shape_k)
        warp_shape_k = block_shape_k

    config = {
        "block_shape": (block_shape_m, block_shape_n, block_shape_k),
        "warp_shape": (block_shape_m, warp_shape_n, warp_shape_k),
        "use_stream_k": not problem.use_batch_invariant,
        "use_f16_accum": problem.use_f16_accum,
        "num_stages": 4,
    }

    if problem.gemm_type != GemmType.INDEXED:
        config["use_warp_spec"] = True
        config["use_tma"] = True
        if dense_small_fp4:
            config["num_ctas_per_sm"] = 2

        if (
            layer_config.shape_n % (block_shape_n * 2) == 0
            and problem.shape_m / block_shape_m >= 4
            and problem.gemm_type == GemmType.DENSE
        ):
            config["multi_cast_size_a"] = 2

    return config


def select_grouped_scale(
    problem: TuningProblem,
) -> TuningDecision:
    layer_config = problem.layer_config
    if problem.use_f16_accum:
        max_block_m = 256
    elif layer_config.input_scale_group_size > 0:
        max_block_m = 160
    elif layer_config.weight_scale_group_size < 128:
        max_block_m = 192
    else:
        max_block_m = 200
    block_shape_m = _select_sm90_block_m(
        layer_config,
        problem.shape_m,
        max_block_m,
    )
    if problem.use_batch_invariant:
        block_ks = (min(128, layer_config.shape_k & -layer_config.shape_k),)
    else:
        block_ks = (256, 128, 64) if block_shape_m <= 32 else (128, 64)
    use_multicast = problem.gemm_type == GemmType.DENSE and problem.shape_m / block_shape_m >= 4

    candidates = []
    # Candidate order records measured preference; legality supplies fallbacks.
    for block_shape_n, warp_shape_n in ((128, 32), (64, 16)):
        for block_shape_k in block_ks:
            multicast_values = (True, False) if use_multicast else (False,)
            for multicast in multicast_values:
                config = {
                    "block_shape": (
                        block_shape_m,
                        block_shape_n,
                        block_shape_k,
                    ),
                    "warp_shape": (
                        block_shape_m,
                        warp_shape_n,
                        min(128, block_shape_k),
                    ),
                    "use_stream_k": not problem.use_batch_invariant,
                    "use_f16_accum": problem.use_f16_accum,
                    "num_stages": 4,
                }
                if problem.gemm_type != GemmType.INDEXED:
                    config["use_warp_spec"] = True
                    config["use_tma"] = True
                if multicast:
                    config["multi_cast_size_a"] = 2
                candidate = ScheduleCandidate.from_config(
                    "grouped_scale_"
                    f"n{block_shape_n}_k{block_shape_k}_"
                    f"{'multicast' if multicast else 'direct'}",
                    config,
                )
                candidates.append(fit_pipeline_stages(problem, candidate))

    analyses = tuple(analyze_candidate(problem, candidate) for candidate in candidates)
    selected = next(
        (analysis for analysis in analyses if analysis.legal),
        None,
    )
    if selected is None:
        rejected = {analysis.candidate.candidate_id: analysis.rejection_reasons for analysis in analyses}
        raise AssertionError(f"no legal grouped-scale SM90 schedule: {rejected}")

    return TuningDecision(
        problem=problem,
        family="grouped_scale",
        selected=selected.candidate,
        considered=analyses,
        reason="selected the first legal measured-priority candidate",
    )


# Packed MXFP4 x FP8(GS128) Hopper schedule. M-major grouped inputs size their
# M tile per expert; that policy was measured on H200, other SM90 devices use M128.
_W4A8_DEFAULT_TILE_M = 128
_W4A8_MAX_MODELED_EXPERT_ROWS = 4096


def _w4a8_tile_m_for_expert_rows(rows_per_expert: int) -> int:
    if rows_per_expert > _W4A8_MAX_MODELED_EXPERT_ROWS:
        return 176
    num_tiles = max(1, math.ceil(rows_per_expert / _W4A8_DEFAULT_TILE_M))
    target_rows = rows_per_expert + math.sqrt(rows_per_expert)
    alignment = 32 if num_tiles == 1 else 16
    block_m = math.ceil(target_rows / num_tiles / alignment) * alignment
    block_m = min(_W4A8_DEFAULT_TILE_M, max(64, block_m))
    if rows_per_expert < _W4A8_DEFAULT_TILE_M:
        return block_m

    # Only the mean is available here, not the actual per-expert histogram.
    # Approximate random routing with variance equal to the mean, so crossing a
    # tile boundary has a probability rather than assuming perfectly equal rows.
    deviation = math.sqrt(rows_per_expert)

    def expected_work(tile: int) -> float:
        upper = math.ceil((rows_per_expert + 6 * deviation) / tile)
        expected_tiles = sum(
            0.5 * math.erfc((index * tile + 0.5 - rows_per_expert) / (math.sqrt(2) * deviation))
            for index in range(upper)
        )
        # A small per-tile cost accounts for repeated weight loading and setup.
        return expected_tiles * (tile + 16)

    return min((block_m, 128, 144, 160, 176), key=expected_work)


_W4A8_EXPERT_ROW_BOUNDARIES = tuple(
    rows
    for rows in range(1, _W4A8_MAX_MODELED_EXPERT_ROWS + 1)
    if _w4a8_tile_m_for_expert_rows(rows) != _w4a8_tile_m_for_expert_rows(rows + 1)
)


def _is_packed_w4a8_layer(layer_config: LayerConfig) -> bool:
    return (
        layer_config.sm_version == 90
        and layer_config.shape_n % 128 == 0
        and get_default_mma_type(layer_config) == MmaType.WGMMA
        and layer_config.a_dtype == dtypes.float8e4m3
        and layer_config.b_dtype == dtypes.float4e2m1
        and layer_config.as_dtype == dtypes.float32
        and layer_config.bs_dtype == dtypes.float8e8m0
        and layer_config.use_fused_e8m0_scale
        and layer_config.use_packed_k_layout
        and layer_config.input_scale_group_size == 128
        and layer_config.weight_scale_group_size == 32
    )


def _uses_m_major_w4a8_tiles(use_m_major_input_scale: bool, gemm_type: GemmType) -> bool:
    return gemm_type == GemmType.GROUPED_CONTIGUOUS and use_m_major_input_scale


# Smallest M tile where the N16 schedule beats the seed schedule, measured on H200.
_W4A8_MIN_TILE_M = {GemmType.DENSE: 64, GemmType.INDEXED: 64}
_W4A8_DEFAULT_MIN_TILE_M = 128


def _w4a8_n16_tile_m(config: dict, gemm_type: GemmType) -> int | None:
    block_m = min(config["block_shape"][0], _W4A8_DEFAULT_TILE_M)
    min_tile_m = _W4A8_MIN_TILE_M.get(gemm_type, _W4A8_DEFAULT_MIN_TILE_M)
    return block_m if block_m >= min_tile_m else None


def _w4a8_dense_stream_k_max_shape_m(layer_config: LayerConfig, block_m: int) -> int:
    """Largest dense M whose N128 output tiles do not yet fill every SM."""
    num_n_tiles = layer_config.shape_n // 128
    num_m_tiles = math.ceil(current_device.sm_count / num_n_tiles)
    return (num_m_tiles - 1) * block_m


def _set_w4a8_n16_config(
    config: dict,
    layer_config: LayerConfig,
    gemm_type: GemmType,
    shape_m: int,
) -> None:
    """Apply the packed N16 schedule shared by every row-major GEMM type."""
    # Late AS keeps group scales out of the registers live across WGMMA.
    config["wgmma_use_late_as"] = True
    block_m = _w4a8_n16_tile_m(config, gemm_type)
    if block_m is None:
        return

    use_stream_k = False
    if gemm_type == GemmType.DENSE:
        fills_device = shape_m > _w4a8_dense_stream_k_max_shape_m(layer_config, block_m)
        use_stream_k = config.get("use_stream_k", False) and not fills_device
    config.update(
        block_shape=(block_m, 128, 128),
        warp_shape=(block_m, 16, 128),
        num_stages=4,
        wgmma_split_issue_wait=True,
        use_stream_k=use_stream_k,
        multi_cast_size_a=1,
        multi_cast_size_b=1,
    )
    if gemm_type == GemmType.INDEXED:
        config.update(use_warp_spec=False, use_tma=False)


def _w4a8_block_m(layer_config: LayerConfig, shape_m: int) -> int:
    rows_per_expert = (shape_m + layer_config.num_experts - 1) // layer_config.num_experts
    return _w4a8_tile_m_for_expert_rows(rows_per_expert)


def _w4a8_uses_variable_m_tiles() -> bool:
    return "H200" in torch.cuda.get_device_name()


# Row-major grouped-contiguous sizes its M tile per expert too. The generic seed
# sizes tiles from total rows, which leaves seed-schedule holes and M128 tiles that
# split unevenly routed experts. At most this many rows per expert, both scale layouts
# use the seed schedule with an M tile sized for one expert.
_W4A8_SMALL_EXPERT_ROWS = 96
# Without TMA for row-major AS, M176 only pays off while one tile covers the expert.
_W4A8_ROW_MAJOR_MAX_SINGLE_TILE_ROWS = 176


def _w4a8_row_major_tile_m(rows_per_expert: int) -> int | None:
    if rows_per_expert <= _W4A8_SMALL_EXPERT_ROWS:
        return None
    block_m = _w4a8_tile_m_for_expert_rows(rows_per_expert)
    if block_m <= 160:
        return block_m
    return 176 if rows_per_expert <= _W4A8_ROW_MAJOR_MAX_SINGLE_TILE_ROWS else _W4A8_DEFAULT_TILE_M


def _w4a8_small_expert_tile_m(rows_per_expert: int) -> int:
    """Seed-schedule M tile for small experts: one expert plus one standard deviation of routing."""
    return max(8, math.ceil((rows_per_expert + math.sqrt(rows_per_expert)) / 8) * 8)


_W4A8_SMALL_EXPERT_ROW_BOUNDARIES = tuple(
    rows
    for rows in range(1, _W4A8_SMALL_EXPERT_ROWS)
    if _w4a8_small_expert_tile_m(rows) != _w4a8_small_expert_tile_m(rows + 1)
)


def _w4a8_rows_per_expert(layer_config: LayerConfig, shape_m: int) -> int:
    return (shape_m + layer_config.num_experts - 1) // layer_config.num_experts


def _set_w4a8_small_expert_config(config: dict, layer_config: LayerConfig, rows_per_expert: int) -> bool:
    """Keep the seed schedule but size its M tile for one small expert."""
    # The seed sizes its tile from total rows, about twice the rows of one small expert.
    if rows_per_expert > _W4A8_SMALL_EXPERT_ROWS or layer_config.shape_n % 256:
        return False
    block_m = _w4a8_small_expert_tile_m(rows_per_expert)
    config.update(block_shape=(block_m, 256, 128), warp_shape=(block_m, 32, 128), wgmma_use_late_as=True)
    return True


def _uses_row_major_w4a8_tiles(use_m_major_input_scale: bool, gemm_type: GemmType) -> bool:
    return (
        gemm_type == GemmType.GROUPED_CONTIGUOUS
        and not use_m_major_input_scale
        and _w4a8_uses_variable_m_tiles()
    )


def _set_w4a8_row_major_config(config: dict, layer_config: LayerConfig, shape_m: int) -> None:
    config["wgmma_use_late_as"] = True
    rows_per_expert = _w4a8_rows_per_expert(layer_config, shape_m)
    block_m = _w4a8_row_major_tile_m(rows_per_expert)
    if block_m is None:
        _set_w4a8_small_expert_config(config, layer_config, rows_per_expert)
        return
    config.update(
        block_shape=(block_m, 128, 128),
        warp_shape=(block_m, 16, 128),
        num_stages=4,
        use_warp_spec=True,
        use_tma=True,
        use_stream_k=False,
        wgmma_split_issue_wait=block_m <= 160,
        multi_cast_size_a=1,
        multi_cast_size_b=1,
    )
    config.pop("raster_group_m", None)


def _set_w4a8_config(config: dict, block_m: int, shape_k: int = 0) -> None:
    config.update(
        block_shape=(block_m, 128, 128),
        warp_shape=(block_m, 16, 128),
        num_stages=4 if block_m >= 160 or (block_m >= 128 and shape_k > 2048) else 5,
        use_warp_spec=True,
        use_stream_k=False,
        use_packed_k_layout=True,
        wgmma_use_late_as=True,
        # Wider tiles exceed the split issue-wait register budget.
        wgmma_split_issue_wait=block_m <= 144,
        raster_group_m=16,
        multi_cast_size_a=1,
        multi_cast_size_b=1,
    )


def apply_packed_w4a8_config(
    config: dict,
    layer_config: LayerConfig,
    use_m_major_input_scale: bool,
    gemm_type: GemmType,
    shape_m: int,
) -> None:
    """Select the packed MXFP4 x FP8(GS128) schedule for any GEMM type."""
    if not _is_packed_w4a8_layer(layer_config) or config.get("use_f16_accum", False):
        return
    if _uses_row_major_w4a8_tiles(use_m_major_input_scale, gemm_type):
        _set_w4a8_row_major_config(config, layer_config, shape_m)
        return
    if not _uses_m_major_w4a8_tiles(use_m_major_input_scale, gemm_type):
        _set_w4a8_n16_config(config, layer_config, gemm_type, shape_m)
        return

    use_h200_policy = _w4a8_uses_variable_m_tiles()
    if use_h200_policy and _set_w4a8_small_expert_config(
        config, layer_config, _w4a8_rows_per_expert(layer_config, shape_m)
    ):
        return
    block_m = _w4a8_block_m(layer_config, shape_m) if use_h200_policy else _W4A8_DEFAULT_TILE_M
    _set_w4a8_config(config, block_m, layer_config.shape_k if use_h200_policy else 0)


def _w4a8_range_boundaries(
    config: dict,
    layer_config: LayerConfig,
    use_m_major_input_scale: bool,
    gemm_type: GemmType,
) -> tuple[int, ...]:
    if _uses_m_major_w4a8_tiles(use_m_major_input_scale, gemm_type):
        if not _w4a8_uses_variable_m_tiles():
            return ()
        rows = {*_W4A8_SMALL_EXPERT_ROW_BOUNDARIES, *_W4A8_EXPERT_ROW_BOUNDARIES, _W4A8_SMALL_EXPERT_ROWS}
        return tuple(sorted(row * layer_config.num_experts for row in rows))
    if _uses_row_major_w4a8_tiles(use_m_major_input_scale, gemm_type):
        rows = {
            *_W4A8_SMALL_EXPERT_ROW_BOUNDARIES,
            *_W4A8_EXPERT_ROW_BOUNDARIES,
            _W4A8_SMALL_EXPERT_ROWS,
            _W4A8_ROW_MAJOR_MAX_SINGLE_TILE_ROWS,
        }
        return tuple(sorted(row * layer_config.num_experts for row in rows))

    block_m = _w4a8_n16_tile_m(config, gemm_type)
    if gemm_type != GemmType.DENSE or block_m is None:
        return ()
    return (_w4a8_dense_stream_k_max_shape_m(layer_config, block_m),)


def specialize_packed_w4a8_ranges(
    configs: list,
    layer_config: LayerConfig,
    use_m_major_input_scale: bool,
    gemm_type: GemmType,
) -> list:
    """Split shape-M ranges wherever the packed schedule depends on M."""
    if not _is_packed_w4a8_layer(layer_config):
        return configs

    tuned_configs: list = []
    for lower, upper, base_config in configs:
        boundaries = _w4a8_range_boundaries(base_config, layer_config, use_m_major_input_scale, gemm_type)
        cuts = [lower, *(x for x in boundaries if lower < x < upper), upper]
        for interval_lower, interval_upper in zip(cuts, cuts[1:], strict=False):
            config = dict(base_config)
            apply_packed_w4a8_config(config, layer_config, use_m_major_input_scale, gemm_type, interval_upper)
            if tuned_configs and tuned_configs[-1][1] == interval_lower and tuned_configs[-1][2] == config:
                tuned_configs[-1][1] = interval_upper
            else:
                tuned_configs.append([interval_lower, interval_upper, config])
    return tuned_configs


def _indexed_a16_ctas_per_sm(
    problem: TuningProblem,
    analysis: CandidateAnalysis,
    policy: Sm90CandidatePolicy,
) -> int:
    # Thread caps stand in for the measured register launch-bound cliffs.
    resource_limit = 1
    if (
        analysis.num_threads <= policy.max_indexed_threads_for_two_ctas
        and analysis.smem_size * 2 <= problem.device.resident_smem_size
    ):
        resource_limit = 2
    block_shape = analysis.candidate.block_shape
    if (
        problem.layer_config.a_dtype.num_bits == 16
        and problem.layer_config.b_dtype.num_bits == 4
        and block_shape[0] == 8
        and analysis.num_threads <= policy.max_indexed_threads_for_three_ctas
        and analysis.smem_size * 3 <= problem.device.resident_smem_size
    ):
        resource_limit = 3

    resource_limit = min(resource_limit, analysis.thread_smem_cta_limit)
    if problem.device.num_sms is None:
        raise ValueError("indexed-A16 selection requires a device SM count")
    grid_limit = math.ceil(analysis.num_output_tiles / problem.device.num_sms)
    return max(1, min(resource_limit, grid_limit))


def _analyze_indexed_a16_candidate(
    problem: TuningProblem,
    candidate: ScheduleCandidate,
    policy: Sm90CandidatePolicy,
) -> CandidateAnalysis:
    analysis = analyze_candidate(problem, candidate)
    if not analysis.legal:
        return analysis
    candidate = candidate.with_updates(
        num_ctas_per_sm=_indexed_a16_ctas_per_sm(
            problem,
            analysis,
            policy,
        )
    )
    return analyze_candidate(problem, candidate)


def _half_k_candidate(
    problem: TuningProblem,
    source: ScheduleCandidate,
) -> ScheduleCandidate | None:
    block_shape = source.block_shape
    warp_shape = source.warp_shape
    smaller_block_k = block_shape[2] // 2
    scale_groups_align = all(
        not group_size or group_size % smaller_block_k == 0 or smaller_block_k % group_size == 0
        for group_size in (
            problem.layer_config.input_scale_group_size,
            problem.layer_config.weight_scale_group_size,
        )
    )
    if block_shape[2] < warp_shape[2] * 2 or not scale_groups_align:
        return None

    config = source.to_config()
    config.pop("num_ctas_per_sm", None)
    return ScheduleCandidate.from_config(
        "indexed_a16_half_k",
        config,
    ).with_updates(
        block_shape=(*block_shape[:2], smaller_block_k),
        warp_shape=(
            *warp_shape[:2],
            min(warp_shape[2], smaller_block_k),
        ),
    )


def _split_n_widen_k_candidate(
    problem: TuningProblem,
    source: ScheduleCandidate,
    *,
    candidate_id: str,
) -> ScheduleCandidate | None:
    block_shape = source.block_shape
    if not (
        problem.layer_config.a_dtype.num_bits == 16
        and problem.layer_config.b_dtype.num_bits == 4
        and block_shape[1] >= 256
        and block_shape[2] == 64
    ):
        return None

    config = source.to_config()
    config.pop("num_ctas_per_sm", None)
    return ScheduleCandidate.from_config(candidate_id, config).with_updates(
        block_shape=(
            block_shape[0],
            block_shape[1] // 2,
            block_shape[2] * 2,
        ),
    )


def select_indexed_a16(
    problem: TuningProblem,
    policy: Sm90CandidatePolicy,
) -> TuningDecision:
    if problem.device.num_sms is None:
        raise ValueError("indexed-A16 selection requires a device SM count")
    base = fit_pipeline_stages(
        problem,
        ScheduleCandidate.from_config(
            "indexed_a16_base",
            build_sm90_seed_config(problem),
        ),
    )
    base_option = _IndexedOption(base, "base", 0)
    options = [base_option]
    half_k = _half_k_candidate(problem, base)
    half_option = None
    if half_k is not None:
        half_option = _IndexedOption(half_k, "half_k", 1, base_option)
        options.append(half_option)
    for source, candidate_id in (
        (base_option, "indexed_a16_split_n_widen_k_from_base"),
        (half_option, "indexed_a16_split_n_widen_k"),
    ):
        if source is None:
            continue
        split = _split_n_widen_k_candidate(
            problem,
            source.candidate,
            candidate_id=candidate_id,
        )
        if split is not None:
            options.append(_IndexedOption(split, "split_n_widen_k", 2, source))

    analyses = {
        option: _analyze_indexed_a16_candidate(
            problem,
            option.candidate,
            policy,
        )
        for option in options
    }
    base_analysis = analyses[base_option]
    if not base_analysis.legal:
        raise AssertionError(base_analysis.rejection_reasons)
    eligible_reasons = {
        base_option: "selected the base indexed-A16 schedule",
    }

    if half_option is not None:
        half_analysis = analyses[half_option]
        if (
            base_analysis.candidate.num_ctas_per_sm == 1
            and half_analysis.legal
            and half_analysis.candidate.num_ctas_per_sm > base_analysis.candidate.num_ctas_per_sm
        ):
            eligible_reasons[half_option] = "halved K because it increased CTA residency"

    for option in options:
        if option.transform != "split_n_widen_k":
            continue
        assert option.parent is not None
        parent_reason = eligible_reasons.get(option.parent)
        analysis = analyses[option]
        parent_analysis = analyses[option.parent]
        if (
            parent_reason is not None
            and analysis.legal
            and analysis.candidate.num_ctas_per_sm > parent_analysis.candidate.num_ctas_per_sm
            and analysis.waves is not None
            and parent_analysis.waves is not None
            and analysis.waves <= parent_analysis.waves
        ):
            eligible_reasons[option] = f"{parent_reason}; split N and widened K without adding a grid wave"
    selected = max(
        eligible_reasons,
        key=lambda option: option.priority,
    )

    final_candidate = analyses[selected].candidate.with_updates(
        use_stream_k=(
            bool(analyses[selected].candidate.get("use_stream_k", True))
            and analyses[selected].num_output_tiles < problem.device.num_sms
        )
    )
    final_analysis = analyze_candidate(problem, final_candidate)
    if not final_analysis.legal:
        raise AssertionError(final_analysis.rejection_reasons)
    return TuningDecision(
        problem=problem,
        family="indexed_a16",
        selected=final_candidate,
        considered=tuple(final_analysis if option is selected else analyses[option] for option in options),
        reason=eligible_reasons[selected],
    )
