"""Choose process-input launch geometry with deterministic heuristic scores.

Candidates must satisfy the kernel's alignment and shared-memory constraints.
Scores are compared lexicographically: earlier entries have higher priority.
"""

import dataclasses
import functools
import math
from types import SimpleNamespace

from humming import dtypes
from humming.config import (
    ActivationType,
    InputQuantizationMode,
    ProcessInputProblemConfig,
    ProcessInputTuningConfig,
)
from humming.config.mma import UNDOCUMENTED_FP_DTYPES, supports_undocumented_fp_dtypes
from humming.device import DeviceInfo, get_device_index
from humming.utils.math import ceil_div, is_pow_of_two, positive_divisors, powers_of_two_up_to, round_up

_MINIMUM_SM_BY_DTYPE = {
    dtypes.int4: 75,
    dtypes.int8: 75,
    dtypes.float8e4m3: 89,
    dtypes.float8e5m2: 89,
    dtypes.float4e0m3: 100,
    dtypes.float4e2m1: 100,
    dtypes.float8e3m4: 100,
}


def _is_pure_hadamard(problem) -> bool:
    return (
        not problem.should_quantize
        and problem.activation_type == ActivationType.None_
        and problem.hadamard_block_size > 1
    )


def _finalize_tokens_per_block(num_work_rows: int) -> int:
    num_work_rows = max(num_work_rows, 1)
    largest_power_of_two = 1 << (num_work_rows.bit_length() - 1)
    return min(8, largest_power_of_two)


def _valid_values_per_thread(problem, values_per_thread: int, transform_size: int) -> bool:
    if problem.should_quantize and values_per_thread < 2:
        return False
    for size in (problem.hidden_size, problem.tile_size, transform_size):
        if size % values_per_thread:
            return False
    if problem.should_quantize:
        output_bits_per_thread = values_per_thread * problem.target_bits
        if output_bits_per_thread % 8:
            return False

    return True


def _fits_shared_memory(
    problem, device, transform_size: int, threads_per_task: int, values_per_thread: int, tokens_per_block: int
) -> bool:
    threads_per_block = threads_per_task * tokens_per_block
    warps_per_block = threads_per_block // 32
    transform_lanes = transform_size // values_per_thread

    transform_values = 0
    if problem.hadamard_block_size > 1 and transform_lanes > 32:
        transform_values = threads_per_block * values_per_thread

    reduction_values = 0
    dynamic_scale_mode = problem.quant_mode.dynamic_scale_mode
    if dynamic_scale_mode is not None:
        scale_size = problem.quant_group_size
        if dynamic_scale_mode == "token":
            scale_size = problem.hidden_size
        if scale_size // values_per_thread > 32:
            reduction_values = warps_per_block
        if dynamic_scale_mode == "group_token" and threads_per_task > 32:
            reduction_values = warps_per_block

    required_bytes = 4 * max(1, transform_values + reduction_values)

    # Before SM100, static shared storage is limited to the default capacity.
    available_bytes = device.default_smem_size
    if device.sm_major >= 10:
        available_bytes = device.max_smem_size

    return required_bytes <= available_bytes


def _token_tuning_candidates(problem, device, transform_size: int):
    dynamic_scale_mode = problem.quant_mode.dynamic_scale_mode
    max_values_per_thread = min(problem.hidden_size, problem.tile_size, transform_size)
    finalize_tokens = _finalize_tokens_per_block(problem.num_work_rows)

    for values_per_thread in powers_of_two_up_to(max_values_per_thread):
        if not _valid_values_per_thread(problem, values_per_thread, transform_size):
            continue

        transform_lanes = transform_size // values_per_thread
        row_lanes = problem.hidden_size // values_per_thread
        thread_alignment = max(32, transform_lanes)
        threads_per_task = round_up(row_lanes, thread_alignment)

        if not 32 <= threads_per_task <= 1024:
            continue
        if dynamic_scale_mode == "group_token":
            if values_per_thread > 32 or not is_pow_of_two(threads_per_task):
                continue

        max_tokens_per_block = min(problem.num_work_rows, 1024 // threads_per_task, 16)
        for tokens_per_block in powers_of_two_up_to(max_tokens_per_block):
            threads_per_block = threads_per_task * tokens_per_block
            if transform_lanes > 32 and threads_per_block // transform_lanes > 16:
                continue
            if not _fits_shared_memory(
                problem, device, transform_size, threads_per_task, values_per_thread, tokens_per_block
            ):
                continue

            two_stage = dynamic_scale_mode == "token" and values_per_thread > 32
            yield ProcessInputTuningConfig(
                threads_per_task=threads_per_task,
                values_per_thread=values_per_thread,
                tokens_per_block=tokens_per_block,
                two_stage=two_stage,
                finalize_tokens_per_block=finalize_tokens,
            )


def _tile_tuning_candidates(problem, device, transform_size: int):
    min_tiles_per_block = max(1, transform_size // problem.tile_size)
    tile_counts = set(positive_divisors(problem.num_tiles))
    tile_counts.update(powers_of_two_up_to(problem.num_tiles))

    max_values_per_thread = min(problem.tile_size, transform_size)
    finalize_tokens = _finalize_tokens_per_block(problem.num_work_rows)
    pure_hadamard = _is_pure_hadamard(problem)

    for tiles_per_block in sorted(tile_counts):
        if tiles_per_block < min_tiles_per_block:
            continue
        if pure_hadamard and tiles_per_block > 16:
            continue

        requested_columns = tiles_per_block * problem.tile_size
        for values_per_thread in powers_of_two_up_to(max_values_per_thread):
            if not _valid_values_per_thread(problem, values_per_thread, transform_size):
                continue

            tile_lanes = problem.tile_size // values_per_thread
            transform_lanes = transform_size // values_per_thread
            thread_alignment = max(32, tile_lanes, transform_lanes)
            threads_per_task = round_up(requested_columns // values_per_thread, thread_alignment)
            columns_per_task = threads_per_task * values_per_thread

            if columns_per_task % transform_size or columns_per_task % problem.tile_size:
                continue
            if threads_per_task % 32 or not 32 <= threads_per_task <= 1024:
                continue
            if transform_lanes > 32 and threads_per_task // transform_lanes > 16:
                continue
            if problem.quant_mode.has_dynamic_scale:
                if tile_lanes > 32 and threads_per_task // tile_lanes > 16:
                    continue
            if not _fits_shared_memory(
                problem, device, transform_size, threads_per_task, values_per_thread, 1
            ):
                continue

            yield ProcessInputTuningConfig(
                threads_per_task=threads_per_task,
                values_per_thread=values_per_thread,
                use_tile_partition=True,
                finalize_tokens_per_block=finalize_tokens,
            )


def _token_score(problem, device, tuning_config: ProcessInputTuningConfig):
    """Estimate whole-row work and penalize expensive per-thread or staged work."""
    values_per_thread = tuning_config.values_per_thread
    threads_per_block = tuning_config.threads
    num_blocks = ceil_div(problem.num_work_rows, tuning_config.tokens_per_block)

    resident_blocks_per_sm = min(16, device.max_threads_per_sm // threads_per_block)
    blocks_per_wave = device.sm_count * resident_blocks_per_sm
    full_waves, remaining_blocks = divmod(num_blocks, blocks_per_wave)
    block_slots_per_sm = full_waves * resident_blocks_per_sm
    if remaining_blocks:
        block_slots_per_sm += ceil_div(remaining_blocks, device.sm_count)

    transform_stages = problem.hadamard_block_size.bit_length() - 1
    sm100_without_hadamard = device.sm_major >= 10 and transform_stages == 0
    setup_units = 6
    if sm100_without_hadamard:
        setup_units = 13

    thread_work = threads_per_block * values_per_thread * (6 + transform_stages)
    setup_work = problem.hidden_size * setup_units
    estimated_work = block_slots_per_sm * (thread_work + setup_work)

    if values_per_thread > 16:
        vector_penalty = values_per_thread + 48
        vector_divisor = 64
        if sm100_without_hadamard:
            vector_penalty = values_per_thread
            vector_divisor = 16

        estimated_work = estimated_work * vector_penalty // vector_divisor

    token_reduction = problem.quant_mode.dynamic_scale_mode == "token"
    no_activation = problem.activation_type == ActivationType.None_
    multi_warp_row = problem.hidden_size // values_per_thread > 32
    prefer_single_token = sm100_without_hadamard and token_reduction and no_activation and multi_warp_row

    if prefer_single_token and tuning_config.tokens_per_block > 1:
        estimated_work *= 2
    if tuning_config.two_stage:
        estimated_work *= 2

    thread_error = abs(threads_per_block - 512)
    if problem.hadamard_block_size > 1:
        return estimated_work, num_blocks, abs(values_per_thread - 8), thread_error

    if prefer_single_token:
        value_error = abs(values_per_thread.bit_length() - 5)
        return estimated_work, value_error, thread_error

    return estimated_work, -tuning_config.tokens_per_block, thread_error, values_per_thread


def _direct_group_score(problem, device, tuning_config: ProcessInputTuningConfig):
    """Prefer enough parallel group work, then wider vectors and fewer blocks."""
    values_per_thread = tuning_config.values_per_thread
    threads_per_block = tuning_config.threads
    has_activation = problem.activation_type != ActivationType.None_
    large_grid = problem.num_work_rows >= 2 * device.sm_count
    prefer_full_row = device.sm_major >= 10 and large_grid and not has_activation

    min_values_per_thread = 4
    if problem.target_bits == 4:
        min_values_per_thread = 8
    thread_limit = 256
    if prefer_full_row:
        thread_limit = device.max_threads_per_block
    if threads_per_block > thread_limit or values_per_thread < min_values_per_thread:
        return (2,)

    blocks_per_row = ceil_div(problem.hidden_size, tuning_config.columns_per_task)
    num_blocks = problem.num_work_rows * blocks_per_row
    useful_columns = min(problem.hidden_size, tuning_config.columns_per_task)

    if prefer_full_row:
        idle_values = threads_per_block * values_per_thread - useful_columns
        value_error = abs(values_per_thread.bit_length() - 5)
        return 0, num_blocks, idle_values, value_error, threads_per_block

    effective_warps = num_blocks * useful_columns // (32 * values_per_thread)
    if values_per_thread > 8:
        if problem.target_bits == 4:
            effective_warps = effective_warps * 8 // values_per_thread
        if has_activation:
            effective_warps = effective_warps * 8 // values_per_thread

    underfilled = effective_warps < device.sm_count * 32
    thread_work = 2048 // values_per_thread
    if underfilled:
        thread_work = problem.tile_size // values_per_thread * 8

    preferred_values = 16
    target_threads = min(256, max(128, thread_work))
    if has_activation:
        preferred_values = 8
        target_threads = 128

    value_error = abs(values_per_thread.bit_length() - preferred_values.bit_length())
    thread_error = abs(threads_per_block - target_threads)

    if underfilled:
        return 1, -effective_warps, thread_error, num_blocks, value_error

    return 0, value_error, thread_error, num_blocks, threads_per_block


def _partitioned_tile_score(problem, device, tuning_config: ProcessInputTuningConfig):
    """Balance grid capacity and vector width for transforms or staged scales."""
    num_work_rows = problem.num_work_rows
    blocks_per_row = ceil_div(problem.hidden_size, tuning_config.columns_per_task)
    num_blocks = num_work_rows * blocks_per_row
    exceeds_l2 = problem.working_set_bytes > device.l2_cache_size

    target_threads = 128
    if device.sm_major >= 10:
        target_threads = 32
        if problem.hadamard_block_size > 1:
            target_threads = 64
    elif exceeds_l2:
        target_threads *= 2

    blocks_per_sm = max(1, device.max_threads_per_sm // target_threads // 4)
    grid_capacity = device.sm_count * blocks_per_sm
    row_block_target = max(1, num_work_rows * 4)
    occupancy_block_target = min(device.sm_count // 2, num_work_rows * min(8, num_work_rows * 4))
    target_blocks = min(grid_capacity, max(row_block_target, occupancy_block_target))

    preferred_values = 16
    if exceeds_l2:
        preferred_values = 8

    binary_activation = problem.activation_type.is_binary
    natural_values = 16
    if binary_activation or not problem.should_quantize:
        natural_values = 8

    natural_threads = round_up(problem.hidden_size // natural_values, 32)
    raw_unary = problem.activation_type == ActivationType.Unary and not problem.should_quantize
    binary_lowbit = binary_activation and problem.target_bits == 4
    resident_blocks_per_sm = 1
    if raw_unary or binary_lowbit:
        resident_blocks_per_sm = 4

    thread_limit = min(device.max_threads_per_block, device.max_threads_per_sm // resident_blocks_per_sm)
    large_grid = num_work_rows >= 2 * device.sm_count
    prefer_full_row = device.sm_major >= 10 and large_grid
    prefer_full_row = prefer_full_row and problem.quant_mode.dynamic_scale_mode != "group_token"
    prefer_full_row = prefer_full_row and natural_threads <= thread_limit
    if prefer_full_row:
        target_blocks = num_work_rows
        target_threads = natural_threads
        preferred_values = natural_values

    useful_columns = min(problem.hidden_size, tuning_config.columns_per_task)
    idle_values = tuning_config.threads * tuning_config.values_per_thread - useful_columns
    block_error = abs(num_blocks - target_blocks) / max(num_blocks, target_blocks)
    value_error = abs(tuning_config.values_per_thread - preferred_values)
    thread_error = abs(tuning_config.threads - target_threads)
    return block_error, idle_values, value_error, thread_error


def _raw_tile_score(problem, device, tuning_config: ProcessInputTuningConfig):
    """Rank elementwise work by vector width, grid size, and unused capacity."""
    values_per_thread = tuning_config.values_per_thread
    threads_per_block = tuning_config.threads
    useful_columns = min(problem.hidden_size, tuning_config.columns_per_task)
    idle_values = threads_per_block * values_per_thread - useful_columns
    underfilled = problem.num_work_rows < 2 * device.sm_count
    has_activation = problem.activation_type != ActivationType.None_
    sm100_unary = device.sm_major >= 10 and problem.activation_type == ActivationType.Unary

    preferred_values = max(1, 128 // problem.source_bits)
    if has_activation:
        preferred_values = 8

    target_threads = 128
    if problem.quant_mode == InputQuantizationMode.StaticTensor:
        target_threads = 256
        if underfilled:
            target_threads = 128

        target_columns = min(problem.hidden_size, 8 * target_threads)
        return (
            abs(useful_columns - target_columns),
            idle_values,
            abs(values_per_thread - 8),
            abs(threads_per_block - target_threads),
        )

    if sm100_unary:
        # Small grids favor more blocks; large grids favor wider vectors.
        blocks_per_row = ceil_div(problem.hidden_size, tuning_config.columns_per_task)
        num_blocks = problem.num_work_rows * blocks_per_row
        preferred_values = 16
        target_threads = 256
        if underfilled:
            preferred_values = 8
            target_threads = 128

        grid_shortfall = max(0, 2 * device.sm_count - num_blocks)
        value_error = abs(values_per_thread - preferred_values)
        thread_error = abs(threads_per_block - target_threads)
        return grid_shortfall, idle_values, value_error, thread_error, num_blocks

    if device.sm_major >= 10 and not underfilled:
        target_threads = device.max_threads_per_block
        if problem.should_quantize:
            preferred_values = 16

    return idle_values, abs(values_per_thread - preferred_values), abs(threads_per_block - target_threads)


def _pure_hadamard_score(problem, device, tuning_config: ProcessInputTuningConfig):
    """Balance lanes per transform and transforms per block without quantization."""
    transform_size = problem.hadamard_block_size
    transforms_per_row = problem.hidden_size // transform_size
    tiles_per_block = tuning_config.columns_per_task // problem.tile_size
    values_per_thread = tuning_config.values_per_thread
    threads_per_block = tuning_config.threads

    if problem.source_bits == 32:
        preferred_values = min(4, transform_size)
        transform_lanes = transform_size // preferred_values
        tile_capacity = 1
        if transform_lanes <= 32:
            tile_capacity = max(1, 128 // transform_lanes)

        largest_power_of_two = 1 << (tile_capacity.bit_length() - 1)
        target_tiles = min(transforms_per_row, largest_power_of_two)
        while transforms_per_row % target_tiles:
            target_tiles //= 2

        preferred_values = min(preferred_values, max(1, transform_size * target_tiles // 32))
        target_threads = transform_size // preferred_values * target_tiles
        return (
            values_per_thread != preferred_values,
            tiles_per_block != target_tiles,
            abs(threads_per_block - target_threads),
        )

    natural_values = max(128 // problem.source_bits, transform_size // 32)
    preferred_values = natural_values
    underfilled = problem.num_work_rows < 2 * device.sm_count
    irregular_tiles = transforms_per_row < 16 and not is_pow_of_two(transforms_per_row)
    if underfilled and irregular_tiles:
        preferred_values = max(4, natural_values // 2)

    blocks_per_row = ceil_div(transforms_per_row, tiles_per_block)
    num_blocks = problem.num_work_rows * blocks_per_row
    useful_columns = tiles_per_block * transform_size
    idle_values = threads_per_block * values_per_thread - useful_columns

    if not underfilled:
        target_tiles = min(transforms_per_row, preferred_values)
        row_capacity = blocks_per_row * threads_per_block * values_per_thread
        capacity_overhead = (row_capacity - problem.hidden_size) / problem.hidden_size
        tile_error = abs(tiles_per_block - target_tiles) / target_tiles
        return (
            values_per_thread != preferred_values,
            capacity_overhead + tile_error,
            idle_values,
            abs(threads_per_block - 128),
        )

    target_blocks = ceil_div(3 * device.sm_count, 2)
    grid_shortfall = max(0, target_blocks - num_blocks)
    return (
        values_per_thread < 4,
        grid_shortfall,
        idle_values,
        abs(threads_per_block - 128),
        abs(values_per_thread - natural_values),
        num_blocks,
    )


def _select_tuning_config(problem, device) -> ProcessInputTuningConfig:
    transform_size = problem.tile_size
    if problem.hadamard_block_size > 1:
        transform_size = problem.hadamard_block_size

    dynamic_scale_mode = problem.quant_mode.dynamic_scale_mode
    if dynamic_scale_mode in ("token", "group_token"):
        token_candidates = tuple(_token_tuning_candidates(problem, device, transform_size))
        use_token_partition = dynamic_scale_mode == "token"
        if dynamic_scale_mode == "group_token":
            use_token_partition = bool(token_candidates) and problem.num_work_rows <= device.sm_count

        if use_token_partition:
            assert token_candidates
            score = functools.partial(_token_score, problem, device)
            return min(token_candidates, key=score)

    tile_candidates = tuple(_tile_tuning_candidates(problem, device, transform_size))
    assert tile_candidates

    if _is_pure_hadamard(problem):
        score = _pure_hadamard_score
    elif dynamic_scale_mode == "group" and problem.hadamard_block_size <= 1:
        score = _direct_group_score
    elif problem.hadamard_block_size > 1 or dynamic_scale_mode == "group_token":
        score = _partitioned_tile_score
    else:
        score = _raw_tile_score

    tuning_config = min(tile_candidates, key=functools.partial(score, problem, device))
    if dynamic_scale_mode == "group_token":
        tuning_config = dataclasses.replace(tuning_config, finalize_tokens_per_block=4)

    return tuning_config


def _working_set_bytes(config: ProcessInputProblemConfig, num_rows: int) -> int:
    source_bits = config.input_dtype.num_bits
    target_bits = source_bits
    if config.quant_dtype is not None:
        target_bits = config.quant_dtype.num_bits

    num_output_rows = num_rows * config.scatter_width
    input_bits = num_rows * config.input_row_size * source_bits
    output_bits = num_output_rows * config.hidden_size * target_bits
    working_set_bytes = (input_bits + output_bits) // 8

    if config.quant_mode.has_group_scale:
        scale_shape = config.get_group_scale_shape(num_output_rows)
        working_set_bytes += math.prod(scale_shape) * 4
    if config.quant_mode.has_token_scale:
        working_set_bytes += num_output_rows * 4

    return working_set_bytes


@functools.lru_cache(maxsize=1024)
def _tuning_config_for_device(config: ProcessInputProblemConfig, shape_m: int, device_index: int):
    device = DeviceInfo(device_index)
    quant_mode = config.quant_mode
    minimum_sm = _MINIMUM_SM_BY_DTYPE.get(config.quant_dtype)

    if config.quant_dtype is not None and minimum_sm is None:
        raise ValueError(f"unsupported quant_dtype: {config.quant_dtype}")
    if minimum_sm is not None and device.sm_version < minimum_sm:
        raise RuntimeError(f"{config.quant_dtype} output requires SM{minimum_sm} or newer")
    is_undocumented_fp_supported = supports_undocumented_fp_dtypes(device.sm_version)
    if config.quant_dtype in UNDOCUMENTED_FP_DTYPES and not is_undocumented_fp_supported:
        raise RuntimeError(f"{config.quant_dtype} output is not supported on SM{device.sm_version}")

    num_rows = max(shape_m, 1)
    tile_size = min(config.hidden_size & -config.hidden_size, 256)
    if quant_mode.has_group_scale:
        tile_size = config.quant_group_size
    elif not quant_mode.should_quantize and config.hadamard_block_size > 1:
        tile_size = config.hadamard_block_size

    # Candidate enumeration must satisfy the kernel's group/transform boundaries.
    for size in (tile_size, config.hadamard_block_size):
        assert 0 < size <= 512 and is_pow_of_two(size)
        assert config.hidden_size % size == 0
    if quant_mode.has_group_scale:
        assert config.quant_group_size >= 2

    num_output_rows = num_rows * config.scatter_width
    separate_row_limit = (device.sm_count + 1) // 2
    if quant_mode.has_token_scale:
        separate_row_limit = device.sm_count

    separate_outputs = (
        config.scatter_width > 1
        and config.activation_type == ActivationType.None_
        and config.hadamard_block_size == 1
        and num_output_rows <= separate_row_limit
    )
    num_work_rows = num_rows
    if separate_outputs:
        num_work_rows = num_output_rows

    target_bits = config.input_dtype.num_bits
    if config.quant_dtype is not None:
        target_bits = config.quant_dtype.num_bits

    problem = SimpleNamespace(
        **config.to_dict(),
        should_quantize=quant_mode.should_quantize,
        tile_size=tile_size,
        num_tiles=config.hidden_size // tile_size,
        source_bits=config.input_dtype.num_bits,
        target_bits=target_bits,
        num_work_rows=num_work_rows,
        working_set_bytes=_working_set_bytes(config, num_rows),
    )
    tuning_config = _select_tuning_config(problem, device)
    return dataclasses.replace(tuning_config, separate_outputs=separate_outputs)


@functools.lru_cache(maxsize=128)
def _tuning_intervals_for_device(config: ProcessInputProblemConfig, device_index: int, use_pdl: bool):
    device = DeviceInfo(device_index)
    use_pdl = use_pdl and device.sm_major >= 9
    tuning_configs_by_shape_m = {}

    def get_tuning_config(shape_m: int):
        if shape_m not in tuning_configs_by_shape_m:
            tuning_config = _tuning_config_for_device(config, shape_m, device_index)
            tuning_configs_by_shape_m[shape_m] = dataclasses.replace(tuning_config, use_pdl=use_pdl)
        return tuning_configs_by_shape_m[shape_m]

    def partition(min_shape_m: int, max_shape_m: int):
        # Launcher intervals are left-open: min_shape_m < shape_m <= max_shape_m.
        if min_shape_m >= max_shape_m:
            return []

        first_shape_m = min_shape_m + 1
        distance = max_shape_m - first_shape_m
        probe_rows = {
            first_shape_m,
            max_shape_m,
            first_shape_m + distance // 4,
            first_shape_m + distance // 2,
            first_shape_m + 3 * distance // 4,
        }
        probe_configs = {get_tuning_config(shape_m) for shape_m in probe_rows}
        if len(probe_configs) == 1:
            tuning_config = probe_configs.pop()
            return [(min_shape_m, max_shape_m, tuning_config)]

        if distance <= 16:
            intervals = []
            for shape_m in range(first_shape_m, max_shape_m + 1):
                intervals.append((shape_m - 1, shape_m, get_tuning_config(shape_m)))
            return intervals

        middle_shape_m = (first_shape_m + max_shape_m) // 2
        left_intervals = partition(min_shape_m, middle_shape_m)
        right_intervals = partition(middle_shape_m, max_shape_m)
        return left_intervals + right_intervals

    # Small grids are sensitive to individual rows, so enumerate them exactly.
    small_shape_m_limit = max(1, 4 * device.sm_count)
    tuning_intervals = []
    for shape_m in range(1, small_shape_m_limit + 1):
        tuning_intervals.append((shape_m - 1, shape_m, get_tuning_config(shape_m)))

    # Probe large ranges separately on each side of the estimated L2 boundary.
    bytes_per_row = _working_set_bytes(config, 1)
    l2_row_capacity = max(1, device.l2_cache_size // bytes_per_row)
    maximum_shape_m = 1 << 30
    interval_boundaries = {small_shape_m_limit, maximum_shape_m}
    for shape_m in range(l2_row_capacity - 2, l2_row_capacity + 3):
        if small_shape_m_limit < shape_m < maximum_shape_m:
            interval_boundaries.add(shape_m)

    min_shape_m = small_shape_m_limit
    for max_shape_m in sorted(interval_boundaries):
        if max_shape_m <= min_shape_m:
            continue

        tuning_intervals.extend(partition(min_shape_m, max_shape_m))
        min_shape_m = max_shape_m

    merged_intervals = []
    for min_shape_m, max_shape_m, tuning_config in tuning_intervals:
        if merged_intervals:
            previous_min_shape_m, previous_max_shape_m, previous_config = merged_intervals[-1]
            if previous_max_shape_m == min_shape_m and previous_config == tuning_config:
                merged_intervals[-1] = (previous_min_shape_m, max_shape_m, tuning_config)
                continue

        merged_intervals.append((min_shape_m, max_shape_m, tuning_config))

    return merged_intervals


def get_process_input_tuning_intervals(config: ProcessInputProblemConfig, use_pdl: bool = False):
    """Return (min_shape_m, max_shape_m, tuning_config) with min < shape_m <= max."""
    device_index = get_device_index()
    return _tuning_intervals_for_device(config, device_index, use_pdl)
