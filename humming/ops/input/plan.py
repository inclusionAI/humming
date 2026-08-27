"""Hardware-derived launch planning for process-input kernels."""

import dataclasses
import math

from .enums import ActivationType


@dataclasses.dataclass(frozen=True)
class ProcessInputPlan:
    threads_per_task: int
    values_per_thread: int
    tokens_per_block: int = 1
    use_tile_partition: bool = False
    tiles_per_block: int = 1
    separate_outputs: bool = False
    two_stage: bool = False
    finalize_tokens_per_block: int = 4

    @property
    def threads(self) -> int:
        return self.threads_per_task * self.tokens_per_block


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _powers(limit: int):
    value = 1
    while value <= limit:
        yield value
        value *= 2


def _divisors(value: int):
    result = []
    for divisor in range(1, math.isqrt(value) + 1):
        if value % divisor:
            continue
        result.append(divisor)
        if divisor * divisor != value:
            result.append(value // divisor)
    return sorted(result)


def _finalize_rows(rows: int) -> int:
    return min(8, 1 << (max(rows, 1).bit_length() - 1))


def _valid_values(operation, values: int, block_size: int) -> bool:
    if operation.quantized and values < 2:
        return False
    if operation.hidden_size % values or operation.tile_size % values or block_size % values:
        return False
    return not operation.quantized or values * operation.target_bits % 8 == 0


def _fits_shared_memory(operation, device, block_size: int, threads: int, values: int, tokens: int) -> bool:
    transform = (
        threads * values * tokens if operation.hadamard_block_size > 1 and block_size // values > 32 else 0
    )
    num_warps = threads * tokens // 32
    reduce = 0
    if operation.quantized and operation.dynamic_scale_mode not in ("none", "static"):
        scale_size = operation.quant_group_size
        if operation.dynamic_scale_mode == "token":
            scale_size = operation.hidden_size
        group_reduce = num_warps if scale_size // values > 32 else 0
        token_reduce = num_warps if operation.dynamic_scale_mode == "group_token" and threads > 32 else 0
        reduce = max(group_reduce, token_reduce)
    layout = tokens * (16 + 8 * operation.schedule_width)
    required = 4 * (transform + reduce) + layout
    # Pre-SM100 ptxas limits statically declared shared storage to the default
    # per-block capacity; its larger opt-in capacity applies to dynamic shared.
    limit = device.shared_memory_per_block
    if device.major >= 10:
        limit = getattr(device, "shared_memory_per_block_optin", limit)
    return required <= limit


def _token_candidates(operation, device, block_size: int):
    for values in _powers(min(operation.hidden_size, operation.tile_size, block_size)):
        if not _valid_values(operation, values, block_size):
            continue
        lanes = operation.hidden_size // values
        unit = max(32, block_size // values)
        threads = _ceil_div(lanes, unit) * unit
        if not 32 <= threads <= 1024:
            continue
        if operation.dynamic_scale_mode == "group_token" and (values > 32 or threads & (threads - 1)):
            continue
        tokens = 1
        while tokens <= min(operation.schedule_rows, 1024 // threads, 16):
            transform_lanes = block_size // values
            cta_threads = threads * tokens
            if (transform_lanes <= 32 or cta_threads // transform_lanes <= 16) and _fits_shared_memory(
                operation, device, block_size, threads, values, tokens
            ):
                yield ProcessInputPlan(
                    threads,
                    values,
                    tokens_per_block=tokens,
                    two_stage=operation.dynamic_scale_mode == "token" and values > 32,
                    finalize_tokens_per_block=_finalize_rows(operation.schedule_rows),
                )
            tokens *= 2


def _tile_candidates(operation, device, block_size: int):
    num_tiles = operation.num_tiles
    first_tile_count = max(1, block_size // operation.tile_size)
    divisors = [count for count in _divisors(num_tiles) if count >= first_tile_count]
    pure_hadamard = not operation.quantized and operation.activation_type == ActivationType.None_
    pure_hadamard &= operation.hadamard_block_size > 1
    if pure_hadamard:
        powers = [count for count in _powers(num_tiles) if count >= first_tile_count]
        tile_counts = [count for count in sorted(set(powers + divisors)) if count <= 16]
    else:
        powers = [count for count in _powers(num_tiles) if count >= first_tile_count]
        tile_counts = sorted(set(powers + divisors))

    for tiles_per_block in tile_counts:
        columns = tiles_per_block * operation.tile_size
        for values in _powers(min(operation.tile_size, block_size)):
            if not _valid_values(operation, values, block_size):
                continue
            tile_lanes = operation.tile_size // values
            transform_lanes = block_size // values
            unit = max(32, tile_lanes, transform_lanes)
            threads = _ceil_div(columns // values, unit) * unit
            if threads % 32 or not 32 <= threads <= 1024:
                continue
            if transform_lanes > 32 and threads // transform_lanes > 16:
                continue
            if operation.quantized and operation.dynamic_scale_mode != "static":
                if tile_lanes > 32 and threads // tile_lanes > 16:
                    continue
            if not _fits_shared_memory(operation, device, block_size, threads, values, 1):
                continue
            yield ProcessInputPlan(
                threads,
                values,
                use_tile_partition=True,
                tiles_per_block=tiles_per_block,
                finalize_tokens_per_block=_finalize_rows(operation.schedule_rows),
            )


def _token_score(operation, device, plan: ProcessInputPlan):
    blocks = _ceil_div(operation.schedule_rows, plan.tokens_per_block)
    resident = min(16, device.max_threads_per_multi_processor // plan.threads)
    wave_blocks = device.multi_processor_count * resident
    full_waves, remaining = divmod(blocks, wave_blocks)
    remaining_slots = _ceil_div(remaining, device.multi_processor_count) if remaining else 0
    slots = full_waves * resident + remaining_slots

    transform_stages = operation.hadamard_block_size.bit_length() - 1
    identity = transform_stages == 0 and device.major >= 10
    setup_units = 13 if identity else 6
    thread_work = plan.threads * plan.values_per_thread * (6 + transform_stages)
    work = slots * (thread_work + operation.hidden_size * setup_units)
    if plan.values_per_thread > 16:
        vector_penalty = plan.values_per_thread if identity else plan.values_per_thread + 48
        vector_divisor = 16 if identity else 64
        work = work * vector_penalty // vector_divisor

    token_reduction = (
        operation.dynamic_scale_mode == "token" and operation.activation_type == ActivationType.None_
    )
    multi_warp = operation.hidden_size // plan.values_per_thread > 32
    prefer_single_token = identity and token_reduction and multi_warp
    work *= 2 if prefer_single_token and plan.tokens_per_block > 1 else 1
    if plan.two_stage:
        work *= 2
    if operation.hadamard_block_size > 1:
        return work, blocks, abs(plan.values_per_thread - 8), abs(plan.threads - 512)
    if prefer_single_token:
        return work, abs(plan.values_per_thread.bit_length() - 5), abs(plan.threads - 512)
    return work, -plan.tokens_per_block, abs(plan.threads - 512), plan.values_per_thread


def _direct_group_score(operation, device, plan: ProcessInputPlan):
    vector_floor = 8 if operation.target_bits == 4 else 4
    large_grid = operation.schedule_rows >= 2 * device.multi_processor_count
    full_row_friendly = device.major >= 10 and large_grid
    full_row_friendly &= operation.activation_type == ActivationType.None_
    thread_limit = device.max_threads_per_block if full_row_friendly else 256
    if plan.threads > thread_limit or plan.values_per_thread < vector_floor:
        return (2,)

    blocks = operation.schedule_rows * _ceil_div(operation.num_tiles, plan.tiles_per_block)
    useful = min(operation.hidden_size, plan.tiles_per_block * operation.tile_size)
    if full_row_friendly:
        idle = plan.threads * plan.values_per_thread - useful
        value_error = abs(plan.values_per_thread.bit_length() - 5)
        return 0, blocks, idle, value_error, plan.threads

    warps = blocks * useful // (32 * plan.values_per_thread)
    for costly in (operation.target_bits == 4, operation.activation_type != ActivationType.None_):
        if costly and plan.values_per_thread > 8:
            warps = warps * 8 // plan.values_per_thread

    underfilled = warps < device.multi_processor_count * 32
    preferred_values = 8 if operation.activation_type != ActivationType.None_ else 16
    if underfilled:
        thread_work = operation.tile_size // plan.values_per_thread * 8
    else:
        thread_work = 2048 // plan.values_per_thread
    target_threads = (
        128 if operation.activation_type != ActivationType.None_ else min(256, max(128, thread_work))
    )
    value_error = abs(plan.values_per_thread.bit_length() - preferred_values.bit_length())
    thread_error = abs(plan.threads - target_threads)
    if underfilled:
        return 1, -warps, thread_error, blocks, value_error
    return 0, value_error, thread_error, blocks, plan.threads


def _partitioned_tile_score(operation, device, plan: ProcessInputPlan):
    """Shared-transform and staged-scale score from grid capacity."""
    rows = operation.schedule_rows
    blocks = rows * _ceil_div(operation.num_tiles, plan.tiles_per_block)
    target_threads = 128 if device.major < 10 else (64 if operation.hadamard_block_size > 1 else 32)
    if device.major < 10 and operation.working_set_bytes > device.L2_cache_size:
        target_threads *= 2
    capacity = device.multi_processor_count * max(
        1, device.max_threads_per_multi_processor // target_threads // 4
    )
    row_blocks = max(1, rows * 4)
    occupancy_blocks = min(device.multi_processor_count // 2, rows * min(8, rows * 4))
    target_blocks = min(capacity, max(row_blocks, occupancy_blocks))

    largest = max(operation.tile_size, operation.hadamard_block_size)
    preferred_values = max(16, largest // 32) if largest <= 1024 else 16
    if operation.working_set_bytes > device.L2_cache_size:
        preferred_values = min(preferred_values, 8)

    binary_activation = operation.activation_type in (
        ActivationType.BinarySplit,
        ActivationType.BinaryInterleaved,
    )
    natural_values = 8 if binary_activation or not operation.quantized else 16
    natural_threads = _ceil_div(operation.hidden_size // natural_values, 32) * 32
    raw_unary = operation.activation_type == ActivationType.Unary and not operation.quantized
    binary_lowbit = binary_activation and operation.target_bits == 4
    resident_blocks = 4 if raw_unary or binary_lowbit else 1
    thread_limit = device.max_threads_per_multi_processor // resident_blocks
    thread_limit = min(device.max_threads_per_block, thread_limit)
    large_grid = operation.schedule_rows >= 2 * device.multi_processor_count
    full_row = device.major >= 10 and large_grid
    full_row &= operation.dynamic_scale_mode != "group_token" and natural_threads <= thread_limit
    if full_row:
        target_blocks = rows
        target_threads = natural_threads
        preferred_values = natural_values

    columns = min(operation.hidden_size, plan.tiles_per_block * operation.tile_size)
    idle = plan.threads * plan.values_per_thread - columns
    block_error = abs(blocks - target_blocks) / max(blocks, target_blocks)
    return (
        block_error,
        idle,
        abs(plan.values_per_thread - preferred_values),
        abs(plan.threads - target_threads),
    )


def _raw_tile_score(operation, device, plan: ProcessInputPlan):
    """Rank elementwise plans from occupancy, tail work, and launch work."""
    source_bits = operation.source_bits
    preferred_values = 8 if operation.activation_type != ActivationType.None_ else max(1, 128 // source_bits)
    target_threads = 256 if device.major >= 10 and operation.activation_type == ActivationType.Unary else 128
    columns = min(operation.hidden_size, plan.tiles_per_block * operation.tile_size)
    idle = plan.threads * plan.values_per_thread - columns
    underfilled = operation.schedule_rows < 2 * device.multi_processor_count
    if operation.dynamic_scale_mode == "static" and operation.quantized:
        target_threads = 128 if underfilled else 256
        target_columns = min(operation.hidden_size, 8 * target_threads)
        return (
            abs(columns - target_columns),
            idle,
            abs(plan.values_per_thread - 8),
            abs(plan.threads - target_threads),
        )
    if device.major >= 10 and operation.activation_type == ActivationType.Unary and underfilled:
        blocks = operation.schedule_rows * _ceil_div(operation.num_tiles, plan.tiles_per_block)
        target_blocks = 2 * device.multi_processor_count
        return (
            plan.values_per_thread < 4,
            plan.values_per_thread > 8,
            max(0, target_blocks - blocks),
            idle,
            abs(plan.threads - 128),
            abs(plan.values_per_thread - 8),
            blocks,
        )
    if device.major >= 10 and operation.activation_type == ActivationType.Unary and not underfilled:
        blocks = operation.schedule_rows * _ceil_div(operation.num_tiles, plan.tiles_per_block)
        useful = operation.schedule_rows * operation.hidden_size
        capacity = blocks * plan.threads * plan.values_per_thread
        tail_ratio = (capacity - useful) / useful
        warps = blocks * plan.threads // 32
        warps_per_sm = min(32, device.max_threads_per_multi_processor // 32)
        warp_target = warps_per_sm * device.multi_processor_count
        warp_shortfall = max(0, warp_target - warps) / warp_target
        resident_blocks = 4
        latency_waves = 4
        vector_block_target = latency_waves * resident_blocks * device.multi_processor_count
        vector_block_shortfall = max(0, vector_block_target - blocks) / vector_block_target
        target_threads = min(256, device.max_threads_per_block)
        vector_thread_shortfall = max(0, target_threads - plan.threads) / target_threads
        vector_shortfall = max(vector_block_shortfall, vector_thread_shortfall)
        vector_shortfall = vector_shortfall if plan.values_per_thread > 8 else 0
        thread_limit = device.max_threads_per_multi_processor // resident_blocks
        setup_values = resident_blocks * operation.tile_size
        estimated_work = capacity + capacity / plan.values_per_thread + blocks * setup_values
        work_cost = estimated_work / useful
        execution_threads = target_threads if plan.values_per_thread > 8 else target_threads // 2
        thread_error = abs(plan.threads - execution_threads) / execution_threads
        occupancy_weight = (resident_blocks - 1) / resident_blocks
        work_cost += occupancy_weight * thread_error
        return (
            plan.values_per_thread < 4,
            plan.values_per_thread > 16,
            vector_shortfall,
            warp_shortfall,
            plan.values_per_thread > 8 and plan.threads > target_threads,
            plan.threads > thread_limit,
            work_cost,
            tail_ratio,
            blocks,
        )
    if device.major >= 10 and not underfilled and operation.activation_type != ActivationType.Unary:
        target_threads = device.max_threads_per_block
        if operation.quantized:
            preferred_values = 16
    return idle, abs(plan.values_per_thread - preferred_values), abs(plan.threads - target_threads)


def _pure_hadamard_score(operation, device, plan: ProcessInputPlan):
    """Rank transform tilings; short irregular rows need extra lane parallelism."""
    block_size = operation.hadamard_block_size
    transforms = operation.hidden_size // block_size
    source_bits = operation.source_bits
    fp32 = source_bits == 32
    natural_values = min(4, block_size) if fp32 else max(128 // source_bits, block_size // 32)
    values = natural_values
    lanes = block_size // values
    if fp32:
        tile_capacity = 1 if lanes > 32 else max(1, 128 // lanes)
        tiles = min(transforms, 1 << (tile_capacity.bit_length() - 1))
        while transforms % tiles:
            tiles //= 2
        values = min(values, max(1, block_size * tiles // 32))
        threads = block_size // values * tiles
        return plan.values_per_thread != values, plan.tiles_per_block != tiles, abs(plan.threads - threads)

    underfilled = operation.schedule_rows < 2 * device.multi_processor_count
    irregular_tiles = transforms < 16 and transforms & (transforms - 1)
    if underfilled and irregular_tiles:
        values = max(4, natural_values // 2)
    blocks = operation.schedule_rows * _ceil_div(transforms, plan.tiles_per_block)
    columns = plan.tiles_per_block * block_size
    idle = plan.threads * plan.values_per_thread - columns
    if not underfilled:
        target_tiles = min(transforms, values)
        row_capacity = _ceil_div(transforms, plan.tiles_per_block) * plan.threads * plan.values_per_thread
        capacity_overhead = (row_capacity - operation.hidden_size) / operation.hidden_size
        tile_error = abs(plan.tiles_per_block - target_tiles) / target_tiles
        return (
            plan.values_per_thread != values,
            capacity_overhead + tile_error,
            idle,
            abs(plan.threads - 128),
        )
    target_blocks = _ceil_div(3 * device.multi_processor_count, 2)
    grid_shortfall = max(0, target_blocks - blocks)
    return (
        plan.values_per_thread < 4,
        grid_shortfall,
        idle,
        abs(plan.threads - 128),
        abs(plan.values_per_thread - natural_values),
        blocks,
    )


def select_process_input_plan(operation, device) -> ProcessInputPlan:
    block_size = operation.tile_size
    if operation.hadamard_block_size > 1:
        block_size = operation.hadamard_block_size
    token_plans = tuple(_token_candidates(operation, device, block_size))
    tile_plans = tuple(_tile_candidates(operation, device, block_size))

    token_partition = operation.dynamic_scale_mode == "token"
    if operation.dynamic_scale_mode == "group_token":
        token_partition = bool(token_plans) and operation.schedule_rows <= device.multi_processor_count
    if token_partition:
        assert token_plans
        return min(token_plans, key=lambda plan: _token_score(operation, device, plan))

    assert tile_plans
    pure_hadamard = not operation.quantized and operation.activation_type == ActivationType.None_
    pure_hadamard &= operation.hadamard_block_size > 1
    direct_group = operation.quantized and operation.hadamard_block_size <= 1
    direct_group &= operation.dynamic_scale_mode == "group"
    if pure_hadamard:
        plan = min(tile_plans, key=lambda item: _pure_hadamard_score(operation, device, item))
    elif direct_group:
        plan = min(tile_plans, key=lambda item: _direct_group_score(operation, device, item))
    elif operation.hadamard_block_size > 1 or operation.dynamic_scale_mode == "group_token":
        plan = min(tile_plans, key=lambda item: _partitioned_tile_score(operation, device, item))
    else:
        plan = min(tile_plans, key=lambda item: _raw_tile_score(operation, device, item))
    if operation.dynamic_scale_mode == "group_token":
        plan = dataclasses.replace(plan, finalize_tokens_per_block=4)
    return plan
