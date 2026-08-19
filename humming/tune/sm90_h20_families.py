"""H20 tuning policies: the seed heuristic and the candidate selectors.

build_h20_seed_config owns the H20 seed heuristic (the former legacy
get_config body) for every construction; the fused-E8M0 8-bit families
additionally run bounded candidate selection on top of the shared schedule
construction, recording the alternatives that were considered and why one
was preferred (guarded by tests/test_sm90_h20_candidate_parity.py).
"""

import math

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.tune.candidate import (
    CandidateAnalysis,
    DeviceProfile,
    ScheduleCandidate,
    TuningDecision,
    TuningProblem,
    analyze_candidate,
)
from humming.utils.smem import estimate_smem_size_layer

_H20_MAX_SMEM_SIZE = 227 * 1024
_MOE_BLOCK_SIZE_THRESHOLDS = ((8, 0.7), (16, 0.8), (32, 0.9), (48, 0.9), (64, 0.9))

# Measured on H20 (DSV4-Pro W4A8, fused-E8M0, grouped input scale): the
# mainloop keeps per-group scale double buffers live and needs ~236 registers
# per thread when unconstrained; a budget of ~170 (128 threads x 3 CTAs) forces
# local-memory spills (34M local loads, 53M spilling requests, 1.4x latency).
_GROUPED_INPUT_SCALE_REGISTER_DEMAND = 232


def make_h20_device_profile(num_sms: int) -> DeviceProfile:
    return DeviceProfile(
        name="NVIDIA H20",
        sm_version=90,
        num_sms=num_sms,
        max_smem_size=_H20_MAX_SMEM_SIZE,
        max_smem_per_sm=_H20_MAX_SMEM_SIZE,
    )


def _fit_block_n(shape_n: int, block_n: int, warp_n: int) -> tuple[int, int]:
    while shape_n % block_n:
        block_n //= 2
        warp_n = min(warp_n, block_n // 4)
    return block_n, warp_n


def _tile_shape(layer_config: LayerConfig, shape_m: int) -> tuple[int, int, int]:
    """Derived (block_m, block_n, warp_n) shared by the scope predicate and
    the schedule construction, so the two cannot drift apart."""
    block_m = _moe_block_shape_m(shape_m, layer_config.num_experts, 64)
    block_n, warp_n = _fit_block_n(layer_config.shape_n, 128, 32)
    return block_m, block_n, warp_n


def _moe_block_shape_m(shape_m: int, num_experts: int, base_block_m: int) -> int:
    for moe_block_size, threshold in _MOE_BLOCK_SIZE_THRESHOLDS:
        if shape_m / num_experts / moe_block_size < threshold:
            break

    new_shape_m = max(int(shape_m / num_experts / 0.9), 1)
    if base_block_m == 128:
        if math.ceil(new_shape_m / 96) * 96 < math.ceil(new_shape_m / 64) * 64:
            return 96
        if math.ceil(new_shape_m / 128) * 128 < math.ceil(new_shape_m / 64) * 64 * 1.05:
            return 128
        return moe_block_size
    if 64 <= new_shape_m < 96:
        return 48
    return moe_block_size


def fused_e8m0_moe_in_scope(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_batch_invariant: bool,
) -> bool:
    """Scope of the migrated family; everything else stays on the legacy path.

    Long-K fused-E8M0 8-bit MoE, all tile sizes: large tiles select a CTA
    residency count, small tiles (block_m <= 32) additionally weigh the
    wide-tile and pipeline-residency alternatives of the legacy long-K
    residency tuner.
    """
    if use_batch_invariant:
        return False
    if gemm_type == GemmType.DENSE:
        return False
    if layer_config.a_dtype.num_bits != 8:
        return False
    if not layer_config.use_fused_e8m0_scale:
        return False
    if not layer_config.num_experts:
        return False
    if layer_config.shape_k <= 1024:
        return False
    input_scale_group_size = layer_config.input_scale_group_size or 0
    if input_scale_group_size and layer_config.shape_k % input_scale_group_size:
        return False
    if layer_config.use_packed_k_layout:
        return False
    block_m, block_n, warp_n = _tile_shape(layer_config, shape_m)
    if warp_n < 16:
        return False
    return True


def _grid_fill_ctas(
    num_blocks: int,
    num_sms: int,
    num_ctas_per_sm: int,
) -> int:
    while num_blocks * 2 < num_sms * num_ctas_per_sm and num_ctas_per_sm > 1:
        num_ctas_per_sm -= 1
    return num_ctas_per_sm


def _dense_block_shape_m(shape_m: int, base_block_m: int) -> int:
    if shape_m <= base_block_m:
        return math.ceil(shape_m / 8) * 8
    blocks = [
        math.ceil(shape_m / ((i + 1) * 8)) for i in range(base_block_m // 8)
    ]
    return min(range(len(blocks)), key=blocks.__getitem__) * 8 + 8


def _fit_dense_block_m_to_output_grid(
    layer_config: LayerConfig,
    shape_m: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_sms: int,
) -> int:
    num_n_tiles = layer_config.shape_n // block_n
    current_m_tiles = math.ceil(shape_m / block_m)

    stream_k_grid_gain = 1.0
    current_output_tiles = current_m_tiles * num_n_tiles
    if layer_config.shape_k <= 1024:
        if current_output_tiles >= math.ceil(num_sms * 0.5):
            return block_m
    else:
        if current_output_tiles >= math.ceil(num_sms * 0.2):
            return block_m
        stream_k_grid_gain = min(4.5, layer_config.shape_k / (12 * block_k))
    target_wave_fraction = 0.8
    if layer_config.shape_k > 1024:
        target_wave_fraction = max(0.5, 1 - layer_config.shape_k / (8 * 1024))
    target_output_tiles = math.ceil(
        num_sms * target_wave_fraction / stream_k_grid_gain
    )
    if current_output_tiles >= target_output_tiles:
        return block_m

    current_padded_rows = current_m_tiles * block_m
    candidates = []
    padding_safe_candidates = []
    min_stream_k_block_m = (
        16 if layer_config.shape_k > 1024 and block_m >= 32 else 8
    )
    for candidate_m in range(8, block_m, 8):
        if (
            layer_config.a_dtype == dtypes.int8
            and candidate_m > 32
            and candidate_m % 16
        ):
            continue
        candidate_m_tiles = math.ceil(shape_m / candidate_m)
        padded_rows = candidate_m_tiles * candidate_m
        if padded_rows > current_padded_rows * 1.05:
            continue
        padding_safe_candidates.append(candidate_m)
        if candidate_m_tiles * num_n_tiles < target_output_tiles:
            continue
        if candidate_m < min_stream_k_block_m:
            continue
        candidates.append(candidate_m)
    if candidates:
        return max(candidates)
    if layer_config.shape_k <= 1024 and padding_safe_candidates:
        return min(padding_safe_candidates)
    if layer_config.shape_k > 1024 and block_m >= 32:
        reuse_candidates = [c for c in padding_safe_candidates if c >= 16]
        if reuse_candidates:
            return max(reuse_candidates)
    return block_m


def _base_schedule(
    problem: TuningProblem,
) -> tuple[dict, int]:
    """Replicate the legacy transform chain for the in-scope construction.

    Returns the config dict (without num_ctas_per_sm) plus the grid-fill CTA
    count that every derived decision below is based on. Order quirks that the
    legacy code depends on are preserved deliberately:
    - num_stages is fitted against the grid-fill CTA count, before any
      register-budget preference is applied;
    - the num_sms trim for single-CTA grids also reads the grid-fill count;
    - the smem estimate for stage fitting uses the legacy call form (no
      warp_shape / accum-width arguments).
    """
    layer_config = problem.layer_config
    shape_m = problem.shape_m
    num_sms_physical = problem.device.num_sms

    block_m, block_n, warp_n = _tile_shape(layer_config, shape_m)
    block_k = 1024 // layer_config.a_dtype.num_bits
    warp_k = block_k

    num_blocks_n = layer_config.shape_n // block_n
    num_blocks_m = (
        shape_m
        if shape_m < layer_config.num_experts
        else layer_config.num_experts
    )
    grid_ctas = _grid_fill_ctas(
        num_blocks_n * num_blocks_m, num_sms_physical, 3
    )

    num_warps = (block_n // warp_n) * (block_k // warp_k) * grid_ctas
    if num_warps == 4:
        warp_k = 512 // layer_config.a_dtype.num_bits
        block_k = warp_k * 2

    if num_warps <= 8 and block_m <= 32:
        num_warps_k = block_k // warp_k
        warp_k = 512 // layer_config.a_dtype.num_bits
        block_k = warp_k * num_warps_k * 2

    if warp_k == block_k and warp_k == 512 // layer_config.a_dtype.num_bits:
        smem_size = estimate_smem_size_layer(
            layer_config,
            (block_m, block_n, block_k * 2),
            problem.gemm_type,
            3,
        )
        if smem_size * grid_ctas < _H20_MAX_SMEM_SIZE:
            block_k *= 2
            warp_k *= 2

    num_stages = 3
    for num_stages_new in (4,):
        smem_size = estimate_smem_size_layer(
            layer_config,
            (block_m, block_n, block_k),
            problem.gemm_type,
            num_stages_new,
        )
        if smem_size * grid_ctas < _H20_MAX_SMEM_SIZE:
            num_stages = num_stages_new

    num_sms = num_sms_physical
    if grid_ctas == 1:
        factor = min(4.5, layer_config.shape_k / (3 * block_k))
        if layer_config.shape_k > 1024:
            factor = min(9, max(factor, layer_config.shape_k / (8 * block_k)))
        num_sms = min(num_sms, math.ceil(num_blocks_n * num_blocks_m * factor))

    while layer_config.shape_k % block_k != 0:
        warp_k = 512 // layer_config.a_dtype.num_bits
        block_k //= 2

    config = {
        "block_shape": (block_m, block_n, block_k),
        "warp_shape": (block_m, warp_n, warp_k),
        "use_stream_k": layer_config.shape_k > 1024,
        "use_f16_accum": problem.use_f16_accum,
        "num_sms": num_sms,
        "num_stages": num_stages,
    }
    return config, grid_ctas


def fused_e8m0_dense_in_scope(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_batch_invariant: bool,
) -> bool:
    """Fused-E8M0 8-bit dense GEMMs; the small-M dense override is
    unreachable here (it requires an integer B dtype), so the candidate set
    is the plain schedule plus the TMA and deep-pipeline alternatives."""
    if use_batch_invariant:
        return False
    if gemm_type != GemmType.DENSE:
        return False
    if layer_config.a_dtype.num_bits != 8 or layer_config.a_dtype.is_integer_type:
        return False
    if not layer_config.use_fused_e8m0_scale:
        return False
    if layer_config.num_experts:
        return False
    if layer_config.b_dtype.is_integer_type and layer_config.b_dtype.num_bits <= 4:
        return False
    input_scale_group_size = layer_config.input_scale_group_size or 0
    if input_scale_group_size and layer_config.shape_k % input_scale_group_size:
        return False
    if layer_config.use_packed_k_layout:
        return False
    block_n, warp_n = _fit_block_n(layer_config.shape_n, 128, 32)
    if warp_n < 16:
        return False
    return True


def _dense_base_schedule(problem: TuningProblem) -> tuple[dict, int, int]:
    """Replicate the legacy dense transform chain for the in-scope
    construction; returns (config without num_stages/tma keys, cta count,
    the pre-reshape warp count the TMA arm keys on)."""
    layer_config = problem.layer_config
    shape_m = problem.shape_m
    num_sms_physical = problem.device.num_sms

    block_n, warp_n = _fit_block_n(layer_config.shape_n, 128, 32)
    block_m = _dense_block_shape_m(shape_m, 128)
    block_k = 1024 // layer_config.a_dtype.num_bits
    warp_k = block_k

    num_blocks_n = layer_config.shape_n // block_n
    num_blocks_m = math.ceil(shape_m / block_m)
    num_ctas_per_sm = _grid_fill_ctas(
        num_blocks_n * num_blocks_m, num_sms_physical, 2
    )

    num_warps = (block_n // warp_n) * (block_k // warp_k) * num_ctas_per_sm
    if num_warps == 4:
        warp_k = 512 // layer_config.a_dtype.num_bits
        block_k = warp_k * 2

    if num_warps <= 8 and block_m <= 32:
        num_warps_k = block_k // warp_k
        warp_k = 512 // layer_config.a_dtype.num_bits
        block_k = warp_k * num_warps_k * 2

    if warp_k == block_k and warp_k == 512 // layer_config.a_dtype.num_bits:
        smem_size = estimate_smem_size_layer(
            layer_config,
            (block_m, block_n, block_k * 2),
            problem.gemm_type,
            3,
        )
        if smem_size * num_ctas_per_sm < _H20_MAX_SMEM_SIZE:
            block_k *= 2
            warp_k *= 2

    num_stages = 3
    for num_stages_new in (4,):
        smem_size = estimate_smem_size_layer(
            layer_config,
            (block_m, block_n, block_k),
            problem.gemm_type,
            num_stages_new,
        )
        if smem_size * num_ctas_per_sm < _H20_MAX_SMEM_SIZE:
            num_stages = num_stages_new

    block_m = _fit_dense_block_m_to_output_grid(
        layer_config, shape_m, block_m, block_n, block_k, num_sms_physical
    )
    num_blocks_m = math.ceil(shape_m / block_m)

    num_sms = num_sms_physical
    if num_ctas_per_sm == 1:
        factor = min(4.5, layer_config.shape_k / (3 * block_k))
        if layer_config.shape_k > 1024:
            factor = min(9, max(factor, layer_config.shape_k / (8 * block_k)))
        num_sms = min(num_sms, math.ceil(num_blocks_n * num_blocks_m * factor))

    while layer_config.shape_k % block_k != 0:
        warp_k = 512 // layer_config.a_dtype.num_bits
        block_k //= 2

    if _register_preference_applies(problem):
        num_ctas_per_sm = min(num_ctas_per_sm, 2)

    config = {
        "block_shape": (block_m, block_n, block_k),
        "warp_shape": (block_m, warp_n, warp_k),
        "use_stream_k": layer_config.shape_k > 1024,
        "use_f16_accum": problem.use_f16_accum,
        "num_sms": num_sms,
        "num_stages": num_stages,
        "num_ctas_per_sm": num_ctas_per_sm,
    }
    return config, num_ctas_per_sm, num_warps


def select_fused_e8m0_dense(problem: TuningProblem) -> TuningDecision:
    base_config, num_ctas_per_sm, num_warps = _dense_base_schedule(problem)
    layer_config = problem.layer_config
    block_m, block_n, block_k = base_config["block_shape"]

    candidates: list[ScheduleCandidate] = []
    tma_candidate = None
    use_dense_tma = (
        block_m >= 48
        and num_ctas_per_sm <= 2
        and num_warps <= 8
        and layer_config.shape_k // block_k >= 24
    )
    if use_dense_tma:
        tma_candidate = ScheduleCandidate.from_config(
            "dense_tma_warp_spec",
            {
                **base_config,
                "num_stages": 3,
                "use_tma": True,
                "use_warp_spec": True,
                "use_mbarrier": True,
            },
        )
        candidates.append(tma_candidate)

    deep_candidate = None
    if base_config["num_stages"] == 4 and block_m <= 32:
        smem_size = estimate_smem_size_layer(
            layer_config,
            base_config["block_shape"],
            problem.gemm_type,
            5,
        )
        if (
            smem_size * num_ctas_per_sm < _H20_MAX_SMEM_SIZE
            and not base_config["use_stream_k"]
        ):
            deep_candidate = ScheduleCandidate.from_config(
                "dense_deep_pipeline",
                {**base_config, "num_stages": 5},
            )
            candidates.append(deep_candidate)

    plain_candidate = ScheduleCandidate.from_config("dense_plain", base_config)
    candidates.append(plain_candidate)

    register_demand = (
        _GROUPED_INPUT_SCALE_REGISTER_DEMAND
        if (layer_config.input_scale_group_size or 0) > 0
        else None
    )
    analyses = tuple(
        analyze_candidate(problem, candidate, register_demand=register_demand)
        for candidate in candidates
    )
    analysis_by_id = {
        analysis.candidate.candidate_id: analysis for analysis in analyses
    }

    def _legal(candidate: ScheduleCandidate | None) -> bool:
        return (
            candidate is not None
            and analysis_by_id[candidate.candidate_id].legal
        )

    arms = (
        (
            tma_candidate,
            "TMA warp-spec preference: a large tile with few warps and a "
            "deep K loop keeps the TMA pipeline fed at three stages",
        ),
        (
            deep_candidate,
            "deep-pipeline preference: a small non-stream-K tile with smem "
            "headroom takes a fifth stage",
        ),
        (
            plain_candidate,
            "grid-fill schedule: neither the TMA nor the deep-pipeline "
            "alternative applies",
        ),
    )
    selected = None
    reason = ""
    for candidate, arm_reason in arms:
        if _legal(candidate):
            selected = candidate
            reason = arm_reason
            break
    if selected is None:
        rejected = {
            analysis.candidate.candidate_id: analysis.rejection_reasons
            for analysis in analyses
        }
        raise AssertionError(f"no legal fused-E8M0 dense schedule: {rejected}")

    return TuningDecision(
        problem=problem,
        family="fused_e8m0_dense",
        selected=selected,
        considered=analyses,
        reason=reason,
    )


def _h20_base_config(
    layer_config: LayerConfig,
    use_f16_accum: bool,
    gemm_type: GemmType,
) -> dict:
    a_dtype = layer_config.a_dtype
    group_size = (
        layer_config.input_scale_group_size or layer_config.weight_scale_group_size
    )
    is_moe = gemm_type != GemmType.DENSE
    if a_dtype.num_bits == 16:
        return {
            "block_shape": (64, 256, 512 // a_dtype.num_bits),
            "warp_shape": (64, 64, 512 // a_dtype.num_bits),
            "num_ctas_per_sm": 2,
        }
    elif layer_config.use_fused_e8m0_scale and not is_moe:
        return {
            "block_shape": (128, 128, 1024 // a_dtype.num_bits),
            "warp_shape": (128, 32, 1024 // a_dtype.num_bits),
            "num_ctas_per_sm": 2,
        }
    elif layer_config.use_fused_e8m0_scale and is_moe:
        return {
            "block_shape": (64, 128, 1024 // a_dtype.num_bits),
            "warp_shape": (64, 32, 1024 // a_dtype.num_bits),
            "num_ctas_per_sm": 3,
        }
    elif group_size == 0 and not is_moe:
        return {
            "block_shape": (64, 256, 512 // a_dtype.num_bits),
            "warp_shape": (64, 64, 512 // a_dtype.num_bits),
            "num_ctas_per_sm": 2,
        }
    elif group_size == 0 and is_moe:
        return {
            "block_shape": (64, 128, 512 // a_dtype.num_bits),
            "warp_shape": (64, 32, 512 // a_dtype.num_bits),
            "num_ctas_per_sm": 3,
        }
    elif group_size >= 128 and layer_config.shape_k > 512:
        return {
            "block_shape": (64, 128, 1024 // a_dtype.num_bits),
            "warp_shape": (64, 16, 1024 // a_dtype.num_bits),
            "num_ctas_per_sm": 2,
        }
    else:
        return {
            "block_shape": (64, 128, 512 // a_dtype.num_bits),
            "warp_shape": (64, 32, 512 // a_dtype.num_bits),
            "num_ctas_per_sm": 3 if is_moe else 2,
        }


def _small_m_dense_override(
    layer_config: LayerConfig,
    shape_m: int,
    block_shape_m: int,
    num_sms: int,
) -> dict | None:
    a_bits = layer_config.a_dtype.num_bits
    b_bits = layer_config.b_dtype.num_bits
    shape_n, shape_k = layer_config.shape_n, layer_config.shape_k
    if shape_m > block_shape_m or a_bits not in (8, 16) or shape_n % 128:
        return None
    if a_bits == 8 and (not layer_config.b_dtype.is_integer_type or b_bits > 4):
        return None

    warp_k = 512 // a_bits
    reference_k = 2 * warp_k
    reference_tiles = (
        math.ceil(shape_m / block_shape_m) * (shape_n // 128) * (shape_k // reference_k)
    )
    tiles_per_sm = reference_tiles / num_sms

    def make_config(block_n, block_k, warp_n, warp_k, num_stages, use_tma, overlap=False):
        config = {
            "block_shape": (block_shape_m, block_n, block_k),
            "warp_shape": (block_shape_m, warp_n, warp_k),
            "use_stream_k": True,
            "num_sms": num_sms,
            "num_stages": num_stages,
            "num_ctas_per_sm": 1,
            "use_tma": use_tma,
            "use_warp_spec": use_tma,
            "use_mbarrier": use_tma,
        }
        if overlap:
            config["reduce_overlap_last_stage_only"] = True
        return config

    max_output_values = 6 * 1024
    wide_k = 2 * reference_k if a_bits == 16 or b_bits >= 4 else reference_k
    wide_num_iters = (
        math.ceil(shape_m / block_shape_m) * (shape_n // 256) * (shape_k // wide_k)
    )
    stage4_slice = max(4, math.ceil(wide_num_iters / num_sms / 4) * 4)
    stage4_active_ctas = min(num_sms, math.ceil(wide_num_iters / stage4_slice))
    if (
        block_shape_m * 256 <= max_output_values
        and shape_n % 256 == 0
        and shape_k % wide_k == 0
        and shape_k // wide_k >= 4 * 4
        and stage4_active_ctas * 2 >= num_sms
    ):
        block_shape = (block_shape_m, 256, wide_k)
        stage_scores = []
        for num_stages in range(4, 7):
            smem_size = estimate_smem_size_layer(
                layer_config, block_shape, GemmType.DENSE, num_stages
            )
            if smem_size > _H20_MAX_SMEM_SIZE * 0.8:
                continue
            slice_iters = math.ceil(wide_num_iters / num_sms / num_stages) * num_stages
            active_ctas = min(num_sms, math.ceil(wide_num_iters / slice_iters))
            pipeline_gain = 0.05 * min(b_bits, 4) / 4
            stage_scores.append(
                (active_ctas * (1 + pipeline_gain * num_stages), num_stages)
            )
        if stage_scores:
            return make_config(
                256, wide_k, 64, warp_k, max(stage_scores)[1], True, overlap=True
            )

    if a_bits != 16 or block_shape_m * 128 > max_output_values:
        return None
    k_pipeline_turns = shape_k // warp_k
    if k_pipeline_turns <= 32:
        return None
    if k_pipeline_turns <= 64 and tiles_per_sm <= 6 and shape_k % warp_k == 0:
        return make_config(128, warp_k, 32, warp_k, 4, False)
    if shape_k % reference_k == 0 and shape_n // 128 <= math.ceil(num_sms / 8):
        return make_config(128, reference_k, 32, reference_k, 4, True)
    return None


def _tune_long_k_moe_residency(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    config: dict,
    num_sms_physical: int,
) -> None:
    block_m, block_n, block_k = config["block_shape"]
    if layer_config.shape_k <= 1024 or block_m > 32:
        return

    num_stages = 4
    warp_shape = config["warp_shape"]
    smem_size = estimate_smem_size_layer(
        layer_config,
        config["block_shape"],
        gemm_type,
        num_stages,
        warp_shape=warp_shape,
        mma_accum_bits=16 if config["use_f16_accum"] else 32,
    )
    num_threads = math.prod(config["block_shape"]) // math.prod(warp_shape) * 32
    num_experts = layer_config.num_experts
    if shape_m < num_experts:
        estimated_m_blocks = shape_m
    else:
        blocks_per_expert = math.ceil(shape_m / num_experts / block_m)
        estimated_m_blocks = num_experts * blocks_per_expert

    if layer_config.shape_n >= 1024 and layer_config.shape_n % 512 == 0:
        wide_block_n = 512
    elif layer_config.shape_n >= 512 and layer_config.shape_n % 256 == 0:
        wide_block_n = 256
    else:
        wide_block_n = 0
    wide_block_k = 64
    wide_warp_n = 64
    wide_num_stages = 3
    wide_output_tiles = 0
    if wide_block_n:
        wide_output_tiles = estimated_m_blocks * (layer_config.shape_n // wide_block_n)
    expert_tile_fill = shape_m / (estimated_m_blocks * block_m)
    has_wide_grid = wide_output_tiles >= 2 * num_sms_physical
    underfilled_expert_tiles = expert_tile_fill < 0.5 or (
        expert_tile_fill <= 0.5
        and (has_wide_grid or layer_config.b_dtype.num_bits < 4)
    )
    wide_k_tiles = layer_config.shape_k // wide_block_k
    stream_k_grid_gain = min(16, max(4, wide_k_tiles // (2 * wide_num_stages)))
    stream_k_can_fill_grid = (
        layer_config.b_dtype.num_bits <= 4
        and wide_output_tiles * stream_k_grid_gain >= 3 * num_sms_physical
        and wide_k_tiles >= 64
    )
    has_wide_tile = wide_block_n > 0 and block_m * wide_block_n <= 8 * 1024
    wide_n_aligned = wide_block_n > 0 and layer_config.shape_n % wide_block_n == 0
    wide_k_aligned = layer_config.shape_k % wide_block_k == 0
    has_wide_parallelism = has_wide_grid or stream_k_can_fill_grid
    is_wide_tile_legal = has_wide_tile and wide_n_aligned and wide_k_aligned
    use_wide_moe_tile = (
        is_wide_tile_legal and underfilled_expert_tiles and has_wide_parallelism
    )
    if use_wide_moe_tile:
        wide_block_shape = (block_m, wide_block_n, wide_block_k)
        wide_warp_shape = (block_m, wide_warp_n, wide_block_k)
        wide_smem_size = estimate_smem_size_layer(
            layer_config,
            wide_block_shape,
            gemm_type,
            wide_num_stages,
            warp_shape=wide_warp_shape,
            mma_accum_bits=16 if config["use_f16_accum"] else 32,
        )
        wide_num_threads = (
            math.prod(wide_block_shape) // math.prod(wide_warp_shape) * 32
        )
        wide_num_ctas = min(
            3, _H20_MAX_SMEM_SIZE // wide_smem_size, 1024 // wide_num_threads
        )
        if wide_num_ctas >= 1:
            config.update(
                block_shape=wide_block_shape,
                warp_shape=wide_warp_shape,
                num_stages=wide_num_stages,
                num_ctas_per_sm=wide_num_ctas,
                num_sms=num_sms_physical,
            )
            return

    num_output_tiles = estimated_m_blocks * (layer_config.shape_n // block_n)
    num_ctas_per_sm = min(3, _H20_MAX_SMEM_SIZE // smem_size, 1024 // num_threads)
    if num_output_tiles < num_sms_physical:
        num_ctas_per_sm = min(num_ctas_per_sm, 2)
    if num_ctas_per_sm < 1:
        return

    k_tiles = layer_config.shape_k // block_k
    useful_ctas = num_output_tiles * math.ceil(k_tiles / num_stages)
    num_sms = min(num_sms_physical, math.ceil(useful_ctas / num_ctas_per_sm))
    config.update(
        num_stages=num_stages,
        num_ctas_per_sm=num_ctas_per_sm,
        num_sms=max(1, num_sms),
    )


def build_h20_seed_config(problem: TuningProblem) -> dict:
    """The single H20 seed heuristic (the former legacy get_config body).

    In-scope fused-E8M0 constructions are served by the candidate selectors
    below; every other construction gets its config directly from this seed."""
    layer_config = problem.layer_config
    shape_m = problem.shape_m
    gemm_type = problem.gemm_type
    use_f16_accum = problem.use_f16_accum
    use_batch_invariant = problem.use_batch_invariant
    is_moe = gemm_type != GemmType.DENSE
    a_dtype = layer_config.a_dtype

    config = _h20_base_config(layer_config, use_f16_accum, gemm_type)
    block_shape_m, block_shape_n, block_shape_k = config["block_shape"]
    num_ctas_per_sm = config.get("num_ctas_per_sm", 1)
    warp_shape_m, warp_shape_n, warp_shape_k = config["warp_shape"]
    if layer_config.use_packed_k_layout:
        warp_shape_n = max(warp_shape_n, 32)
    num_stages = 3
    min_warp_shape_n = (
        32 if a_dtype.num_bits == 16 or layer_config.use_packed_k_layout else 16
    )
    while layer_config.shape_n % block_shape_n:
        block_shape_n //= 2
        warp_shape_n = min(warp_shape_n, block_shape_n // 4)
    assert warp_shape_n >= min_warp_shape_n

    if not layer_config.num_experts:
        block_shape_m = _dense_block_shape_m(shape_m, block_shape_m)
        if (
            a_dtype == dtypes.int8
            and block_shape_m > 32
            and block_shape_m % 16 != 0
        ):
            block_shape_m = math.ceil(block_shape_m / 16) * 16
    else:
        block_shape_m = _moe_block_shape_m(
            shape_m, layer_config.num_experts, block_shape_m
        )

    warp_shape_m = block_shape_m
    num_blocks_n = layer_config.shape_n // block_shape_n
    if not layer_config.num_experts:
        num_blocks_m = math.ceil(shape_m / block_shape_m)
    elif shape_m < layer_config.num_experts:
        num_blocks_m = shape_m
    else:
        num_blocks_m = layer_config.num_experts

    num_sms = problem.device.num_sms
    num_sms_physical = num_sms
    while num_blocks_n * num_blocks_m * 2 < num_sms * num_ctas_per_sm:
        if warp_shape_n == 64:
            warp_shape_n = warp_shape_n // 2
            block_shape_n = block_shape_n // 2
            num_blocks_n = num_blocks_n * 2
            if num_ctas_per_sm == 2:
                num_ctas_per_sm = 3
            continue
        elif num_ctas_per_sm > 1:
            num_ctas_per_sm = num_ctas_per_sm - 1
            continue
        else:
            break

    num_warps_m = block_shape_m // warp_shape_m
    num_warps_n = block_shape_n // warp_shape_n
    num_warps_k = block_shape_k // warp_shape_k
    num_warps = num_warps_m * num_warps_n * num_warps_k * num_ctas_per_sm

    if num_warps == 4:
        warp_shape_k = 512 // a_dtype.num_bits
        block_shape_k = warp_shape_k * 2

    if num_warps <= 8 and block_shape_m <= 32:
        if is_moe and warp_shape_n == 64:
            warp_shape_n = warp_shape_n // 2
        else:
            num_warps_k = block_shape_k // warp_shape_k
            warp_shape_k = 512 // a_dtype.num_bits
            block_shape_k = warp_shape_k * num_warps_k * 2

    if (
        is_moe
        and layer_config.shape_k <= 512
        and layer_config.shape_n >= 2048
        and block_shape_m <= 32
    ):
        if block_shape_n == 256:
            warp_shape_n = 32
            block_shape_n = 128
            num_blocks_n = num_blocks_n * 2

        if num_blocks_n * num_blocks_m >= num_sms * 4:
            num_ctas_per_sm = 4

    if warp_shape_k == block_shape_k and warp_shape_k == 512 // a_dtype.num_bits:
        block_shape = (block_shape_m, block_shape_n, block_shape_k * 2)
        smem_size = estimate_smem_size_layer(
            layer_config, block_shape, gemm_type, num_stages
        )
        if smem_size * num_ctas_per_sm < _H20_MAX_SMEM_SIZE:
            block_shape_k = block_shape_k * 2
            warp_shape_k = warp_shape_k * 2

    max_num_stages = 4
    for num_stages_new in range(num_stages + 1, max_num_stages + 1):
        block_shape = (block_shape_m, block_shape_n, block_shape_k)
        smem_size = estimate_smem_size_layer(
            layer_config, block_shape, gemm_type, num_stages_new
        )
        if smem_size * num_ctas_per_sm < _H20_MAX_SMEM_SIZE:
            num_stages = num_stages_new

    if not is_moe:
        block_shape_m = _fit_dense_block_m_to_output_grid(
            layer_config,
            shape_m,
            block_shape_m,
            block_shape_n,
            block_shape_k,
            num_sms_physical,
        )
        warp_shape_m = block_shape_m
        num_blocks_m = math.ceil(shape_m / block_shape_m)

    if num_ctas_per_sm == 1:
        factor = min(4.5, layer_config.shape_k / (3 * block_shape_k))
        if layer_config.shape_k > 1024:
            factor = min(9, max(factor, layer_config.shape_k / (8 * block_shape_k)))
        num_sms = min(num_sms, math.ceil(num_blocks_n * num_blocks_m * factor))

    while layer_config.shape_k % block_shape_k != 0:
        warp_shape_k = 512 // a_dtype.num_bits
        block_shape_k = block_shape_k // 2
        assert block_shape_k >= warp_shape_k

    if (
        a_dtype.num_bits == 8
        and layer_config.input_scale_group_size > 0
        and gemm_type != GemmType.GROUPED_MASKED
        and shape_m >= 6144
    ):
        num_ctas_per_sm = min(num_ctas_per_sm, 2)

    config = {
        "block_shape": (block_shape_m, block_shape_n, block_shape_k),
        "warp_shape": (warp_shape_m, warp_shape_n, warp_shape_k),
        "use_stream_k": layer_config.shape_k > 1024,
        "use_f16_accum": use_f16_accum,
        "num_sms": num_sms,
        "num_stages": num_stages,
        "num_ctas_per_sm": num_ctas_per_sm,
    }

    if layer_config.shape_k <= 512 and is_moe and shape_m >= 2048:
        config["use_tma"] = True
        config["use_mbarrier"] = True
        if gemm_type == GemmType.INDEXED:
            config["use_tma_a"] = False
            config["use_tma_c"] = False

        if config["num_ctas_per_sm"] > 1 and shape_m >= 24576:
            tiles_per_cta = 5
            block_m, block_n, _ = config["block_shape"]
            num_tiles = (layer_config.shape_n // block_n) * (shape_m // block_m)
            sms_target = num_tiles / (config["num_ctas_per_sm"] * tiles_per_cta)
            config["num_sms"] = max(config["num_sms"], 1 << round(math.log2(sms_target)))

    has_tma_tile = block_shape_m >= 48
    has_tma_resources = num_ctas_per_sm <= 2 and num_warps <= 8
    has_tma_pipeline = layer_config.shape_k // block_shape_k >= 24
    use_dense_tma = (
        not is_moe and has_tma_tile and has_tma_resources and has_tma_pipeline
    )
    if use_dense_tma:
        config["use_tma"] = True
        config["use_warp_spec"] = True
        config["use_mbarrier"] = True
        config["num_stages"] = 3
    elif config["num_stages"] == 4 and block_shape_m <= 32:
        block_shape = (block_shape_m, block_shape_n, block_shape_k)
        smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, 5)
        if smem_size * num_ctas_per_sm < _H20_MAX_SMEM_SIZE and not config["use_stream_k"]:
            config["num_stages"] = 5

    if not is_moe and not use_batch_invariant:
        config.update(
            _small_m_dense_override(
                layer_config, shape_m, block_shape_m, num_sms_physical
            )
            or {}
        )
    elif is_moe and not use_batch_invariant:
        _tune_long_k_moe_residency(
            layer_config, shape_m, gemm_type, config, num_sms_physical
        )

    if use_batch_invariant:
        warp_shape_k = 512 // a_dtype.num_bits
        block_shape_k = 512 // a_dtype.num_bits
        config["block_shape"] = (block_shape_m, block_shape_n, block_shape_k)
        config["warp_shape"] = (warp_shape_m, warp_shape_n, warp_shape_k)
        config["use_tma"] = False
        config["use_warp_spec"] = False
        config["use_mbarrier"] = False
        config["use_stream_k"] = False

    return config


def _register_preference_applies(problem: TuningProblem) -> bool:
    return (
        (problem.layer_config.input_scale_group_size or 0) > 0
        and problem.gemm_type != GemmType.GROUPED_MASKED
        and problem.shape_m >= 6144
    )


def _select_small_tile(
    problem: TuningProblem,
    base_config: dict,
    grid_ctas: int,
) -> TuningDecision:
    """Replicates the legacy long-K MoE residency tuner as a candidate choice:
    wide-tile rewrite vs pipeline-residency retune vs the grid-fill schedule."""
    layer_config = problem.layer_config
    shape_m = problem.shape_m
    num_sms_physical = problem.device.num_sms
    block_m, block_n, block_k = base_config["block_shape"]
    warp_shape = base_config["warp_shape"]
    accum_bits = 16 if base_config["use_f16_accum"] else 32

    base_ctas = grid_ctas
    if _register_preference_applies(problem):
        base_ctas = min(base_ctas, 2)

    num_experts = layer_config.num_experts
    if shape_m < num_experts:
        estimated_m_blocks = shape_m
    else:
        estimated_m_blocks = num_experts * math.ceil(
            shape_m / num_experts / block_m
        )

    shape_n = layer_config.shape_n
    if shape_n >= 1024 and shape_n % 512 == 0:
        wide_block_n = 512
    elif shape_n >= 512 and shape_n % 256 == 0:
        wide_block_n = 256
    else:
        wide_block_n = 0
    wide_block_k = 64
    wide_num_stages = 3
    wide_output_tiles = (
        estimated_m_blocks * (shape_n // wide_block_n) if wide_block_n else 0
    )
    expert_tile_fill = shape_m / (estimated_m_blocks * block_m)
    has_wide_grid = wide_output_tiles >= 2 * num_sms_physical
    underfilled_expert_tiles = expert_tile_fill < 0.5 or (
        expert_tile_fill <= 0.5
        and (has_wide_grid or layer_config.b_dtype.num_bits < 4)
    )
    wide_k_tiles = layer_config.shape_k // wide_block_k
    stream_k_grid_gain = min(16, max(4, wide_k_tiles // (2 * wide_num_stages)))
    stream_k_can_fill_grid = (
        layer_config.b_dtype.num_bits <= 4
        and wide_output_tiles * stream_k_grid_gain >= 3 * num_sms_physical
        and wide_k_tiles >= 64
    )
    has_wide_tile = wide_block_n > 0 and block_m * wide_block_n <= 8 * 1024
    wide_aligned = (
        wide_block_n > 0
        and shape_n % wide_block_n == 0
        and layer_config.shape_k % wide_block_k == 0
    )
    use_wide_moe_tile = (
        has_wide_tile
        and wide_aligned
        and underfilled_expert_tiles
        and (has_wide_grid or stream_k_can_fill_grid)
    )

    candidates: list[ScheduleCandidate] = []
    wide_candidate = None
    if wide_block_n:
        wide_block_shape = (block_m, wide_block_n, wide_block_k)
        wide_warp_shape = (block_m, 64, wide_block_k)
        wide_smem = estimate_smem_size_layer(
            layer_config,
            wide_block_shape,
            problem.gemm_type,
            wide_num_stages,
            warp_shape=wide_warp_shape,
            mma_accum_bits=accum_bits,
        )
        wide_threads = (
            math.prod(wide_block_shape) // math.prod(wide_warp_shape) * 32
        )
        wide_ctas = min(
            3, _H20_MAX_SMEM_SIZE // wide_smem, 1024 // wide_threads
        )
        if wide_ctas >= 1:
            wide_candidate = ScheduleCandidate.from_config(
                "small_tile_wide",
                {
                    **base_config,
                    "block_shape": wide_block_shape,
                    "warp_shape": wide_warp_shape,
                    "num_stages": wide_num_stages,
                    "num_ctas_per_sm": wide_ctas,
                    "num_sms": num_sms_physical,
                },
            )
            candidates.append(wide_candidate)

    smem_size = estimate_smem_size_layer(
        layer_config,
        base_config["block_shape"],
        problem.gemm_type,
        4,
        warp_shape=warp_shape,
        mma_accum_bits=accum_bits,
    )
    num_threads = (
        math.prod(base_config["block_shape"]) // math.prod(warp_shape) * 32
    )
    res_ctas = min(3, _H20_MAX_SMEM_SIZE // smem_size, 1024 // num_threads)
    num_output_tiles = estimated_m_blocks * (shape_n // block_n)
    if num_output_tiles < num_sms_physical:
        res_ctas = min(res_ctas, 2)
    residency_candidate = None
    if res_ctas >= 1:
        k_tiles = layer_config.shape_k // block_k
        useful_ctas = num_output_tiles * math.ceil(k_tiles / 4)
        res_sms = max(
            1, min(num_sms_physical, math.ceil(useful_ctas / res_ctas))
        )
        residency_candidate = ScheduleCandidate.from_config(
            "small_tile_residency",
            {
                **base_config,
                "num_stages": 4,
                "num_ctas_per_sm": res_ctas,
                "num_sms": res_sms,
            },
        )
        candidates.append(residency_candidate)

    grid_fill_candidate = ScheduleCandidate.from_config(
        "small_tile_grid_fill",
        {**base_config, "num_ctas_per_sm": base_ctas},
    )
    candidates.append(grid_fill_candidate)

    register_demand = (
        _GROUPED_INPUT_SCALE_REGISTER_DEMAND
        if (layer_config.input_scale_group_size or 0) > 0
        else None
    )
    analyses = tuple(
        analyze_candidate(problem, candidate, register_demand=register_demand)
        for candidate in candidates
    )
    analysis_by_id = {
        analysis.candidate.candidate_id: analysis for analysis in analyses
    }

    def _legal(candidate: ScheduleCandidate | None) -> bool:
        return (
            candidate is not None
            and analysis_by_id[candidate.candidate_id].legal
        )

    arms = (
        (
            wide_candidate if use_wide_moe_tile else None,
            "wide-tile preference: underfilled expert tiles with enough wide "
            "output parallelism; a wider N tile amortizes the long-K pipeline",
        ),
        (
            residency_candidate,
            "residency preference: trade excess pipeline storage for more "
            "resident CTAs on the long-K small-tile schedule",
        ),
        (
            grid_fill_candidate,
            "grid-fill fallback: neither the wide-tile nor the residency "
            "alternative fits the resource limits",
        ),
    )
    selected = None
    reason = ""
    for candidate, arm_reason in arms:
        if _legal(candidate):
            selected = candidate
            reason = arm_reason
            break
    if selected is None:
        rejected = {
            analysis.candidate.candidate_id: analysis.rejection_reasons
            for analysis in analyses
        }
        raise AssertionError(
            f"no legal small-tile fused-E8M0 MoE schedule: {rejected}"
        )

    return TuningDecision(
        problem=problem,
        family="fused_e8m0_moe_small_tile",
        selected=selected,
        considered=analyses,
        reason=reason,
    )


def select_fused_e8m0_moe(problem: TuningProblem) -> TuningDecision:
    base_config, grid_ctas = _base_schedule(problem)

    if base_config["block_shape"][0] <= 32:
        return _select_small_tile(problem, base_config, grid_ctas)

    register_demand = (
        _GROUPED_INPUT_SCALE_REGISTER_DEMAND
        if (problem.layer_config.input_scale_group_size or 0) > 0
        else None
    )

    candidates: list[ScheduleCandidate] = []
    for num_ctas_per_sm in range(grid_ctas, 0, -1):
        candidates.append(
            ScheduleCandidate.from_config(
                f"fused_e8m0_moe_cta{num_ctas_per_sm}",
                {**base_config, "num_ctas_per_sm": num_ctas_per_sm},
            )
        )

    analyses = tuple(
        analyze_candidate(problem, candidate, register_demand=register_demand)
        for candidate in candidates
    )

    legal: list[CandidateAnalysis] = [a for a in analyses if a.legal]
    if not legal:
        rejected = {
            analysis.candidate.candidate_id: analysis.rejection_reasons
            for analysis in analyses
        }
        raise AssertionError(f"no legal fused-E8M0 MoE schedule: {rejected}")

    # Default: the grid-fill CTA count (occupancy first). The measured register
    # preference demotes exactly one step, and only from 3 CTAs/SM — the one
    # construction where the launch-bounds budget was measured to spill.
    selected = legal[0]
    demoted = False
    if (
        _register_preference_applies(problem)
        and selected.candidate.num_ctas_per_sm == 3
        and selected.register_pressured
        and len(legal) > 1
    ):
        selected = legal[1]
        demoted = True

    if demoted:
        reason = (
            "register-budget preference: large-M grouped-input-scale keeps "
            "per-group scale buffers in registers only below 3 CTAs/SM "
            "(measured on H20: 1.4x latency from local-memory spills at CTA3)"
        )
    else:
        reason = (
            "occupancy preference: grid-fill CTA count; register pressure "
            "only outweighs the extra resident CTA for large-M "
            "grouped-input-scale shapes (not grouped-masked decode)"
        )

    return TuningDecision(
        problem=problem,
        family="fused_e8m0_moe",
        selected=selected.candidate,
        considered=analyses,
        reason=reason,
    )
