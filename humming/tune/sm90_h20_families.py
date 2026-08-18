"""Bounded schedule candidates for SM90 H20 heuristics.

First H20 family migrated onto the candidate flow from tune/candidate.py:
fused-E8M0 8-bit MoE GEMMs with long K. Selection preserves the legacy
`Sm90H20Heuristics.get_config` output for every shape in scope (guarded by
tests/test_sm90_h20_candidate_parity.py); the candidate list additionally
records the residency alternatives and why the measured policy prefers one.
"""

import math

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

    Long-K fused-E8M0 MoE with block_m >= 48: the block_m gate keeps the
    small-tile paths (long-K residency tuning, num_warps<=8 reshaping) on the
    legacy heuristic until they are migrated with their own parity coverage.
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
    if block_m < 48:
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


def _register_preference_applies(problem: TuningProblem) -> bool:
    return (
        (problem.layer_config.input_scale_group_size or 0) > 0
        and problem.gemm_type != GemmType.GROUPED_MASKED
        and problem.shape_m >= 6144
    )


def select_fused_e8m0_moe(problem: TuningProblem) -> TuningDecision:
    base_config, grid_ctas = _base_schedule(problem)

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
