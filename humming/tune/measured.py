import json
import logging
import math
import os
import tempfile
from typing import TYPE_CHECKING

from humming.config import GemmType
from humming.tune import cache, get_heuristics_config, measure
from humming.tune.space import get_search_space

if TYPE_CHECKING:
    from humming.config import LayerConfig


_LAST_BUCKET_MAX = 1 << 30
_MEASURE_TOPK = 8
_LOGGER = logging.getLogger(__name__)


def _normalize_meta(meta: "LayerConfig | dict") -> "LayerConfig":
    if isinstance(meta, dict):
        from humming.config import LayerConfig

        return LayerConfig(**meta)
    return meta


def _normalize_gemm_type(gemm_type: str | GemmType) -> GemmType:
    if isinstance(gemm_type, str):
        return GemmType(gemm_type)
    return gemm_type


def _make_flags(
    use_f16_accum: bool,
    use_batch_invariant: bool,
    use_m_major_input_scale: bool,
) -> dict:
    return {
        "use_f16_accum": use_f16_accum,
        "use_batch_invariant": use_batch_invariant,
        "use_m_major_input_scale": use_m_major_input_scale,
    }


def _shape_m_values(shape_m_list) -> list[int]:
    values = sorted({int(shape_m) for shape_m in shape_m_list})
    if not values:
        raise ValueError("shape_m_list must not be empty")
    if values[0] <= 0:
        raise ValueError("shape_m_list values must be positive")
    return values


def _config_key(config: dict) -> str:
    return json.dumps(config, sort_keys=True)


def _default_progress_log_path(meta: "LayerConfig", gemm_type: GemmType, flags: dict) -> str:
    filename = cache.make_cache_key(meta, gemm_type, flags) + ".progress"
    return os.path.join(tempfile.gettempdir(), "humming_tune_progress", filename)


def _make_layer_args(
    meta: "LayerConfig",
    gemm_type: GemmType,
    top_k: int,
    is_moe_down: bool,
    balanced: bool,
    expert_max_tokens: int | None,
    input_scale_group_size: int,
) -> dict:
    if gemm_type == GemmType.GROUPED_CONTIGUOUS:
        raise ValueError("grouped_contiguous is not supported by humming.tune.measure")
    if gemm_type != GemmType.DENSE:
        if meta.num_experts <= 0:
            raise ValueError("meta.num_experts must be positive for MoE gemm types")
        if top_k <= 0:
            raise ValueError("top_k must be positive for MoE gemm types")

    meta_input_scale_group_size = int(meta.input_scale_group_size)
    if input_scale_group_size not in (0, meta_input_scale_group_size):
        raise ValueError(
            "input_scale_group_size must match meta.input_scale_group_size "
            f"when provided: {input_scale_group_size=} {meta_input_scale_group_size=}"
        )

    args = {
        "shape_n": meta.shape_n,
        "shape_k": meta.shape_k,
        "num_experts": meta.num_experts,
        "a_dtype": str(meta.a_dtype),
        "b_dtype": str(meta.b_dtype),
        "bs_dtype": str(meta.bs_dtype),
        "c_dtype": str(meta.c_dtype),
        "weight_scale_group_size": meta.weight_scale_group_size,
        "input_scale_group_size": meta_input_scale_group_size,
        "top_k": top_k,
        "is_moe_down": is_moe_down,
        "balanced": balanced,
        "expert_max_tokens": expert_max_tokens,
    }
    missing = [name for name in measure._LAYER_ARG_NAMES if name not in args]
    assert not missing, f"{missing=}"
    return {name: args[name] for name in measure._LAYER_ARG_NAMES}


def _prepare_config(
    config: dict,
    meta: "LayerConfig",
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_m_major_input_scale: bool,
) -> dict:
    prepared = dict(config)
    prepared["use_f16_accum"] = use_f16_accum
    if (
        use_m_major_input_scale
        and prepared.get("use_tma")
        and meta.input_scale_group_size > 0
        and gemm_type == GemmType.DENSE
    ):
        prepared["use_tma_as"] = True
    return prepared


def _find_timing(timings: list[measure.ConfigTiming], config: dict) -> measure.ConfigTiming | None:
    key = _config_key(config)
    for timing in timings:
        if _config_key(timing.config) == key:
            return timing
    return None


def _select_winner(
    shape_m: int,
    timings: list[measure.ConfigTiming],
    baseline: dict,
) -> tuple[dict, float | None, bool]:
    legal = [
        timing
        for timing in timings
        if timing.correctness_ok is True
        and timing.fine_ms is not None
        and math.isfinite(timing.fine_ms)
    ]
    if legal:
        winner = min(legal, key=lambda timing: timing.fine_ms)
        return dict(winner.config), winner.fine_ms, False

    _LOGGER.warning("falling back to heuristic config for shape_m=%s", shape_m)
    return dict(baseline), None, True


def _merge_adjacent_buckets(buckets: list) -> list:
    merged = []
    for lo, hi, config in buckets:
        assert lo < hi
        if merged:
            assert merged[-1][1] == lo
            if _config_key(merged[-1][2]) == _config_key(config):
                merged[-1][1] = hi
                continue
        merged.append([lo, hi, config])
    return merged


def _make_measured_buckets(per_m: list[dict]) -> list:
    last_shape_m = 0
    buckets = []
    for item in per_m:
        shape_m = item["shape_m"]
        buckets.append([last_shape_m, shape_m, item["config"]])
        last_shape_m = shape_m
    return _merge_adjacent_buckets(buckets)


def _splice_heuristic_range(
    table: list,
    lo: int,
    hi: int,
    heuristics: list,
    meta: "LayerConfig",
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_m_major_input_scale: bool,
) -> list:
    replacement = [
        [
            max(bucket_lo, lo),
            min(bucket_hi, hi),
            _prepare_config(
                config,
                meta,
                gemm_type,
                use_f16_accum,
                use_m_major_input_scale,
            ),
        ]
        for bucket_lo, bucket_hi, config in heuristics
        if bucket_hi > lo and bucket_lo < hi
    ]
    assert replacement[0][0] == lo
    assert replacement[-1][1] == hi

    before = []
    after = []
    for bucket_lo, bucket_hi, config in table:
        if bucket_lo < lo:
            before.append([bucket_lo, min(bucket_hi, lo), config])
        if bucket_hi > hi:
            after.append([max(bucket_lo, hi), bucket_hi, config])
    return _merge_adjacent_buckets(before + replacement + after)


def _bucket_interior_checkpoints(lo: int, hi: int, grid_points: set[int]) -> list[int]:
    checkpoints = []
    for checkpoint in ((lo + hi) // 2, lo + 1):
        if (
            lo < checkpoint < hi
            and checkpoint not in grid_points
            and checkpoint not in checkpoints
        ):
            checkpoints.append(checkpoint)
    return checkpoints


def _verify_bucket_interiors(
    table: list,
    shape_m_values: list[int],
    measurer: measure.Measurer,
    meta: "LayerConfig",
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    use_m_major_input_scale: bool,
) -> list:
    verified_table = table
    grid_points = set(shape_m_values)
    heuristics = None
    for lo, hi, config in table:
        for checkpoint in _bucket_interior_checkpoints(lo, hi, grid_points):
            heuristic_config = get_heuristics_config(
                meta,
                shape_m=checkpoint,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                use_m_major_input_scale=use_m_major_input_scale,
                gemm_type=gemm_type,
            )
            heuristic_config = _prepare_config(
                heuristic_config,
                meta,
                gemm_type,
                use_f16_accum,
                use_m_major_input_scale,
            )
            if _config_key(config) == _config_key(heuristic_config):
                continue

            timings = measurer.measure(
                measure.MeasureRequest(
                    shape_m=checkpoint,
                    configs=[config],
                    baseline_config=heuristic_config,
                )
            )
            timing = _find_timing(timings, config)
            if timing is not None and timing.correctness_ok is True:
                continue

            _LOGGER.warning(
                "bucket=(%s,%s] config=%s checkpoint=%s reason=interior-verification-failed",
                lo,
                hi,
                config,
                checkpoint,
            )
            if heuristics is None:
                heuristics = get_heuristics_config(
                    meta,
                    shape_m=None,
                    use_f16_accum=use_f16_accum,
                    use_batch_invariant=use_batch_invariant,
                    use_m_major_input_scale=use_m_major_input_scale,
                    gemm_type=gemm_type,
                )
            verified_table = _splice_heuristic_range(
                verified_table,
                lo,
                hi,
                heuristics,
                meta,
                gemm_type,
                use_f16_accum,
                use_m_major_input_scale,
            )
            break
    return verified_table


def _make_table(
    per_m: list[dict],
    meta: "LayerConfig",
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    use_m_major_input_scale: bool,
) -> list:
    table = _make_measured_buckets(per_m)
    last_shape_m = table[-1][1]
    heur = get_heuristics_config(
        meta,
        shape_m=None,
        use_f16_accum=use_f16_accum,
        use_batch_invariant=use_batch_invariant,
        use_m_major_input_scale=use_m_major_input_scale,
        gemm_type=gemm_type,
    )
    table = _splice_heuristic_range(
        table,
        last_shape_m,
        _LAST_BUCKET_MAX,
        heur,
        meta,
        gemm_type,
        use_f16_accum,
        use_m_major_input_scale,
    )

    assert table[-1][1] == _LAST_BUCKET_MAX
    return table


def _run_search(
    meta,
    gemm_type,
    shape_m_list,
    *,
    top_k=0,
    is_moe_down=False,
    balanced=True,
    expert_max_tokens=None,
    input_scale_group_size=0,
    use_f16_accum=False,
    use_batch_invariant=False,
    use_m_major_input_scale=False,
    progress_log_path=None,
    num_spares=2,
    device=None,
    fast=False,
    reverify_winners=True,
    verify_bucket_interiors=True,
) -> tuple[list, list[dict]]:
    meta = _normalize_meta(meta)
    gemm_type = _normalize_gemm_type(gemm_type)
    shape_m_values = _shape_m_values(shape_m_list)
    flags = _make_flags(use_f16_accum, use_batch_invariant, use_m_major_input_scale)

    fingerprint = cache.current_fingerprint(device)
    search_space = get_search_space(fingerprint["sm_version"], fingerprint["gpu_name"])
    candidates = search_space.enumerate(meta, gemm_type, fingerprint["num_sms"], fast=fast)
    candidates = [
        _prepare_config(
            config,
            meta,
            gemm_type,
            use_f16_accum,
            use_m_major_input_scale,
        )
        for config in candidates
    ]
    if progress_log_path is None:
        progress_log_path = _default_progress_log_path(meta, gemm_type, flags)

    layer_args = _make_layer_args(
        meta,
        gemm_type,
        top_k,
        is_moe_down,
        balanced,
        expert_max_tokens,
        input_scale_group_size,
    )

    per_m = []
    with measure.Measurer(
        layer_args,
        gemm_type,
        progress_log_path=progress_log_path,
        topk=_MEASURE_TOPK,
        num_spares=num_spares,
        device_index=device,
    ) as measurer:
        for shape_m in shape_m_values:
            baseline = get_heuristics_config(
                meta,
                shape_m=shape_m,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                use_m_major_input_scale=use_m_major_input_scale,
                gemm_type=gemm_type,
            )
            baseline = _prepare_config(
                baseline,
                meta,
                gemm_type,
                use_f16_accum,
                use_m_major_input_scale,
            )
            timings = measurer.measure(
                measure.MeasureRequest(
                    shape_m=shape_m,
                    configs=candidates,
                    baseline_config=baseline,
                )
            )
            winner_config, fine_ms, used_fallback = _select_winner(shape_m, timings, baseline)
            if (
                reverify_winners
                and not used_fallback
                and _config_key(winner_config) != _config_key(baseline)
            ):
                for reverify_round in range(2):
                    reverify_timings = measurer.measure(
                        measure.MeasureRequest(
                            shape_m=shape_m,
                            configs=[winner_config],
                            baseline_config=baseline,
                        )
                    )
                    reverify_timing = _find_timing(reverify_timings, winner_config)
                    if reverify_timing is not None and reverify_timing.correctness_ok is True:
                        break

                    _LOGGER.warning(
                        "config=%s shape_m=%s reason=reverify-failed",
                        winner_config,
                        shape_m,
                    )
                    winner_key = _config_key(winner_config)
                    timings = [
                        timing for timing in timings if _config_key(timing.config) != winner_key
                    ]
                    if reverify_round == 1:
                        winner_config, fine_ms, used_fallback = _select_winner(
                            shape_m, [], baseline
                        )
                        break
                    winner_config, fine_ms, used_fallback = _select_winner(
                        shape_m, timings, baseline
                    )
                    if used_fallback or _config_key(winner_config) == _config_key(baseline):
                        break
            heuristic_timing = _find_timing(timings, baseline)
            heuristic_ms = None
            if heuristic_timing is not None:
                heuristic_ms = heuristic_timing.fine_ms
            speedup = None
            if (
                heuristic_ms is not None
                and fine_ms is not None
                and math.isfinite(heuristic_ms)
                and fine_ms > 0
            ):
                speedup = heuristic_ms / fine_ms

            per_m.append(
                {
                    "shape_m": shape_m,
                    "config": winner_config,
                    "fine_ms": fine_ms,
                    "heuristic_ms": heuristic_ms,
                    "speedup": speedup,
                    "used_fallback": used_fallback,
                    "same_as_heuristic": _config_key(winner_config) == _config_key(baseline),
                }
            )

        measured_buckets = _make_measured_buckets(per_m)
        if verify_bucket_interiors:
            measured_buckets = _verify_bucket_interiors(
                measured_buckets,
                shape_m_values,
                measurer,
                meta,
                gemm_type,
                use_f16_accum,
                use_batch_invariant,
                use_m_major_input_scale,
            )

    table_input = [
        {"shape_m": hi, "config": config} for _lo, hi, config in measured_buckets
    ]

    return (
        _make_table(
            table_input,
            meta,
            gemm_type,
            use_f16_accum,
            use_batch_invariant,
            use_m_major_input_scale,
        ),
        per_m,
    )


def search(
    meta,
    gemm_type,
    shape_m_list,
    *,
    top_k=0,
    is_moe_down=False,
    balanced=True,
    expert_max_tokens=None,
    input_scale_group_size=0,
    use_f16_accum=False,
    use_batch_invariant=False,
    use_m_major_input_scale=False,
    progress_log_path=None,
    num_spares=2,
    device=None,
    fast=False,
    reverify_winners=True,
    verify_bucket_interiors=True,
) -> list:
    """Return measured tuning buckets in the same shape as base.py get_configs."""
    table, _ = _run_search(
        meta,
        gemm_type,
        shape_m_list,
        top_k=top_k,
        is_moe_down=is_moe_down,
        balanced=balanced,
        expert_max_tokens=expert_max_tokens,
        input_scale_group_size=input_scale_group_size,
        use_f16_accum=use_f16_accum,
        use_batch_invariant=use_batch_invariant,
        use_m_major_input_scale=use_m_major_input_scale,
        progress_log_path=progress_log_path,
        num_spares=num_spares,
        device=device,
        fast=fast,
        reverify_winners=reverify_winners,
        verify_bucket_interiors=verify_bucket_interiors,
    )
    return table


def tune_and_save(
    meta,
    gemm_type,
    shape_m_list,
    *,
    reverify_winners=True,
    verify_bucket_interiors=True,
    **kw,
) -> tuple[list, str]:
    """Run search, save the measured table in the tune cache, and return its path."""
    cache_dir = kw.pop("cache_dir", None)
    meta = _normalize_meta(meta)
    gemm_type = _normalize_gemm_type(gemm_type)
    shape_m_values = _shape_m_values(shape_m_list)
    flags = _make_flags(
        kw.get("use_f16_accum", False),
        kw.get("use_batch_invariant", False),
        kw.get("use_m_major_input_scale", False),
    )
    table, per_m = _run_search(
        meta,
        gemm_type,
        shape_m_values,
        reverify_winners=reverify_winners,
        verify_bucket_interiors=verify_bucket_interiors,
        **kw,
    )
    fingerprint = cache.current_fingerprint(kw.get("device"))
    path = cache.save_table(
        meta,
        gemm_type,
        flags,
        fingerprint,
        table,
        cache_dir=cache_dir,
        shape_m_list=shape_m_values,
        per_m=[
            {
                "shape_m": item["shape_m"],
                "fine_ms": item["fine_ms"],
                "heuristic_ms": item["heuristic_ms"],
                "speedup": item["speedup"],
                "used_fallback": item["used_fallback"],
                "same_as_heuristic": item["same_as_heuristic"],
            }
            for item in per_m
        ],
    )
    same_as_heuristic = sum(item["same_as_heuristic"] for item in per_m)
    speedup_points = [item for item in per_m if item["speedup"] is not None]
    if speedup_points:
        max_speedup_point = max(speedup_points, key=lambda item: item["speedup"])
        max_speedup = max_speedup_point["speedup"]
        max_speedup_shape_m = max_speedup_point["shape_m"]
    else:
        max_speedup = float("nan")
        max_speedup_shape_m = "n/a"
    _LOGGER.info(
        "tune summary: %s points, %s same-as-heuristic, max speedup %.2f@m=%s",
        len(per_m),
        same_as_heuristic,
        max_speedup,
        max_speedup_shape_m,
    )
    return table, path
