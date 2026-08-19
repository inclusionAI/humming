import dataclasses
import functools

import torch

from humming.config import GemmType, LayerConfig
from humming.tune.base import DeviceHeuristics
from humming.tune.raster import raster_group_m_for_config
from humming.tune.sm8x import (
    Sm80Heuristics,
    Sm86Heuristics,
    Sm87Heuristics,
    Sm89Heuristics,
)
from humming.tune.sm75 import Sm75Heuristics
from humming.tune.sm90 import Sm90Heuristics
from humming.tune.sm90_h20 import Sm90H20Heuristics
from humming.tune.sm100 import Sm100Heuristics
from humming.tune.sm120 import Sm120Heuristics
from humming.tune.sm121 import Sm121Heuristics

heuristics_map: dict[int, type[DeviceHeuristics]] = {
    75: Sm75Heuristics,
    80: Sm80Heuristics,
    86: Sm86Heuristics,
    87: Sm87Heuristics,
    89: Sm89Heuristics,
    90: Sm90Heuristics,
    100: Sm100Heuristics,
    103: Sm100Heuristics,
    120: Sm120Heuristics,
    121: Sm121Heuristics,
}


def get_heuristics_class(
    sm_version: int | tuple[int, int] | None = None,
    device: int | torch.device | None = None,
) -> type[DeviceHeuristics]:
    if sm_version is None:
        sm_version = torch.cuda.get_device_capability(device)
    if isinstance(sm_version, tuple):
        sm_version = sm_version[0] * 10 + sm_version[1]
    assert isinstance(sm_version, int)
    if sm_version == 90:
        name = torch.cuda.get_device_name(device)
        if "H20" in name and "H200" not in name:
            return Sm90H20Heuristics

    return heuristics_map[sm_version]


def _apply_m_major_input_scale(
    config: dict,
    use_m_major_input_scale: bool,
    layer_config: LayerConfig,
    gemm_type: GemmType,
) -> None:
    if not use_m_major_input_scale:
        return
    use_tma = config.get("use_tma", False)
    if use_tma and layer_config.input_scale_group_size > 0 and gemm_type == GemmType.DENSE:
        config["use_tma_as"] = True


def _apply_raster_group_m(config: dict, layer_config, gemm_type) -> None:
    if gemm_type != GemmType.DENSE:
        return
    if config.get("raster_group_m") is not None or "block_shape" not in config:
        return
    try:
        config["raster_group_m"] = raster_group_m_for_config(
            layer_config,
            config["block_shape"],
            config.get("multi_cast_size_a", 1),
        )
    except Exception:
        pass


@functools.lru_cache(maxsize=1024)
def get_heuristics_config(
    layer_config: LayerConfig | dict,
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    use_m_major_input_scale: bool = False,
    fuse_e8m0_scale: bool | None = None,
    gemm_type: str | GemmType = "dense",
):
    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)

    if isinstance(layer_config, dict):
        layer_config = LayerConfig(**layer_config)
    if fuse_e8m0_scale is not None:
        if not layer_config.use_shared_e8m0_scale_storage and (
            fuse_e8m0_scale != layer_config.use_fused_e8m0_scale
        ):
            raise ValueError("runtime E8M0 scale switching requires use_shared_e8m0_scale_storage=True")
        layer_config = dataclasses.replace(
            layer_config,
            use_fused_e8m0_scale=fuse_e8m0_scale,
        )
    heuristics_cls = get_heuristics_class()

    # Shared-storage layers with no explicit fuse_e8m0_scale request let the
    # device heuristics choose the execution path per routed-M interval, so
    # callers need no per-phase logic.  AUTO engages only for INDEXED MoE on
    # shapes with measured winner bands; everything else (other gemm types,
    # unmeasured shapes, devices without bands) keeps the previous behaviour
    # (the LayerConfig default path).
    auto_bands = None
    if (
        fuse_e8m0_scale is None
        and layer_config.use_shared_e8m0_scale_storage
        and gemm_type == GemmType.INDEXED
    ):
        band_map = getattr(heuristics_cls, "shared_e8m0_auto_fused_bands", None)
        if band_map:
            auto_bands = band_map.get((layer_config.shape_n, layer_config.shape_k, layer_config.num_experts))

    if isinstance(shape_m, int):
        if auto_bands is not None:
            layer_config, config = _shared_auto_single_config(
                heuristics_cls,
                layer_config,
                shape_m=shape_m,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
                fused_bands=auto_bands,
            )
        else:
            config = heuristics_cls.get_config(
                layer_config=layer_config,
                shape_m=shape_m,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
            )
        _apply_m_major_input_scale(config, use_m_major_input_scale, layer_config, gemm_type)
        _apply_raster_group_m(config, layer_config, gemm_type)
        return config
    else:
        if auto_bands is not None:
            configs = _shared_auto_config_table(
                heuristics_cls,
                layer_config,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
                fused_bands=auto_bands,
            )
        else:
            configs = heuristics_cls.get_configs(
                layer_config=layer_config,
                use_f16_accum=use_f16_accum,
                use_batch_invariant=use_batch_invariant,
                gemm_type=gemm_type,
            )
        for entry in configs:
            _apply_m_major_input_scale(entry[2], use_m_major_input_scale, layer_config, gemm_type)
            _apply_raster_group_m(entry[2], layer_config, gemm_type)
        return configs


def _in_fused_bands(fused_bands, routed_m: int) -> bool:
    return any(lo < routed_m and (hi is None or routed_m <= hi) for lo, hi in fused_bands)


def _shared_auto_single_config(
    heuristics_cls,
    layer_config: LayerConfig,
    *,
    shape_m: int,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    gemm_type: GemmType,
    fused_bands,
):
    """Resolve the shared-storage execution path for one routed-M value."""
    use_fused = _in_fused_bands(fused_bands, shape_m)
    path_layer = dataclasses.replace(layer_config, use_fused_e8m0_scale=use_fused)
    config = heuristics_cls.get_config(
        layer_config=path_layer,
        shape_m=shape_m,
        use_f16_accum=use_f16_accum,
        use_batch_invariant=use_batch_invariant,
        gemm_type=gemm_type,
    )
    # Route through ComputeConfig.fuse_e8m0_scale so per-interval kernels take
    # the same validated override path as an explicit caller request.
    config["fuse_e8m0_scale"] = use_fused
    return path_layer, config


def _shared_auto_config_table(
    heuristics_cls,
    layer_config: LayerConfig,
    *,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    gemm_type: GemmType,
    fused_bands,
):
    """Merge the explicit/fused interval tables into one path-annotated table.

    Table entries are ``(lo_exclusive, hi_inclusive, config)``.  Every merged
    sub-interval picks the path from the measured winner bands; the choice is
    recorded as ``fuse_e8m0_scale`` in the config so each interval compiles
    its own kernel variant through the standard ComputeConfig override.
    """
    tables = {}
    for fused in (False, True):
        tables[fused] = heuristics_cls.get_configs(
            layer_config=dataclasses.replace(layer_config, use_fused_e8m0_scale=fused),
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )

    def lookup(table, routed_m):
        for lo, hi, config in table:
            if lo < routed_m <= hi:
                return config
        raise RuntimeError(f"no heuristic interval contains routed_m={routed_m}")

    breakpoints = {bound for table in tables.values() for lo, hi, _ in table for bound in (lo, hi)}
    max_bound = max(breakpoints)
    # Winner-band edges must become interval boundaries too, or the path
    # decision would be quantized to the underlying tables' sampling grid.
    for lo, hi in fused_bands:
        for bound in (lo, hi):
            if bound is not None and 0 < bound < max_bound:
                breakpoints.add(bound)
    breakpoints = sorted(breakpoints)
    merged = []
    for lo, hi in zip(breakpoints, breakpoints[1:], strict=False):
        probe_m = lo + 1
        use_fused = _in_fused_bands(fused_bands, probe_m)
        config = dict(lookup(tables[use_fused], probe_m))
        config["fuse_e8m0_scale"] = use_fused
        if merged and merged[-1][1] == lo and merged[-1][2] == config:
            merged[-1] = (merged[-1][0], hi, merged[-1][2])
        else:
            merged.append((lo, hi, config))
    return merged
