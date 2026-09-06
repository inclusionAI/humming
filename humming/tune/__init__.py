import functools
import os

import torch

from humming.config import GemmType, LayerConfig
from humming.device import DeviceInfo, get_device_index
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
from humming.tune.sm90_h200 import Sm90H200Heuristics
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
    110: Sm100Heuristics,
    120: Sm120Heuristics,
    121: Sm121Heuristics,
}


def _forced_heuristics_class() -> type[DeviceHeuristics] | None:
    """Resolve HUMMING_FORCE_HEURISTICS to a heuristics class.

    Accepts either a heuristics class name (e.g. ``sm90_h200``) or an SM
    version key from ``heuristics_map`` (e.g. ``90``). Useful on OEM-renamed
    GPUs where name-based dispatch picks the wrong variant.
    """
    forced = os.environ.get("HUMMING_FORCE_HEURISTICS")
    if not forced:
        return None
    by_name = {cls.__name__: cls for cls in heuristics_map.values()}
    by_name.update(
        {
            Sm90H20Heuristics.__name__: Sm90H20Heuristics,
            Sm90H200Heuristics.__name__: Sm90H200Heuristics,
        }
    )
    if forced in by_name:
        return by_name[forced]
    # Case-insensitive / module-style aliases (e.g. "sm90_h200" for
    # Sm90H200Heuristics).
    canonical = forced.lower().replace("_", "").removesuffix("heuristics")
    for name, cls in by_name.items():
        if name.lower().removesuffix("heuristics") == canonical:
            return cls
    if forced.isdigit():
        sm_version = int(forced)
        if sm_version in heuristics_map:
            return heuristics_map[sm_version]
        return heuristics_map.get(sm_version // 10 * 10)
    raise ValueError(
        f"HUMMING_FORCE_HEURISTICS={forced!r} does not match any heuristics "
        f"class ({', '.join(sorted(by_name))}) or SM version key "
        f"({', '.join(map(str, heuristics_map))})"
    )


def get_heuristics_class(device: int | torch.device | None = None) -> type[DeviceHeuristics]:
    forced = _forced_heuristics_class()
    if forced is not None:
        return forced
    info = DeviceInfo(device)
    sm_version = info.sm_version
    if sm_version == 90:
        if "H200" in info.name:
            return Sm90H200Heuristics
        if "H20" in info.name:
            return Sm90H20Heuristics

    if sm_version in heuristics_map:
        return heuristics_map[sm_version]

    sm_version_base = sm_version // 10 * 10

    return heuristics_map[sm_version_base]


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
def _get_heuristics_config(
    layer_config: LayerConfig,
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    use_m_major_input_scale: bool = False,
    gemm_type: str | GemmType = "dense",
    device_index: int = 0,
):
    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)

    heuristics_cls = get_heuristics_class(device=device_index)
    if isinstance(shape_m, int):
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


def get_heuristics_config(
    layer_config: LayerConfig | dict,
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    use_m_major_input_scale: bool = False,
    gemm_type: str | GemmType = "dense",
    device: int | torch.device | None = None,
):
    device_index = get_device_index(device)
    with torch.cuda.device(device_index):
        if isinstance(layer_config, dict):
            layer_config = LayerConfig(**layer_config)
        layer_config.check_device(device_index)
        return _get_heuristics_config(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            use_m_major_input_scale,
            gemm_type,
            device_index,
        )
