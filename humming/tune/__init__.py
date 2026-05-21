import functools
from typing import TYPE_CHECKING

import torch

from humming.config import GemmType
from humming.tune.base import DeviceHeuristics
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

if TYPE_CHECKING:
    from humming.layer import HummingLayerMeta

heuristics_map: dict[int, type[DeviceHeuristics]] = {
    75: Sm75Heuristics,
    80: Sm80Heuristics,
    86: Sm86Heuristics,
    87: Sm87Heuristics,
    89: Sm89Heuristics,
    90: Sm90Heuristics,
    100: Sm100Heuristics,
    103: Sm100Heuristics,
    120: Sm89Heuristics,
    121: Sm89Heuristics,
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
    name = torch.cuda.get_device_name(device)
    if "H20" in name and "H200" not in name:
        return Sm90H20Heuristics

    return heuristics_map[sm_version]


@functools.lru_cache(maxsize=1024)
def get_heuristics_config(
    meta: "HummingLayerMeta | dict",
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    gemm_type: str | GemmType = "dense",
):
    from humming.layer import HummingLayerMeta

    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)

    if isinstance(meta, dict):
        meta = HummingLayerMeta(**meta)
    heuristics_cls = get_heuristics_class()
    if isinstance(shape_m, int):
        return heuristics_cls.get_config(
            meta=meta,
            shape_m=shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )
    else:
        return heuristics_cls.get_configs(
            meta=meta,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )
