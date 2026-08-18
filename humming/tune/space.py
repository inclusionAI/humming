import logging
from collections import Counter
from typing import TYPE_CHECKING

from humming.config import GemmType
from humming.utils.smem import estimate_smem_size_layer

if TYPE_CHECKING:
    from humming.config import LayerConfig


_LOGGER = logging.getLogger(__name__)


_BLOCK_M_VALUES = (8, 16, 24, 32, 48, 64, 96, 128)
_BLOCK_N_VALUES = (64, 128, 256)
_WARP_N_VALUES = (16, 32, 64)
_NUM_CTAS_PER_SM_VALUES = (1, 2, 3)
# Coarse grid for fast tuning: a ~3x smaller candidate set that keeps every block_m
# rung that won a shape in the reference sweeps -- indexed winners cluster at small
# block_m (8/16/24/48) while masked/dense reach 128, so both regimes are covered.
# The reduction comes from the dims that never won: block_n=64 (broken for small
# block_m, no indexed win), warp_n=64 (already broken on 8-bit activations), and
# ctas=1. The heuristic config is injected separately by the search driver, so this
# stays a pure coverage grid and never loses the hand-tuned baseline.
_FAST_BLOCK_M_VALUES = (8, 16, 24, 32, 48, 64, 128)
_FAST_BLOCK_N_VALUES = (128, 256)
_FAST_WARP_N_VALUES = (16, 32)
_FAST_NUM_CTAS_PER_SM_VALUES = (2, 3)
_NUM_STAGE_CANDIDATES = (2, 3, 4, 5)
_SUPPORTED_A_BITS = (4, 8, 16)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _dtype_warp_shape_allowed(a_bits: int, warp_n: int, warp_k: int) -> bool:
    if a_bits == 16:
        return warp_n >= 32 and warp_k >= 32
    if a_bits == 8:
        # warp_n=16 no longer compiles for 8-bit activations: the wgmma
        # mainloop static_asserts warp iterations (warp_n/16) >= 2.
        return warp_n >= 32 and warp_k >= 64
    if a_bits == 4:
        return warp_n >= 16 and warp_k >= 128
    return False


def _moe_warp_spec_broken(meta: "LayerConfig", gemm_type: GemmType, config: dict) -> bool:
    return bool(config.get("use_warp_spec")) and gemm_type != GemmType.DENSE


def _bn64_bk64_small_bm_broken(
    meta: "LayerConfig",
    gemm_type: GemmType,
    config: dict,
) -> bool:
    block_m, block_n, block_k = config["block_shape"]
    return block_n == 64 and block_k == 64 and block_m <= 48


def _bn64_small_bm_broken(meta: "LayerConfig", gemm_type: GemmType, config: dict) -> bool:
    block_m, block_n, _ = config["block_shape"]
    return block_n == 64 and block_m <= 32


def _wn64_8bit_broken(meta: "LayerConfig", gemm_type: GemmType, config: dict) -> bool:
    _, warp_n, _ = config["warp_shape"]
    return warp_n == 64 and meta.a_dtype.num_bits <= 8



_BROKEN_FAMILIES = (
    ("moe-warp-spec", "moe-warp-spec: livelock", _moe_warp_spec_broken),
    (
        "bn64-bk64-small-bm",
        "bn64-bk64-small-bm: sticky illegal instruction",
        _bn64_bk64_small_bm_broken,
    ),
    ("bn64-small-bm", "bn64-small-bm: sticky illegal instruction", _bn64_small_bm_broken),
    ("wn64-8bit", "wn64-8bit: sticky illegal instruction", _wn64_8bit_broken),
)


def _block_k_values(a_bits: int) -> list[int]:
    if a_bits not in _SUPPORTED_A_BITS:
        raise NotImplementedError(f"unsupported activation dtype for search space: {a_bits=}")
    return sorted({512 // a_bits, 1024 // a_bits, 2048 // a_bits})


def _warp_k_values(a_bits: int) -> list[int]:
    if a_bits not in _SUPPORTED_A_BITS:
        raise NotImplementedError(f"unsupported activation dtype for search space: {a_bits=}")
    return sorted({512 // a_bits, 1024 // a_bits})


def _pipeline_configs(gemm_type: GemmType, block_m: int) -> list[dict]:
    configs = [{}]
    if gemm_type == GemmType.DENSE and block_m >= 48:
        configs.append({"use_tma": True, "use_mbarrier": True})
        configs.append({"use_tma": True, "use_warp_spec": True, "use_mbarrier": True})
    return configs


def _stream_k_values(meta: "LayerConfig", gemm_type: GemmType) -> list[bool]:
    values = [False]
    if gemm_type in (GemmType.DENSE, GemmType.INDEXED) and meta.shape_k > 1024:
        values.append(True)
    return values


def _stage_values(
    meta: "LayerConfig",
    block_shape: tuple[int, int, int],
    gemm_type: GemmType,
    num_ctas_per_sm: int,
    max_smem_size: int,
) -> list[int]:
    allowed = []
    for num_stages in _NUM_STAGE_CANDIDATES:
        smem_size = estimate_smem_size_layer(meta, block_shape, gemm_type, num_stages)
        if smem_size * num_ctas_per_sm <= max_smem_size:
            allowed.append(num_stages)

    if not allowed:
        return []

    stages = []
    if 3 in allowed:
        stages.append(3)
    max_stage = max(allowed)
    if max_stage not in stages:
        stages.append(max_stage)
    return stages


class DeviceSearchSpace:
    """Per-arch search space; mirror the DeviceHeuristics classmethod style."""

    sm_version: int = 0
    max_smem_size: int = 0

    @classmethod
    def filter_with_analysis(
        cls,
        meta: "LayerConfig",
        gemm_type: GemmType,
        num_sms: int,
        shape_m: int,
        configs: list[dict],
        use_batch_invariant: bool = False,
    ) -> list[dict]:
        """Drop candidates the shared candidate analysis proves illegal.

        m-major input scale is a post-processing flag with no analysis rule,
        so it is not part of the problem here. The base space has no analysis
        wired up and returns the list as-is."""
        return configs

    @classmethod
    def enumerate(
        cls,
        meta: "LayerConfig",
        gemm_type: GemmType,
        num_sms: int,
        fast: bool = False,
    ) -> list[dict]:
        candidates = cls._generate_candidates(meta, gemm_type, num_sms, fast)
        return [
            config
            for config in candidates
            if cls.is_valid(meta, gemm_type, config)
            and cls.broken_reason(meta, gemm_type, config) is None
        ]

    @classmethod
    def _generate_candidates(
        cls,
        meta: "LayerConfig",
        gemm_type: GemmType,
        num_sms: int,
        fast: bool = False,
    ) -> list[dict]:
        a_bits = meta.a_dtype.num_bits
        block_k_values = _block_k_values(a_bits)
        warp_k_values = _warp_k_values(a_bits)
        stream_k_values = _stream_k_values(meta, gemm_type)
        block_m_values = _FAST_BLOCK_M_VALUES if fast else _BLOCK_M_VALUES
        block_n_values = _FAST_BLOCK_N_VALUES if fast else _BLOCK_N_VALUES
        warp_n_values = _FAST_WARP_N_VALUES if fast else _WARP_N_VALUES
        num_ctas_values = _FAST_NUM_CTAS_PER_SM_VALUES if fast else _NUM_CTAS_PER_SM_VALUES
        candidates = []

        for block_m in block_m_values:
            for block_n in block_n_values:
                for block_k in block_k_values:
                    block_shape = (block_m, block_n, block_k)
                    for warp_n in warp_n_values:
                        for warp_k in warp_k_values:
                            warp_shape = (block_m, warp_n, warp_k)
                            for pipeline_config in _pipeline_configs(gemm_type, block_m):
                                for num_ctas_per_sm in num_ctas_values:
                                    stage_values = _stage_values(
                                        meta,
                                        block_shape,
                                        gemm_type,
                                        num_ctas_per_sm,
                                        cls.max_smem_size,
                                    )
                                    for num_stages in stage_values:
                                        for use_stream_k in stream_k_values:
                                            config = {
                                                "block_shape": block_shape,
                                                "warp_shape": warp_shape,
                                                "use_stream_k": use_stream_k,
                                                "use_f16_accum": False,
                                                "num_sms": num_sms,
                                                "num_stages": num_stages,
                                                "num_ctas_per_sm": num_ctas_per_sm,
                                            }
                                            config.update(pipeline_config)
                                            candidates.append(config)

        return candidates

    @classmethod
    def is_valid(cls, meta: "LayerConfig", gemm_type: GemmType, config: dict) -> bool:
        block_m, block_n, block_k = config["block_shape"]
        warp_m, warp_n, warp_k = config["warp_shape"]
        num_ctas_per_sm = config["num_ctas_per_sm"]
        num_stages = config["num_stages"]

        if min(block_m, block_n, block_k, warp_m, warp_n, warp_k, num_ctas_per_sm) <= 0:
            return False
        if num_stages <= 0:
            return False
        if warp_m != block_m:
            return False
        if meta.shape_n % block_n != 0 or meta.shape_k % block_k != 0:
            return False
        if block_n % warp_n != 0 or not _is_power_of_two(block_n // warp_n):
            return False
        if block_k % warp_k != 0 or not _is_power_of_two(block_k // warp_k):
            return False
        if warp_n % 16 != 0 or warp_n > 64:
            return False
        if not _dtype_warp_shape_allowed(meta.a_dtype.num_bits, warp_n, warp_k):
            return False

        num_warps = (block_m // warp_m) * (block_n // warp_n) * (block_k // warp_k)
        num_warps *= num_ctas_per_sm
        if num_warps < 2 or num_warps > 32:
            return False

        smem_size = estimate_smem_size_layer(meta, config["block_shape"], gemm_type, num_stages)
        return smem_size * num_ctas_per_sm <= cls.max_smem_size

    @classmethod
    def broken_reason(
        cls,
        meta: "LayerConfig",
        gemm_type: GemmType,
        config: dict,
    ) -> str | None:
        for _name, reason, predicate in _BROKEN_FAMILIES:
            if predicate(meta, gemm_type, config):
                return reason
        return None


class Sm90H20SearchSpace(DeviceSearchSpace):
    sm_version = 90
    max_smem_size = 227 * 1024

    @classmethod
    def filter_with_analysis(
        cls,
        meta: "LayerConfig",
        gemm_type: GemmType,
        num_sms: int,
        shape_m: int,
        configs: list[dict],
        use_batch_invariant: bool = False,
    ) -> list[dict]:
        from humming.tune.candidate import (
            ScheduleCandidate,
            TuningProblem,
            analyze_candidate,
        )
        from humming.tune.sm90_h20_families import make_h20_device_profile

        device = make_h20_device_profile(num_sms)
        kept = []
        rejected: Counter = Counter()
        for config in configs:
            problem = TuningProblem(
                layer_config=meta,
                shape_m=shape_m,
                gemm_type=gemm_type,
                device=device,
                use_f16_accum=bool(config.get("use_f16_accum")),
                use_batch_invariant=use_batch_invariant,
            )
            analysis = analyze_candidate(
                problem, ScheduleCandidate.from_config("tune", config)
            )
            if analysis.legal:
                kept.append(config)
            else:
                for reason in analysis.rejection_reasons:
                    rejected[reason] += 1
        if rejected:
            _LOGGER.info(
                "candidate analysis rejected %d/%d configs at shape_m=%s: %s",
                len(configs) - len(kept),
                len(configs),
                shape_m,
                dict(rejected.most_common(5)),
            )
        return kept


search_space_map: dict[int, type[DeviceSearchSpace]] = {90: Sm90H20SearchSpace}


def get_search_space(
    sm_version: int | None = None,
    device_name: str | None = None,
) -> type[DeviceSearchSpace]:
    if sm_version is None or device_name is None:
        import torch

    if sm_version is None:
        sm_version = torch.cuda.get_device_capability()
    if isinstance(sm_version, tuple):
        sm_version = sm_version[0] * 10 + sm_version[1]
    assert isinstance(sm_version, int), f"{sm_version=}"

    if device_name is None:
        device_name = torch.cuda.get_device_name()
    if "H20" in device_name and "H200" not in device_name:
        return Sm90H20SearchSpace

    if sm_version not in search_space_map:
        raise NotImplementedError(
            f"no DeviceSearchSpace registered for {sm_version=}; "
            "register a new DeviceSearchSpace in search_space_map"
        )
    return search_space_map[sm_version]
