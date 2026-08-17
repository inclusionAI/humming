import dataclasses
import math
from collections.abc import Mapping
from typing import Any

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.utils.smem import estimate_smem_size_layer


@dataclasses.dataclass(frozen=True, slots=True)
class DeviceProfile:
    name: str
    sm_version: int
    num_sms: int | None
    max_smem_size: int
    max_smem_per_sm: int | None = None
    max_threads_per_sm: int = 2048

    @property
    def resident_smem_size(self) -> int:
        if self.max_smem_per_sm is None:
            return self.max_smem_size
        return self.max_smem_per_sm


@dataclasses.dataclass(frozen=True, slots=True)
class TuningProblem:
    layer_config: LayerConfig
    shape_m: int
    gemm_type: GemmType
    device: DeviceProfile
    use_f16_accum: bool = False
    use_batch_invariant: bool = False
    use_m_major_input_scale: bool = False

    def estimate_num_blocks_m(self, block_shape_m: int) -> int:
        if self.gemm_type == GemmType.DENSE or not self.layer_config.num_experts:
            return math.ceil(self.shape_m / block_shape_m)
        return min(self.shape_m, self.layer_config.num_experts)


@dataclasses.dataclass(frozen=True, slots=True)
class ScheduleCandidate:
    candidate_id: str
    config_items: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        names = tuple(name for name, _ in self.config_items)
        if len(names) != len(set(names)):
            raise ValueError("config_items contains duplicate keys")

    @classmethod
    def from_config(
        cls,
        candidate_id: str,
        config: Mapping[str, Any],
    ) -> "ScheduleCandidate":
        return cls(candidate_id=candidate_id, config_items=tuple(config.items()))

    def to_config(self) -> dict[str, Any]:
        return dict(self.config_items)

    def get(self, name: str, default: Any = None) -> Any:
        for item_name, value in self.config_items:
            if item_name == name:
                return value
        return default

    def with_updates(
        self,
        candidate_id: str | None = None,
        **updates: Any,
    ) -> "ScheduleCandidate":
        config = self.to_config()
        config.update(updates)
        return type(self).from_config(candidate_id or self.candidate_id, config)

    @property
    def block_shape(self) -> tuple[int, int, int]:
        return _require_shape(self.get("block_shape"), "block_shape")

    @property
    def warp_shape(self) -> tuple[int, int, int]:
        return _require_shape(self.get("warp_shape"), "warp_shape")

    @property
    def num_stages(self) -> int:
        return _require_int(self.get("num_stages", 2), "num_stages")

    @property
    def num_ctas_per_sm(self) -> int:
        return _require_int(self.get("num_ctas_per_sm", 1), "num_ctas_per_sm")


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateAnalysis:
    candidate: ScheduleCandidate
    rejection_reasons: tuple[str, ...]
    num_math_threads: int
    num_load_threads: int
    num_threads: int
    smem_size: int
    num_output_tiles: int
    thread_smem_cta_limit: int
    waves: int | None

    @property
    def legal(self) -> bool:
        return not self.rejection_reasons


@dataclasses.dataclass(frozen=True, slots=True)
class TuningDecision:
    problem: TuningProblem
    family: str
    selected: ScheduleCandidate
    considered: tuple[CandidateAnalysis, ...]
    reason: str

    def __post_init__(self) -> None:
        selected_analysis = next(
            (
                analysis
                for analysis in self.considered
                if analysis.candidate == self.selected
            ),
            None,
        )
        if selected_analysis is None:
            raise ValueError(
                "selected candidate must be present in considered analyses"
            )
        if not selected_analysis.legal:
            raise ValueError("selected candidate must be legal")

    @property
    def selected_analysis(self) -> CandidateAnalysis:
        return next(
            analysis
            for analysis in self.considered
            if analysis.candidate == self.selected
        )

    def to_config(self) -> dict[str, Any]:
        return self.selected.to_config()


def _require_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_shape(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a three-dimensional tuple")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain integers")
    return value


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def get_problem_rejection_reasons(
    layer_config: LayerConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    input_group_size = layer_config.input_scale_group_size
    unpadded_shape_k = layer_config.shape_k - layer_config.pad_shape_k
    if (
        input_group_size
        and layer_config.shape_k != input_group_size
        and unpadded_shape_k % input_group_size
    ):
        reasons.append(
            f"unpadded shape_k={unpadded_shape_k} is not divisible by "
            f"input scale group={input_group_size}"
        )
    if (
        layer_config.a_dtype.num_bits != 16
        and layer_config.mma_type != MmaType.MXMMA
        and layer_config.as_dtype != dtypes.float32
    ):
        reasons.append(
            "non-MXMMA input scales must use float32 storage, got "
            f"{layer_config.as_dtype}"
        )
    mma_k = 256 // layer_config.a_dtype.num_bits
    scale_groups = (
        ("input scale group", layer_config.input_scale_group_size),
        ("weight scale group", layer_config.weight_scale_group_size),
    )
    if layer_config.mma_type == MmaType.MXMMA:
        for name, group_size in scale_groups:
            if group_size and (
                mma_k % group_size or mma_k // group_size not in (1, 2, 4)
            ):
                reasons.append(
                    f"{name}={group_size} does not divide MXMMA K={mma_k} "
                    "into one, two, or four groups"
                )
        if (
            layer_config.input_scale_group_size
            and layer_config.weight_scale_group_size
            and layer_config.input_scale_group_size
            != layer_config.weight_scale_group_size
        ):
            reasons.append("MXMMA input and weight scale groups must match")
    else:
        for name, group_size in scale_groups:
            if group_size and group_size < mma_k:
                reasons.append(f"{name}={group_size} is smaller than MMA K={mma_k}")
        if 1 < layer_config.weight_scale_group_size_n < 64:
            reasons.append(
                "weight scale N group="
                f"{layer_config.weight_scale_group_size_n} is smaller than 64"
            )
        if (
            layer_config.is_block_weight_scale
            and layer_config.input_scale_group_size
            and layer_config.input_scale_group_size
            != layer_config.weight_scale_group_size
        ):
            reasons.append("block input and weight scale groups must match")
    return tuple(reasons)


def get_geometry_rejection_reasons(
    layer_config: LayerConfig,
    block_shape: tuple[int, int, int],
    warp_shape: tuple[int, int, int],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for name, shape in (("block_shape", block_shape), ("warp_shape", warp_shape)):
        if any(size <= 0 for size in shape):
            reasons.append(f"{name} dimensions must be positive: {shape}")
    if reasons:
        return tuple(reasons)

    if block_shape[0] > 256:
        reasons.append(f"block_m={block_shape[0]} exceeds 256")
    if layer_config.shape_n % block_shape[1]:
        reasons.append(
            f"shape_n={layer_config.shape_n} is not divisible by "
            f"block_n={block_shape[1]}"
        )
    if layer_config.shape_k % block_shape[2]:
        reasons.append(
            f"shape_k={layer_config.shape_k} is not divisible by "
            f"block_k={block_shape[2]}"
        )
    for name, size in (
        ("block_n", block_shape[1]),
        ("block_k", block_shape[2]),
        ("warp_n", warp_shape[1]),
        ("warp_k", warp_shape[2]),
    ):
        if not _is_power_of_two(size):
            reasons.append(f"{name}={size} must be a power of two")
    if any(block % warp for block, warp in zip(block_shape, warp_shape, strict=True)):
        reasons.append(
            f"block_shape={block_shape} does not nest warp_shape={warp_shape}"
        )
        ratios = None
    else:
        ratios = tuple(
            block // warp for block, warp in zip(block_shape, warp_shape, strict=True)
        )
        if not all(_is_power_of_two(ratio) for ratio in ratios):
            reasons.append(f"block-to-warp ratios must be powers of two: {ratios}")

    if warp_shape[1] > 64:
        reasons.append(f"warp_n={warp_shape[1]} exceeds 64")
    if warp_shape[1] % 16:
        reasons.append(f"warp_n={warp_shape[1]} must be divisible by 16")
    if warp_shape[0] % 8:
        reasons.append(f"warp_m={warp_shape[0]} must be divisible by 8")
    if layer_config.mma_type in (MmaType.MMA, MmaType.MXMMA) and warp_shape[0] % 16:
        reasons.append(
            f"warp_m={warp_shape[0]} must be divisible by 16 for "
            f"{layer_config.mma_type.value}"
        )
    min_warp_n = (
        32
        if layer_config.a_dtype.num_bits == 16 or layer_config.use_packed_k_layout
        else 16
    )
    min_warp_k_by_bits = {16: 32, 8: 64, 4: 128}
    min_warp_k = min_warp_k_by_bits.get(layer_config.a_dtype.num_bits)
    if warp_shape[1] < min_warp_n:
        reasons.append(f"warp_n={warp_shape[1]} is smaller than minimum {min_warp_n}")
    if min_warp_k is None:
        reasons.append(
            f"unsupported activation width {layer_config.a_dtype.num_bits} "
            "for warp-K analysis"
        )
    elif warp_shape[2] < min_warp_k:
        reasons.append(f"warp_k={warp_shape[2]} is smaller than minimum {min_warp_k}")

    if layer_config.mma_type == MmaType.WGMMA and ratios is not None:
        if ratios[1] % 4:
            reasons.append(
                "WGMMA requires the block-N tile to contain a multiple of "
                f"four warp-N tiles: {ratios[1]}"
            )
        swizzle_bytes = (
            128 if layer_config.a_dtype.num_bits * block_shape[2] >= 1024 else 64
        )
        max_warp_k = swizzle_bytes * 8 // layer_config.a_dtype.num_bits
        if warp_shape[2] > max_warp_k:
            reasons.append(
                f"warp_k={warp_shape[2]} exceeds WGMMA swizzle limit {max_warp_k}"
            )

    if layer_config.use_packed_k_layout:
        for scale_name, group_size in (
            ("input scale group", layer_config.input_scale_group_size),
            ("weight scale group", layer_config.weight_scale_group_size),
        ):
            if group_size and warp_shape[2] > group_size:
                reasons.append(
                    f"packed-K warp_k={warp_shape[2]} exceeds {scale_name}={group_size}"
                )

    return tuple(reasons)


def analyze_candidate(
    problem: TuningProblem,
    candidate: ScheduleCandidate,
) -> CandidateAnalysis:
    reasons: list[str] = []
    try:
        block_shape = candidate.block_shape
    except ValueError as error:
        reasons.append(str(error))
        block_shape = (0, 0, 0)
    try:
        warp_shape = candidate.warp_shape
    except ValueError as error:
        reasons.append(str(error))
        warp_shape = (0, 0, 0)
    try:
        num_stages = candidate.num_stages
    except ValueError as error:
        reasons.append(str(error))
        num_stages = 0
    try:
        num_ctas_per_sm = candidate.num_ctas_per_sm
    except ValueError as error:
        reasons.append(str(error))
        num_ctas_per_sm = 0

    try:
        num_write_splits = _require_int(
            candidate.get("num_write_splits", 1),
            "num_write_splits",
        )
    except ValueError as error:
        reasons.append(str(error))
        num_write_splits = 0
    try:
        multi_cast_size_a = _require_int(
            candidate.get("multi_cast_size_a", 1),
            "multi_cast_size_a",
        )
    except ValueError as error:
        reasons.append(str(error))
        multi_cast_size_a = 0
    try:
        multi_cast_size_b = _require_int(
            candidate.get("multi_cast_size_b", 1),
            "multi_cast_size_b",
        )
    except ValueError as error:
        reasons.append(str(error))
        multi_cast_size_b = 0

    if problem.shape_m <= 0:
        reasons.append(f"shape_m={problem.shape_m} must be positive")
    if problem.layer_config.shape_n <= 0:
        reasons.append(f"shape_n={problem.layer_config.shape_n} must be positive")
    if problem.layer_config.shape_k <= 0:
        reasons.append(f"shape_k={problem.layer_config.shape_k} must be positive")
    if problem.device.num_sms is not None and problem.device.num_sms <= 0:
        reasons.append(f"device num_sms={problem.device.num_sms} must be positive")
    if problem.device.max_smem_size <= 0:
        reasons.append(
            f"device max_smem_size={problem.device.max_smem_size} must be positive"
        )
    if problem.device.resident_smem_size <= 0:
        reasons.append(
            "device resident shared memory="
            f"{problem.device.resident_smem_size} must be positive"
        )
    if problem.device.max_threads_per_sm <= 0:
        reasons.append(
            "device max_threads_per_sm="
            f"{problem.device.max_threads_per_sm} must be positive"
        )
    if num_stages <= 0:
        reasons.append(f"num_stages={num_stages} must be positive")
    if num_ctas_per_sm <= 0:
        reasons.append(f"num_ctas_per_sm={num_ctas_per_sm} must be positive")
    if num_write_splits <= 0:
        reasons.append(f"num_write_splits={num_write_splits} must be positive")
    if multi_cast_size_a <= 0:
        reasons.append(f"multi_cast_size_a={multi_cast_size_a} must be positive")
    if multi_cast_size_b <= 0:
        reasons.append(f"multi_cast_size_b={multi_cast_size_b} must be positive")

    shapes_positive = all(size > 0 for size in block_shape + warp_shape)
    ratios: tuple[int, int, int] | None = None
    reasons.extend(get_problem_rejection_reasons(problem.layer_config))
    reasons.extend(
        get_geometry_rejection_reasons(
            problem.layer_config,
            block_shape,
            warp_shape,
        )
    )
    if shapes_positive and not any(
        block % warp for block, warp in zip(block_shape, warp_shape, strict=True)
    ):
        ratios = tuple(
            block // warp for block, warp in zip(block_shape, warp_shape, strict=True)
        )

    if shapes_positive:
        for scale_name, group_size in (
            ("input scale group", problem.layer_config.input_scale_group_size),
            ("weight scale group", problem.layer_config.weight_scale_group_size),
        ):
            if (
                group_size
                and group_size % block_shape[2]
                and block_shape[2] % group_size
            ):
                reasons.append(
                    f"block_k={block_shape[2]} and "
                    f"{scale_name}={group_size} do not nest"
                )
        if multi_cast_size_a > 0 and problem.layer_config.shape_n % (
            block_shape[1] * multi_cast_size_a
        ):
            reasons.append(
                f"shape_n={problem.layer_config.shape_n} is not divisible by "
                f"block_n * multi_cast_size_a="
                f"{block_shape[1] * multi_cast_size_a}"
            )
        if problem.use_batch_invariant and (
            candidate.get("use_stream_k", True) or block_shape[2] != warp_shape[2]
        ):
            reasons.append(
                "batch-invariant schedules require direct output and one warp-K tile"
            )

    num_math_threads = math.prod(ratios) * 32 if ratios is not None else 0
    use_warp_spec = bool(candidate.get("use_warp_spec", False))
    use_tma = bool(candidate.get("use_tma", False))
    tma_fields = {
        name: bool(
            candidate.get(
                name,
                False if name == "use_tma_as" else use_tma,
            )
        )
        for name in (
            "use_tma_a",
            "use_tma_as",
            "use_tma_b",
            "use_tma_c",
            "use_tma_bs",
            "use_tma_bs2",
            "use_tma_bzp",
            "use_tma_bias",
        )
    }
    use_mbarrier_value = candidate.get("use_mbarrier")
    use_mbarrier = (
        use_tma or use_warp_spec
        if use_mbarrier_value is None
        else bool(use_mbarrier_value)
    )
    num_load_threads = 128 if use_warp_spec else 0
    num_threads = num_math_threads + num_load_threads
    if num_threads > 1024:
        reasons.append(f"num_threads={num_threads} exceeds the CTA limit 1024")
    if use_warp_spec and num_math_threads % 128:
        reasons.append(
            "warp specialization requires a multiple of 128 math threads, "
            f"got {num_math_threads}"
        )
    if shapes_positive:
        mma_k = 256 // problem.layer_config.a_dtype.num_bits
        warp_iters = (
            warp_shape[1] // 16
            if problem.layer_config.use_packed_k_layout
            else warp_shape[2] // mma_k
        )
        if use_warp_spec and warp_iters < 2:
            reasons.append(
                f"warp specialization requires at least two warp iterations, got {warp_iters}"
            )
    if (use_warp_spec or use_tma) and not use_mbarrier:
        reasons.append("warp specialization and TMA require mbarrier synchronization")
    if (use_warp_spec or use_tma) and problem.device.sm_version < 90:
        reasons.append(
            f"warp specialization and TMA require SM90, got SM{problem.device.sm_version}"
        )
    if use_mbarrier and problem.device.sm_version < 80:
        reasons.append(f"mbarrier requires SM80, got SM{problem.device.sm_version}")
    if bool(candidate.get("use_cp_async", False)) and problem.device.sm_version < 80:
        reasons.append(f"cp.async requires SM80, got SM{problem.device.sm_version}")
    if not use_tma:
        enabled_tma_fields = [name for name, enabled in tma_fields.items() if enabled]
        if enabled_tma_fields:
            reasons.append(
                f"TMA transfer fields require use_tma=True: {enabled_tma_fields}"
            )
    if problem.gemm_type == GemmType.INDEXED:
        indexed_tma_fields = [
            name
            for name in ("use_tma_a", "use_tma_as", "use_tma_c")
            if tma_fields[name]
        ]
        if indexed_tma_fields:
            reasons.append(
                "indexed GEMM does not support TMA A/AS/C transfers: "
                f"{indexed_tma_fields}"
            )
        if bool(candidate.get("reduce_overlap_last_stage_only", False)):
            reasons.append("indexed GEMM does not support overlap-last-stage reduction")
    if tma_fields["use_tma_as"] and not problem.use_m_major_input_scale:
        reasons.append("TMA input-scale loads require M-major input scales")
    if num_write_splits > 1 and shapes_positive:
        if (
            block_shape[0] != warp_shape[0]
            or block_shape[0] % 32
            or tma_fields["use_tma_c"]
        ):
            reasons.append(
                "split output writes require block_m=warp_m, block_m divisible "
                "by 32, and direct output stores"
            )
    if (
        problem.layer_config.has_zero_point
        and problem.layer_config.is_fp_zero_point
        and tma_fields["use_tma_bzp"]
        and shapes_positive
        and block_shape[1] > 256
    ):
        reasons.append("TMA float zero-point loads require block_n <= 256")
    if problem.layer_config.mma_type == MmaType.WGMMA and num_stages < 3:
        reasons.append(f"WGMMA requires at least three stages, got {num_stages}")

    has_multicast = multi_cast_size_a > 1 or multi_cast_size_b > 1
    if has_multicast:
        if problem.device.sm_version not in (90, 100, 103):
            reasons.append(f"multicast is unsupported on SM{problem.device.sm_version}")
        if problem.gemm_type != GemmType.DENSE:
            reasons.append("multicast requires a dense GEMM")
        if not use_warp_spec:
            reasons.append("multicast requires warp specialization")
        if multi_cast_size_a > 1 and multi_cast_size_b > 1:
            reasons.append("simultaneous A and B multicast is unsupported")
        if multi_cast_size_a > 1 and not bool(candidate.get("use_tma_a", use_tma)):
            reasons.append("A multicast requires TMA-A")
        if multi_cast_size_b > 1 and not bool(candidate.get("use_tma_b", use_tma)):
            reasons.append("B multicast requires TMA-B")

    smem_size = 0
    if shapes_positive and num_stages > 0 and num_write_splits > 0:
        smem_size = estimate_smem_size_layer(
            problem.layer_config,
            block_shape,
            problem.gemm_type,
            num_stages,
            warp_shape=warp_shape,
            reduce_overlap_last_stage_only=bool(
                candidate.get("reduce_overlap_last_stage_only", False)
            ),
            use_mbarrier=use_mbarrier,
            use_warp_spec=use_warp_spec,
            num_write_splits=num_write_splits,
            mma_accum_bits=16 if problem.use_f16_accum else 32,
        )
        if (
            problem.device.max_smem_size > 0
            and smem_size > problem.device.max_smem_size
        ):
            reasons.append(
                f"smem_size={smem_size} exceeds device limit "
                f"{problem.device.max_smem_size}"
            )

    num_output_tiles = 0
    if (
        problem.shape_m > 0
        and problem.layer_config.shape_n > 0
        and block_shape[0] > 0
        and block_shape[1] > 0
        and problem.layer_config.shape_n % block_shape[1] == 0
    ):
        num_output_tiles = (
            problem.layer_config.shape_n
            // block_shape[1]
            * problem.estimate_num_blocks_m(block_shape[0])
        )

    residency_limits: list[int] = []
    if num_threads > 0 and problem.device.max_threads_per_sm > 0:
        residency_limits.append(problem.device.max_threads_per_sm // num_threads)
    if smem_size > 0 and problem.device.resident_smem_size > 0:
        residency_limits.append(problem.device.resident_smem_size // smem_size)
    thread_smem_cta_limit = min(residency_limits, default=0)
    if num_ctas_per_sm > 0 and thread_smem_cta_limit < num_ctas_per_sm:
        reasons.append(
            f"num_ctas_per_sm={num_ctas_per_sm} exceeds thread/SMEM residency "
            f"limit {thread_smem_cta_limit}"
        )

    waves = None
    if (
        num_output_tiles > 0
        and problem.device.num_sms is not None
        and problem.device.num_sms > 0
        and num_ctas_per_sm > 0
    ):
        waves = math.ceil(num_output_tiles / (problem.device.num_sms * num_ctas_per_sm))

    return CandidateAnalysis(
        candidate=candidate,
        rejection_reasons=tuple(reasons),
        num_math_threads=num_math_threads,
        num_load_threads=num_load_threads,
        num_threads=num_threads,
        smem_size=smem_size,
        num_output_tiles=num_output_tiles,
        thread_smem_cta_limit=thread_smem_cta_limit,
        waves=waves,
    )


def fit_pipeline_stages(
    problem: TuningProblem,
    candidate: ScheduleCandidate,
) -> ScheduleCandidate:
    while candidate.num_stages > 3:
        analysis = analyze_candidate(problem, candidate)
        if analysis.smem_size <= problem.device.max_smem_size:
            break
        candidate = candidate.with_updates(num_stages=candidate.num_stages - 1)
    return candidate
