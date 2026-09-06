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

    def __post_init__(self) -> None:
        if self.num_sms is not None and self.num_sms <= 0:
            raise ValueError("num_sms must be positive")
        for name in ("max_smem_size", "max_threads_per_sm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_smem_per_sm is not None and self.max_smem_per_sm <= 0:
            raise ValueError("max_smem_per_sm must be positive")

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

    def __post_init__(self) -> None:
        if self.shape_m <= 0:
            raise ValueError("shape_m must be positive")
        if self.layer_config.shape_n <= 0 or self.layer_config.shape_k <= 0:
            raise ValueError("layer N and K dimensions must be positive")

    def estimate_num_blocks_m(self, block_shape_m: int) -> int:
        if self.gemm_type == GemmType.DENSE or not self.layer_config.num_experts:
            return math.ceil(self.shape_m / block_shape_m)
        return min(self.shape_m, self.layer_config.num_experts)


def estimate_indexed_m_blocks_uniform(
    shape_m: int,
    num_experts: int,
    block_shape_m: int,
) -> int:
    """Estimate Indexed MoE M tiles from deterministic uniform routing."""
    if shape_m <= 0:
        raise ValueError("shape_m must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if block_shape_m <= 0:
        raise ValueError("block_shape_m must be positive")

    base, remainder = divmod(shape_m, num_experts)
    return (
        (num_experts - remainder) * math.ceil(base / block_shape_m)
        + remainder * math.ceil((base + 1) / block_shape_m)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ScheduleCandidate:
    candidate_id: str
    block_shape: tuple[int, int, int]
    warp_shape: tuple[int, int, int]
    use_stream_k: bool = True
    use_f16_accum: bool = False
    num_stages: int = 2
    num_ctas_per_sm: int = 1
    multi_cast_size_a: int = 1
    use_warp_spec: bool = False
    use_tma: bool = False
    use_mbarrier: bool = False
    _explicit_fields: frozenset[str] = dataclasses.field(
        default_factory=frozenset, repr=False
    )

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self._explicit_fields:
            fields = (
                field.name
                for field in dataclasses.fields(self)
                if not field.name.startswith("_") and field.name != "candidate_id"
            )
            object.__setattr__(self, "_explicit_fields", frozenset(fields))

    @classmethod
    def from_config(
        cls,
        candidate_id: str,
        config: Mapping[str, Any],
    ) -> "ScheduleCandidate":
        if not candidate_id:
            raise ValueError("candidate_id must not be empty")
        if "block_shape" not in config or "warp_shape" not in config:
            raise ValueError("block_shape and warp_shape are required")

        known_fields = {
            field.name
            for field in dataclasses.fields(cls)
            if not field.name.startswith("_") and field.name != "candidate_id"
        }
        unknown_fields = set(config) - known_fields
        if unknown_fields:
            raise ValueError(f"unsupported candidate fields: {sorted(unknown_fields)}")
        use_tma = _config_bool(config, "use_tma", False)
        use_warp_spec = _config_bool(config, "use_warp_spec", False)
        return cls(
            candidate_id=candidate_id,
            block_shape=_positive_shape(config["block_shape"], "block_shape"),
            warp_shape=_positive_shape(config["warp_shape"], "warp_shape"),
            use_stream_k=_config_bool(config, "use_stream_k", True),
            use_f16_accum=_config_bool(config, "use_f16_accum", False),
            num_stages=_config_positive_int(config, "num_stages", 2),
            num_ctas_per_sm=_config_positive_int(config, "num_ctas_per_sm", 1),
            multi_cast_size_a=_config_positive_int(config, "multi_cast_size_a", 1),
            use_warp_spec=use_warp_spec,
            use_tma=use_tma,
            use_mbarrier=_config_bool(
                config,
                "use_mbarrier",
                use_tma or use_warp_spec,
                allow_none=True,
            ),
            _explicit_fields=frozenset(config),
        )

    def to_config(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name in self._explicit_fields
        }

    def get(self, name: str, default: Any = None) -> Any:
        if name in self._explicit_fields:
            return getattr(self, name)
        return default

    def with_updates(
        self,
        candidate_id: str | None = None,
        **updates: Any,
    ) -> "ScheduleCandidate":
        config = self.to_config()
        config.update(updates)
        return type(self).from_config(candidate_id or self.candidate_id, config)


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateAnalysis:
    candidate: ScheduleCandidate
    hard_violations: tuple[str, ...]
    resource_violations: tuple[str, ...]
    num_math_threads: int
    num_load_threads: int
    num_threads: int
    smem_size: int
    num_output_tiles: int
    thread_smem_cta_limit: int
    waves: int | None

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return self.hard_violations + self.resource_violations

    @property
    def launchable(self) -> bool:
        return not self.hard_violations

    @property
    def meets_resource_target(self) -> bool:
        return not self.resource_violations

    @property
    def legal(self) -> bool:
        """Compatibility alias for a launchable candidate at requested residency."""
        return self.launchable and self.meets_resource_target


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


def _positive_shape(value: Any, name: str) -> tuple[int, int, int]:
    shape = _require_shape(value, name)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{name} dimensions must be positive")
    return shape


def _config_positive_int(config: Mapping[str, Any], name: str, default: int) -> int:
    value = _require_int(config.get(name, default), name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _config_bool(
    config: Mapping[str, Any],
    name: str,
    default: bool,
    *,
    allow_none: bool = False,
) -> bool:
    value = config.get(name, default)
    if value is None and allow_none:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclasses.dataclass(frozen=True, slots=True)
class _GeometryAnalysis:
    rejection_reasons: tuple[str, ...]
    ratios: tuple[int, int, int] | None


@dataclasses.dataclass(frozen=True, slots=True)
class _ExecutionAnalysis:
    rejection_reasons: tuple[str, ...]
    num_math_threads: int
    num_load_threads: int
    num_threads: int


@dataclasses.dataclass(frozen=True, slots=True)
class _ResourceAnalysis:
    hard_violations: tuple[str, ...]
    residency_violations: tuple[str, ...]
    smem_size: int
    num_output_tiles: int
    thread_smem_cta_limit: int
    waves: int | None


# Validate only invariants exercised by the migrated SM90 WGMMA policies.
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
    if layer_config.mma_type != MmaType.WGMMA:
        return tuple(reasons)
    if layer_config.a_dtype.num_bits != 16 and layer_config.as_dtype != dtypes.float32:
        reasons.append(
            f"WGMMA input scales must use float32 storage, got {layer_config.as_dtype}"
        )
    mma_k = 256 // layer_config.a_dtype.num_bits
    scale_groups = (
        ("input scale group", layer_config.input_scale_group_size),
        ("weight scale group", layer_config.weight_scale_group_size),
    )
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
        and layer_config.input_scale_group_size != layer_config.weight_scale_group_size
    ):
        reasons.append("block input and weight scale groups must match")
    return tuple(reasons)


def _analyze_geometry(
    layer_config: LayerConfig,
    block_shape: tuple[int, int, int],
    warp_shape: tuple[int, int, int],
) -> _GeometryAnalysis:
    reasons: list[str] = []
    for name, shape in (("block_shape", block_shape), ("warp_shape", warp_shape)):
        if any(size <= 0 for size in shape):
            reasons.append(f"{name} dimensions must be positive: {shape}")
    if reasons:
        return _GeometryAnalysis(tuple(reasons), None)

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
    if block_shape[0] > 256:
        reasons.append(f"block_m={block_shape[0]} exceeds 256")
    for name, size in (
        ("block_n", block_shape[1]),
        ("block_k", block_shape[2]),
        ("warp_n", warp_shape[1]),
        ("warp_k", warp_shape[2]),
    ):
        if not _is_power_of_two(size):
            reasons.append(f"{name}={size} must be a power of two")
    if warp_shape[0] % 8:
        reasons.append(f"warp_m={warp_shape[0]} must be divisible by 8")
    if warp_shape[1] > 64:
        reasons.append(f"warp_n={warp_shape[1]} exceeds 64")
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

    min_warp_n = 32 if layer_config.a_dtype.num_bits == 16 else 16
    if warp_shape[1] < min_warp_n:
        reasons.append(f"warp_n={warp_shape[1]} is smaller than minimum {min_warp_n}")
    min_warp_k = {16: 32, 8: 64, 4: 128}[layer_config.a_dtype.num_bits]
    if warp_shape[2] < min_warp_k:
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
    return _GeometryAnalysis(tuple(reasons), ratios)


def get_geometry_rejection_reasons(
    layer_config: LayerConfig,
    block_shape: tuple[int, int, int],
    warp_shape: tuple[int, int, int],
) -> tuple[str, ...]:
    return _analyze_geometry(
        layer_config,
        block_shape,
        warp_shape,
    ).rejection_reasons


def _get_tile_rejection_reasons(
    problem: TuningProblem,
    schedule: ScheduleCandidate,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for scale_name, group_size in (
        ("input scale group", problem.layer_config.input_scale_group_size),
        ("weight scale group", problem.layer_config.weight_scale_group_size),
    ):
        if (
            group_size
            and group_size % schedule.block_shape[2]
            and schedule.block_shape[2] % group_size
        ):
            reasons.append(
                f"block_k={schedule.block_shape[2]} and "
                f"{scale_name}={group_size} do not nest"
            )
    if schedule.multi_cast_size_a > 0 and problem.layer_config.shape_n % (
        schedule.block_shape[1] * schedule.multi_cast_size_a
    ):
        reasons.append(
            f"shape_n={problem.layer_config.shape_n} is not divisible by "
            f"block_n * multi_cast_size_a="
            f"{schedule.block_shape[1] * schedule.multi_cast_size_a}"
        )
    if problem.use_batch_invariant and (
        schedule.use_stream_k or schedule.block_shape[2] != schedule.warp_shape[2]
    ):
        reasons.append(
            "batch-invariant schedules require direct output and one warp-K tile"
        )
    return tuple(reasons)


def _analyze_execution(
    problem: TuningProblem,
    schedule: ScheduleCandidate,
    geometry: _GeometryAnalysis,
) -> _ExecutionAnalysis:
    reasons: list[str] = []
    num_math_threads = (
        math.prod(geometry.ratios) * 32 if geometry.ratios is not None else 0
    )
    num_load_threads = 128 if schedule.use_warp_spec else num_math_threads
    num_threads = (
        num_math_threads + num_load_threads
        if schedule.use_warp_spec
        else num_math_threads
    )
    if num_threads > 1024:
        reasons.append(f"num_threads={num_threads} exceeds the CTA limit 1024")
    if schedule.use_warp_spec and num_math_threads % 128:
        reasons.append(
            "warp specialization requires a multiple of 128 math threads, "
            f"got {num_math_threads}"
        )
    if (schedule.use_warp_spec or schedule.use_tma) and not schedule.use_mbarrier:
        reasons.append("warp specialization and TMA require mbarrier synchronization")
    if problem.layer_config.mma_type == MmaType.WGMMA and schedule.num_stages < 3:
        reasons.append(
            f"WGMMA requires at least three stages, got {schedule.num_stages}"
        )

    if schedule.multi_cast_size_a > 1:
        if problem.gemm_type != GemmType.DENSE:
            reasons.append("multicast requires a dense GEMM")
        if not schedule.use_warp_spec:
            reasons.append("multicast requires warp specialization")
        if not schedule.use_tma:
            reasons.append("multicast requires TMA")
    return _ExecutionAnalysis(
        rejection_reasons=tuple(reasons),
        num_math_threads=num_math_threads,
        num_load_threads=num_load_threads,
        num_threads=num_threads,
    )


def _analyze_resources(
    problem: TuningProblem,
    schedule: ScheduleCandidate,
    execution: _ExecutionAnalysis,
) -> _ResourceAnalysis:
    hard_violations: list[str] = []
    residency_violations: list[str] = []
    smem_size = estimate_smem_size_layer(
        problem.layer_config,
        schedule.block_shape,
        problem.gemm_type,
        schedule.num_stages,
        warp_shape=schedule.warp_shape,
        reduce_overlap_last_stage_only=False,
        use_mbarrier=schedule.use_mbarrier,
        use_warp_spec=schedule.use_warp_spec,
        num_write_splits=1,
        mma_accum_bits=16 if problem.use_f16_accum else 32,
    )
    if smem_size > problem.device.max_smem_size:
        hard_violations.append(
            f"smem_size={smem_size} exceeds device limit {problem.device.max_smem_size}"
        )

    num_output_tiles = 0
    if (
        problem.shape_m > 0
        and problem.layer_config.shape_n > 0
        and schedule.block_shape[0] > 0
        and schedule.block_shape[1] > 0
        and problem.layer_config.shape_n % schedule.block_shape[1] == 0
    ):
        if (
            problem.device.sm_version == 90
            and problem.gemm_type == GemmType.INDEXED
            and problem.layer_config.num_experts > 0
            and problem.layer_config.a_dtype.num_bits == 16
            and problem.layer_config.b_dtype.num_bits == 4
            and not problem.use_batch_invariant
        ):
            num_m_blocks = estimate_indexed_m_blocks_uniform(
                problem.shape_m,
                problem.layer_config.num_experts,
                schedule.block_shape[0],
            )
        else:
            num_m_blocks = problem.estimate_num_blocks_m(schedule.block_shape[0])
        num_output_tiles = (
            problem.layer_config.shape_n
            // schedule.block_shape[1]
            * num_m_blocks
        )

    residency_limits: list[int] = []
    if execution.num_threads > 0:
        residency_limits.append(
            problem.device.max_threads_per_sm // execution.num_threads
        )
    if smem_size > 0:
        residency_limits.append(problem.device.resident_smem_size // smem_size)
    thread_smem_cta_limit = min(residency_limits, default=0)
    if thread_smem_cta_limit < schedule.num_ctas_per_sm:
        residency_violations.append(
            f"num_ctas_per_sm={schedule.num_ctas_per_sm} exceeds "
            "thread/SMEM residency "
            f"limit {thread_smem_cta_limit}"
        )

    waves = None
    if num_output_tiles > 0 and problem.device.num_sms is not None:
        waves = math.ceil(
            num_output_tiles / (problem.device.num_sms * schedule.num_ctas_per_sm)
        )

    return _ResourceAnalysis(
        hard_violations=tuple(hard_violations),
        residency_violations=tuple(residency_violations),
        smem_size=smem_size,
        num_output_tiles=num_output_tiles,
        thread_smem_cta_limit=thread_smem_cta_limit,
        waves=waves,
    )


def analyze_candidate(
    problem: TuningProblem,
    candidate: ScheduleCandidate,
) -> CandidateAnalysis:
    geometry = _analyze_geometry(
        problem.layer_config,
        candidate.block_shape,
        candidate.warp_shape,
    )
    execution = _analyze_execution(problem, candidate, geometry)
    resources = _analyze_resources(problem, candidate, execution)
    hard_violations = (
        get_problem_rejection_reasons(problem.layer_config)
        + geometry.rejection_reasons
        + _get_tile_rejection_reasons(problem, candidate)
        + execution.rejection_reasons
        + resources.hard_violations
    )

    return CandidateAnalysis(
        candidate=candidate,
        hard_violations=hard_violations,
        resource_violations=resources.residency_violations,
        num_math_threads=execution.num_math_threads,
        num_load_threads=execution.num_load_threads,
        num_threads=execution.num_threads,
        smem_size=resources.smem_size,
        num_output_tiles=resources.num_output_tiles,
        thread_smem_cta_limit=resources.thread_smem_cta_limit,
        waves=resources.waves,
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
