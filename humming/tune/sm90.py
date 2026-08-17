import math

import numpy as np

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.tune.base import DeviceHeuristics
from humming.tune.candidate import (
    DeviceProfile,
    ScheduleCandidate,
    TuningDecision,
    TuningProblem,
    analyze_candidate,
    fit_pipeline_stages,
)
from humming.tune.sm90_families import (
    Sm90CandidatePolicy,
    select_grouped_scale,
    select_indexed_a16,
)


class Sm90Heuristics(DeviceHeuristics):
    max_smem_size: int = 227 * 1024
    candidate_policy = Sm90CandidatePolicy()
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [
        dtypes.int8,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    ]
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 90

    @classmethod
    def get_device_profile(cls, *, include_grid_size: bool) -> DeviceProfile:
        return DeviceProfile(
            name=f"sm{cls.sm_version}",
            sm_version=cls.sm_version,
            num_sms=cls.get_num_sms() if include_grid_size else None,
            max_smem_size=cls.max_smem_size,
        )

    @classmethod
    def _make_problem(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool,
        use_batch_invariant: bool,
        gemm_type: GemmType,
        *,
        include_grid_size: bool = False,
    ) -> TuningProblem:
        return TuningProblem(
            layer_config=layer_config,
            shape_m=shape_m,
            gemm_type=gemm_type,
            device=cls.get_device_profile(include_grid_size=include_grid_size),
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
        )

    @classmethod
    def get_config1(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        tune_indexed_a16 = (
            gemm_type == GemmType.INDEXED
            and layer_config.a_dtype.num_bits == 16
            and not use_batch_invariant
        )
        if layer_config.use_packed_k_layout:
            max_block_m = 128
        elif use_f16_accum:
            max_block_m = 256
        else:
            max_block_m = 176

        if tune_indexed_a16:
            # Bound padding when only a few routed rows land on each expert.
            tokens_per_expert = shape_m / layer_config.num_experts
            first_threshold = 1.01 if layer_config.b_dtype.num_bits == 4 else 0.7
            moe_block_size_configs = (
                (8, first_threshold),
                (16, 0.7),
                (24, 0.8),
                (32, 0.9),
                (48, 0.9),
                (64, 0.9),
            )
            for block_shape_m, threshold in moe_block_size_configs:
                if tokens_per_expert / block_shape_m < threshold:
                    break
        else:
            num_blocks_list = cls.calc_num_block_list(
                layer_config,
                shape_m,
                max_block_m,
            )
            block_shape_m = np.argmin(num_blocks_list).item() * 8 + 8
        warp_shape_n = 32
        warp_shape_k = 1024 // layer_config.a_dtype.num_bits

        # Long-K layers need more routed rows before wider N tiles pay off.
        wide_tile_min_shape_m = 64 if layer_config.shape_k > 4096 else 16
        use_wide_indexed_tile = (
            tune_indexed_a16
            and block_shape_m <= 64
            and shape_m >= wide_tile_min_shape_m
        )
        if use_wide_indexed_tile:
            warp_shape_n = 64
            if layer_config.shape_k <= 512 and layer_config.shape_n >= 2048:
                # Shallow K needs more N work per CTA to amortize scheduling.
                block_shape_n = 512
                block_shape_k = 64
            else:
                block_shape_n = 256
                block_shape_k = 128
        elif (
            layer_config.shape_n <= 4096
            and not use_batch_invariant
            and block_shape_m <= 64
        ):
            block_shape_n = 128
            block_shape_k = warp_shape_k * 2
            if block_shape_m <= 32:
                block_shape_k = block_shape_k * 2
            if block_shape_k > 256:
                block_shape_k = block_shape_k // 2
                warp_shape_k = warp_shape_k // 2

            while layer_config.shape_k % block_shape_k != 0:
                block_shape_k = block_shape_k // 2
        else:
            block_shape_n = 256
            block_shape_k = warp_shape_k
            if block_shape_m <= 32 and layer_config.b_dtype.num_bits <= 6:
                block_shape_k = block_shape_k * 2
            elif block_shape_m <= 32:
                warp_shape_k = warp_shape_k // 2

        min_warp_shape_n = 32 if layer_config.a_dtype.num_bits == 16 else 16
        # Keep a complete four-warp WGMMA group while fitting output width.
        while layer_config.shape_n % block_shape_n != 0:
            block_shape_n //= 2
            assert block_shape_n >= min_warp_shape_n * 4
        warp_shape_n = min(warp_shape_n, block_shape_n // 4)

        # Earlier shape fitting can reduce block K below the initial warp K.
        warp_shape_k = min(warp_shape_k, block_shape_k)
        while layer_config.shape_k % block_shape_k != 0:
            block_shape_k = block_shape_k // 2
            warp_shape_k = min(warp_shape_k, block_shape_k)
            assert block_shape_k >= warp_shape_k

        dense_small_fp4 = (
            gemm_type == GemmType.DENSE
            and layer_config.a_dtype.num_bits == 16
            and layer_config.b_dtype.num_bits == 4
            and shape_m <= 128
            and layer_config.shape_n % 128 == 0
            and layer_config.shape_k % 64 == 0
        )
        if dense_small_fp4:
            block_shape_n = 128
            block_shape_k = 64
            warp_shape_n = 32
            warp_shape_k = 64
        config = {
            "block_shape": (block_shape_m, block_shape_n, block_shape_k),
            "warp_shape": (block_shape_m, warp_shape_n, warp_shape_k),
            "use_stream_k": not use_batch_invariant,
            "use_f16_accum": use_f16_accum,
            "num_stages": 4,
        }

        if gemm_type != GemmType.INDEXED:
            config["use_warp_spec"] = True
            config["use_tma"] = True
            config["use_mbarrier"] = True
            if dense_small_fp4:
                config["num_ctas_per_sm"] = 2

            if (
                layer_config.shape_n % (block_shape_n * 2) == 0
                and shape_m / block_shape_m >= 4
            ):
                if gemm_type == GemmType.DENSE:
                    config["multi_cast_size_a"] = 2

        return config

    @classmethod
    def get_config2(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        problem = cls._make_problem(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        return cls._get_grouped_scale_decision(problem).to_config()

    @classmethod
    def _get_grouped_scale_decision(
        cls,
        problem: TuningProblem,
    ) -> TuningDecision:
        layer_config = problem.layer_config
        if problem.use_f16_accum:
            max_block_m = 256
        elif layer_config.input_scale_group_size > 0:
            max_block_m = 160
        elif layer_config.weight_scale_group_size < 128:
            max_block_m = 192
        else:
            max_block_m = 200

        num_blocks_list = cls.calc_num_block_list(
            layer_config,
            problem.shape_m,
            max_block_m,
        )
        block_shape_m = np.argmin(num_blocks_list).item() * 8 + 8
        return select_grouped_scale(problem, block_shape_m)

    @classmethod
    def calc_num_block_list(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        max_block_m: int,
    ):
        num_blocks_list = []
        if not layer_config.num_experts:
            for i in range(max_block_m // 8):
                block_m = i * 8 + 8
                num_blocks_list.append(math.ceil(shape_m / block_m))
        else:
            random_state = np.random.RandomState(seed=0)
            samples = random_state.randint(0, layer_config.num_experts, size=shape_m)
            counts = np.bincount(samples)
            for i in range(max_block_m // 8):
                block_m = i * 8 + 8
                num_blocks = int(np.ceil(counts * 1.1 / block_m).sum().item())
                num_blocks_list.append(num_blocks)

        for i in range(max_block_m // 8):
            num_blocks = num_blocks_list[i]
            block_m = i * 8 + 8
            if (
                layer_config.a_dtype == dtypes.int8
                and block_m % 16 == 8
                and block_m > 32
            ):
                num_blocks_list[i] = 1000000

        return num_blocks_list

    @classmethod
    def _uses_grouped_scale_candidates(
        cls,
        layer_config: LayerConfig,
    ) -> bool:
        if layer_config.a_dtype.num_bits == 16:
            return False
        if layer_config.use_packed_k_layout:
            return False
        if (
            layer_config.input_scale_group_size == 0
            and layer_config.weight_scale_group_size == 0
        ):
            return False
        return not (
            layer_config.use_fused_e8m0_scale
            and layer_config.input_scale_group_size == 0
        )

    @classmethod
    def get_tuning_decision(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ) -> TuningDecision:
        problem = cls._make_problem(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
            include_grid_size=(
                gemm_type == GemmType.INDEXED
                and layer_config.a_dtype.num_bits == 16
                and not use_batch_invariant
            ),
        )
        if cls._uses_grouped_scale_candidates(layer_config):
            decision = cls._get_grouped_scale_decision(problem)
        else:
            config = cls.get_config1(
                layer_config,
                shape_m,
                use_f16_accum,
                use_batch_invariant,
                gemm_type,
            )
            tune_indexed_a16 = (
                gemm_type == GemmType.INDEXED
                and layer_config.a_dtype.num_bits == 16
                and not use_batch_invariant
            )
            if tune_indexed_a16:
                decision = select_indexed_a16(
                    problem,
                    config,
                    cls.candidate_policy,
                )
            else:
                candidate = fit_pipeline_stages(
                    problem,
                    ScheduleCandidate.from_config(
                        "legacy_sm90",
                        config,
                    ),
                )
                analysis = analyze_candidate(problem, candidate)
                if not analysis.legal:
                    raise AssertionError(analysis.rejection_reasons)
                decision = TuningDecision(
                    problem=problem,
                    family="legacy_sm90",
                    selected=candidate,
                    considered=(analysis,),
                    reason="wrapped the existing SM90 heuristic output",
                )
        return decision

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        use_candidates = cls._uses_grouped_scale_candidates(layer_config) or (
            gemm_type == GemmType.INDEXED
            and layer_config.a_dtype.num_bits == 16
            and not use_batch_invariant
        )
        if use_candidates:
            return cls.get_tuning_decision(
                layer_config,
                shape_m,
                use_f16_accum,
                use_batch_invariant,
                gemm_type,
            ).to_config()

        problem = cls._make_problem(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        config = cls.get_config1(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        return fit_pipeline_stages(
            problem,
            ScheduleCandidate.from_config("legacy_sm90", config),
        ).to_config()
