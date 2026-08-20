from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.tune.base import DeviceHeuristics
from humming.tune.candidate import (
    DeviceProfile,
    TuningDecision,
    TuningProblem,
)
from humming.tune.sm90_policies import (
    Sm90CandidatePolicy,
    build_sm90_seed_config,
    calc_sm90_num_block_list,
    select_grouped_scale,
    select_indexed_a16,
    _use_w4a8_moe_bm_heuristic_v1,
)
from humming.utils.smem import estimate_smem_size_layer


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
    def _uses_indexed_a16_policy(
        cls,
        layer_config: LayerConfig,
        use_batch_invariant: bool,
        gemm_type: GemmType,
    ) -> bool:
        return (
            gemm_type == GemmType.INDEXED
            and layer_config.a_dtype.num_bits == 16
            and not use_batch_invariant
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
        problem = cls._make_problem(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        return build_sm90_seed_config(problem)

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
        if _use_w4a8_moe_bm_heuristic_v1(problem):
            problem = cls._make_problem(
                layer_config,
                shape_m,
                use_f16_accum,
                use_batch_invariant,
                gemm_type,
                include_grid_size=True,
            )
        return select_grouped_scale(problem).to_config()

    @classmethod
    def calc_num_block_list(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        max_block_m: int,
    ):
        return calc_sm90_num_block_list(layer_config, shape_m, max_block_m)

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
        tune_indexed_a16 = cls._uses_indexed_a16_policy(
            layer_config,
            use_batch_invariant,
            gemm_type,
        )
        problem = cls._make_problem(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
            include_grid_size=tune_indexed_a16,
        )
        if cls._uses_grouped_scale_candidates(layer_config):
            if _use_w4a8_moe_bm_heuristic_v1(problem):
                problem = cls._make_problem(
                    layer_config,
                    shape_m,
                    use_f16_accum,
                    use_batch_invariant,
                    gemm_type,
                    include_grid_size=True,
                )
            return select_grouped_scale(problem)
        if not tune_indexed_a16:
            raise ValueError(
                "decision traces are only available for migrated SM90 policies"
            )
        return select_indexed_a16(
            problem,
            cls.candidate_policy,
        )

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        use_candidates = cls._uses_grouped_scale_candidates(
            layer_config
        ) or cls._uses_indexed_a16_policy(layer_config, use_batch_invariant, gemm_type)
        if use_candidates:
            return cls.get_tuning_decision(
                layer_config,
                shape_m,
                use_f16_accum,
                use_batch_invariant,
                gemm_type,
            ).to_config()

        config = cls.get_config1(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        while config["num_stages"] > 3:
            smem_size = estimate_smem_size_layer(
                layer_config,
                config["block_shape"],
                gemm_type,
                config["num_stages"],
                warp_shape=config["warp_shape"],
                reduce_overlap_last_stage_only=config.get(
                    "reduce_overlap_last_stage_only", False
                ),
                use_mbarrier=config.get("use_mbarrier", False),
                use_warp_spec=config.get("use_warp_spec", False),
                num_write_splits=config.get("num_write_splits", 1),
                mma_accum_bits=16 if use_f16_accum else 32,
            )
            if smem_size <= cls.max_smem_size:
                break
            config["num_stages"] -= 1
        return config
