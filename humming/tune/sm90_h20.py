from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.tune.base import DeviceHeuristics
from humming.tune.candidate import TuningProblem
from humming.tune.sm90_h20_families import (
    build_h20_seed_config,
    fused_e8m0_dense_in_scope,
    fused_e8m0_moe_in_scope,
    make_h20_device_profile,
    select_fused_e8m0_dense,
    select_fused_e8m0_moe,
)


class Sm90H20Heuristics(DeviceHeuristics):
    max_smem_size: int = 227 * 1024
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 90

    @classmethod
    def _make_problem(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool,
        use_batch_invariant: bool,
        gemm_type: GemmType,
    ) -> TuningProblem:
        return TuningProblem(
            layer_config=layer_config,
            shape_m=shape_m,
            gemm_type=gemm_type,
            device=make_h20_device_profile(cls.get_num_sms()),
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
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
        problem = cls._make_problem(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        if fused_e8m0_moe_in_scope(
            layer_config, shape_m, gemm_type, use_batch_invariant
        ):
            return select_fused_e8m0_moe(problem).to_config()
        if fused_e8m0_dense_in_scope(
            layer_config, shape_m, gemm_type, use_batch_invariant
        ):
            return select_fused_e8m0_dense(problem).to_config()
        return build_h20_seed_config(problem)
