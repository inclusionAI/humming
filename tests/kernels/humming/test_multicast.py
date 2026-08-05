import torch

import humming.testing.runner as runner_module
from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType
from humming.testing import KernelTestCase, KernelTestRunner, skip_if_unsupported


def test_tma_a_multicast_waits_for_receiver_stage(monkeypatch):
    skip_if_unsupported(
        a_dtype=dtypes.float8e4m3,
        mma_type=MmaType.WGMMA.value,
        use_tma=True,
        use_warp_spec=True,
        use_mbarrier=True,
    )
    tuning_values = {
        "block_shape": (128, 128, 128),
        "warp_shape": (128, 16, 128),
        "use_stream_k": True,
        "use_f16_accum": False,
        "num_stages": 4,
        "use_warp_spec": True,
        "use_tma": True,
        "use_mbarrier": True,
        "multi_cast_size_a": 2,
        "raster_group_m": 1,
    }

    def forced_configs(_layer_config, _compute_config, shape_ms):
        return [dict(tuning_values) for _ in shape_ms]

    monkeypatch.setattr(runner_module, "generate_heuristics_configs", forced_configs)
    case = KernelTestCase(
        name="tma-a-multicast-receiver-stage",
        layer_config=LayerConfig(
            shape_n=3072,
            shape_k=4096,
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float4e2m1,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.float8e8m0,
            input_scale_group_size=128,
            weight_scale_group_size=32,
            mma_type=MmaType.WGMMA,
        ),
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        seed=2026,
    )

    (result,) = KernelTestRunner(case).run(shape_ms=(512,))
    assert result.tuning_config.multi_cast_size_a == 2
    torch.testing.assert_close(
        result.outputs,
        result.outputs_ref,
        rtol=case.rtol,
        atol=case.atol,
    )
