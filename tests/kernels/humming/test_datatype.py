import pytest

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.config.mma import get_default_mma_type
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)

SHAPE_N = 2048
SHAPE_K = 2048
A_DTYPES = (
    "float16",
    "bfloat16",
    "float8e4m3",
    "float8e5m2",
    "float8e3m4",
    "int8",
    "int4",
)

B_DTYPES = (
    "uint1",
    "uint2",
    "uint3",
    "uint4",
    "uint5",
    "uint6",
    "uint7",
    "uint8",
    "int4",
    "int8",
    "float2e0m1",
    "float3e0m2",
    "float3e1m1",
    "float3e2m0",
    "float4e0m3",
    "float4e2m1",
    "float4e3m0",
    "float5e2m2",
    "float5e4m0",
    "float6e2m3",
    "float6e4m1",
    "float7e0m6",
    "float7e2m4",
    "float7e4m2",
    "float7e6m0",
    "float8e1m6",
    "float8e4m3",
    "float8e5m2",
)

C_DTYPES = ("float16", "bfloat16")


def _is_compatible(a_dtype, b_dtype) -> bool:
    if b_dtype.num_bits > a_dtype.num_bits:
        return False

    if b_dtype.is_integer_type and a_dtype.is_integer_type:
        if a_dtype.num_bits == b_dtype.num_bits:
            return a_dtype == b_dtype
        return not b_dtype.is_signed

    if b_dtype.is_integer_type and a_dtype.is_floating_point_type:
        return not b_dtype.is_signed and b_dtype.num_bits <= a_dtype.mantissa_bits + 2

    if b_dtype.is_floating_point_type and a_dtype.is_floating_point_type:
        return (
            b_dtype.is_signed
            and b_dtype.exponent_bits <= a_dtype.exponent_bits
            and b_dtype.mantissa_bits <= a_dtype.mantissa_bits
            and (a_dtype.exponent_bits == 0 or b_dtype.exponent_bits >= 1)
        )

    return False


def _make_cases() -> list[KernelTestCase]:
    cases = []
    signatures = set()
    compute_config = ComputeConfig(gemm_type=GemmType.DENSE)

    for a_dtype_str in A_DTYPES:
        a_dtype = dtypes.DataType.from_str(a_dtype_str)
        for b_dtype_str in B_DTYPES:
            for c_dtype_str in C_DTYPES:
                c_dtype = dtypes.DataType.from_str(c_dtype_str)
                if a_dtype.num_bits == 16 and a_dtype != c_dtype:
                    continue
                if a_dtype == dtypes.float8e5m2 and c_dtype == dtypes.float16:
                    continue

                layer_config = LayerConfig(
                    shape_n=SHAPE_N,
                    shape_k=SHAPE_K,
                    a_dtype=a_dtype,
                    b_dtype=dtypes.DataType.from_str(b_dtype_str),
                    c_dtype=c_dtype,
                    bs_dtype=c_dtype,
                    input_scale_group_size=0,
                    weight_scale_group_size=0,
                )
                if not _is_compatible(layer_config.a_dtype, layer_config.b_dtype):
                    continue

                signature = (
                    str(layer_config.a_dtype),
                    str(layer_config.b_dtype),
                    str(layer_config.c_dtype),
                )
                if signature in signatures:
                    continue
                signatures.add(signature)

                name = "-".join(signature)
                case = KernelTestCase(
                    name=name,
                    layer_config=layer_config,
                    compute_config=compute_config,
                    seed=2026,
                    input_std_scale=0.05 if layer_config.b_dtype == dtypes.float8e5m2 else 1.0,
                )
                cases.append(case)

    return cases


DATATYPE_CASES = _make_cases()


@pytest.mark.parametrize("test_case", DATATYPE_CASES, ids=str)
def test_datatype(test_case):
    skip_if_unsupported(
        a_dtype=test_case.layer_config.a_dtype,
        mma_type=get_default_mma_type(test_case.layer_config).value,
    )
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_datatype_case_coverage():
    signatures = {
        (
            str(case.layer_config.a_dtype),
            str(case.layer_config.b_dtype),
            str(case.layer_config.c_dtype),
        )
        for case in DATATYPE_CASES
    }

    for a_dtype in A_DTYPES:
        assert any(signature[0] == a_dtype for signature in signatures)
    for bit_width in range(1, 9):
        assert any(case.layer_config.b_dtype.num_bits == bit_width for case in DATATYPE_CASES)
    assert {signature[2] for signature in signatures} == set(C_DTYPES)


@pytest.mark.parametrize("a_dtype", ("float8e4m3", "int8", "int4"))
@pytest.mark.parametrize("use_tma", (False, True))
@pytest.mark.parametrize("block_k", (64, 128, 256))
def test_raw_weight_backends(a_dtype, use_tma, block_k, monkeypatch):
    import dataclasses

    import torch

    from humming.config import MmaType
    from humming.device import current_device
    from humming.tune.sm100 import Sm100UmmaHeuristics

    skip_if_unsupported(a_dtype=a_dtype, use_tma=use_tma)
    dtype = dtypes.DataType.from_str(a_dtype)
    if block_k * dtype.num_bits < 512:
        pytest.skip("the shared-memory row must contain at least 64 bytes")
    config = LayerConfig(
        shape_n=256,
        shape_k=512,
        a_dtype=dtype,
        b_dtype=dtype,
        c_dtype=dtypes.bfloat16,
    )
    assert "mma_type" not in config.to_dict()
    assert config.use_raw_weight
    test_case = KernelTestCase(
        name="raw-weight-backends",
        layer_config=config,
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
    )
    runner = KernelTestRunner(test_case)
    weight = runner.kernel_tensors["weight"]
    assert weight.shape == (256, 512 * dtype.num_bits // 32)
    original_weight = weight.clone()
    layer_values = config.to_dict()

    mma_types = [MmaType.MMA]
    if current_device.sm_version == 90 and dtype != dtypes.int4:
        mma_types.append(MmaType.WGMMA)
    if config.is_umma_supported:
        mma_types.append(MmaType.UMMA)
    if config.use_block_scaled_mma:
        mma_types = [get_default_mma_type(config)]

    for mma_type in mma_types:

        def select_config(layer_config, shape_m, gemm_type, mma_type=mma_type, **kwargs):
            if mma_type == MmaType.UMMA:
                return Sm100UmmaHeuristics.get_config(layer_config, shape_m, gemm_type=gemm_type)
            use_wgmma = mma_type == MmaType.WGMMA
            warp_n = 16 if use_wgmma else 32
            return dict(
                mma_type=mma_type.value,
                block_shape=(16, 128 if use_wgmma else 64, block_k),
                warp_shape=(16, warp_n, min(block_k, 128) if use_wgmma else block_k),
                num_stages=3,
                use_tma=use_tma,
                use_stream_k=False,
            )

        monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", select_config)
        results = runner.run((17, 129))
        assert all(result.tuning_config.mma_type == mma_type for result in results)
        assert config.to_dict() == layer_values
        torch.testing.assert_close(weight, original_weight, rtol=0, atol=0)
        assert "mma_type" not in {field.name for field in dataclasses.fields(config)}


@pytest.mark.parametrize(
    "sm_version,mma_type,a_dtype,b_dtype,block_k",
    (
        (80, "mma", "int4", "int4", 128),
        (90, "mma", "float8e4m3", "float8e4m3", 64),
        (90, "wgmma", "float8e4m3", "float8e4m3", 64),
        (90, "wgmma", "float8e4m3", "float8e4m3", 128),
        (90, "wgmma", "int8", "int8", 128),
        (120, "mxmma", "float4e2m1", "float4e2m1", 128),
        (120, "mxmma", "float8e4m3", "float8e4m3", 64),
        (90, "mma", "float16", "uint3", 64),
        (90, "wgmma", "float16", "uint3", 64),
        (90, "mma", "int8", "uint4", 128),
        (90, "wgmma", "int8", "uint4", 128),
        (90, "mma", "float8e4m3", "uint5", 128),
        (90, "wgmma", "float8e4m3", "uint5", 128),
    ),
)
def test_weight_cross_architecture_compiles(sm_version, mma_type, a_dtype, b_dtype, block_k, monkeypatch):
    from humming.kernel.humming import HummingKernel

    def init_sm_version(kernel):
        kernel.sm_version_str = f"{kernel.sm_version}{'a' if kernel.sm_version >= 90 else ''}"

    monkeypatch.setattr(HummingKernel, "init_sm_version", init_sm_version)
    monkeypatch.setattr(HummingKernel, "register_kernel", lambda kernel: None)
    use_wgmma = mma_type == "wgmma"
    activation_bits = dtypes.DataType.from_str(a_dtype).num_bits
    warp_n = 16 if use_wgmma and activation_bits < 16 else 32
    kernel = HummingKernel(
        sm_version=sm_version,
        shape_n=256,
        shape_k=512,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=a_dtype if activation_bits == 16 else "bfloat16",
        weight_scale_group_size=64 if a_dtype != b_dtype else 0,
        use_packed_k_layout=False,
        mma_type=mma_type,
        block_shape=(16, 128 if use_wgmma else 64, block_k),
        warp_shape=(16, warp_n, block_k),
        num_stages=3,
        use_tma=sm_version >= 90,
        use_stream_k=False,
    )
    assert kernel.use_raw_weight == (a_dtype == b_dtype)
    assert "kMmaType" not in kernel.to_cpp_str(LayerConfig)


@pytest.mark.parametrize("weight_bits", (3, 4, 7))
def test_ordinary_repack_is_shared_with_sm90(weight_bits):
    import torch

    from humming import ops
    from humming.transform import transform_humming_tensors

    torch.manual_seed(2026)
    values = torch.randint(0, 1 << weight_bits, (128, 256), device="cuda", dtype=torch.int32)
    scales = torch.arange(1, 513, device="cuda", dtype=torch.float32).reshape(128, 4)
    tensors = {
        "weight": ops.pack_weight(values, weight_bits),
        "weight_scale": scales.to(torch.bfloat16),
    }
    transformed = []
    for sm_version in (80, 90):
        config = LayerConfig(
            shape_n=128,
            shape_k=256,
            a_dtype=dtypes.int8,
            b_dtype=dtypes.DataType.from_str(f"uint{weight_bits}"),
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=64,
            use_packed_k_layout=False,
            use_fused_e8m0_scale=False,
            use_int_weight_scale=False,
            sm_version=sm_version,
        )
        transformed.append(transform_humming_tensors(config, tensors))
    for name in ("weight", "weight_scale"):
        torch.testing.assert_close(transformed[0][name], transformed[1][name], rtol=0, atol=0)


@pytest.mark.parametrize("a_dtype", ("float8e4m3", "int8"))
@pytest.mark.parametrize("warp_n", (16, 32, 64))
@pytest.mark.parametrize("transfer", ("sync", "cp_async", "ws_cp_async", "tma", "ws_tma"))
def test_wgmma_raw_ss_layout(a_dtype, warp_n, transfer, monkeypatch):
    """SS fragments must agree with channel scales, bias, and output layout."""
    from humming.config import MmaType

    skip_if_unsupported(a_dtype=a_dtype, mma_type="wgmma")
    dtype = dtypes.DataType.from_str(a_dtype)
    warp_k = 128
    config = LayerConfig(
        shape_n=256, shape_k=512, a_dtype=dtype, b_dtype=dtype,
        c_dtype=dtypes.bfloat16, has_bias=True,
    )
    tuning = dict(
        mma_type="wgmma", block_shape=(16, warp_n * 4, warp_k * 2),
        warp_shape=(16, warp_n, warp_k), num_stages=3,
        use_tma=transfer in ("tma", "ws_tma"),
        use_cp_async=transfer != "sync", use_warp_spec=transfer in ("ws_cp_async", "ws_tma"),
        use_stream_k=False,
    )
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: dict(tuning))
    case = KernelTestCase(
        name=f"ss-{a_dtype}-{warp_n}-{transfer}", layer_config=config,
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE), seed=2026,
    )
    results = KernelTestRunner(case).run((1, 17, 129))
    assert all(result.tuning_config.mma_type == MmaType.WGMMA for result in results)


@pytest.mark.parametrize("mode", ("fp8_channel", "int8_channel", "group64", "group64_fp8", "block64"))
@pytest.mark.parametrize("use_warp_spec", (False, True))
@pytest.mark.parametrize(
    "block_m,block_n,block_k,warp_n,use_stream_k",
    ((16, 128, 256, 32, False), (16, 256, 128, 32, False),
     (16, 512, 128, 64, False), (8, 128, 512, 32, True)),
)
def test_wgmma_cp_async_n_permute(
    mode, use_warp_spec, block_m, block_n, block_k, warp_n, use_stream_k, monkeypatch,
):
    """Scatter N16 fragments without changing global scale or weight layouts."""
    from humming.config import WeightScale2Type, WeightScaleType

    dtype = "int8" if mode == "int8_channel" else "float8e4m3"
    skip_if_unsupported(a_dtype=dtype, mma_type="wgmma")
    has_group = mode in ("group64", "group64_fp8", "block64")
    scale_values = {}
    if mode == "group64_fp8":
        scale_values.update(bs_dtype="float8e4m3", weight_scale_2_type=WeightScale2Type.CHANNEL)
    elif mode == "block64":
        scale_values.update(
            bs_dtype="float32", weight_scale_type=WeightScaleType.BLOCK, weight_scale_group_size_n=64,
        )
    config = LayerConfig(
        shape_n=1024, shape_k=2048, a_dtype=dtype, b_dtype=dtype,
        c_dtype="bfloat16", has_bias=True,
        input_scale_group_size=64 if has_group else 0,
        weight_scale_group_size=64 if has_group else 0,
        use_int_weight_scale=False, use_fused_e8m0_scale=False, **scale_values,
    )
    tuning = dict(
        mma_type="wgmma", block_shape=(block_m, block_n, block_k),
        warp_shape=(block_m, warp_n, 128), num_stages=3,
        use_tma=False, use_cp_async=True, use_warp_spec=use_warp_spec, use_stream_k=use_stream_k,
    )
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: dict(tuning))
    case = KernelTestCase(
        name=f"cp-async-n-permute-{mode}", layer_config=config,
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE), seed=2026,
    )
    results = KernelTestRunner(case).run((1, 17, 129))
    assert_kernel_test_shape_coverage(results, (1, 17, 129))


@pytest.mark.parametrize("mode", ("fp8_channel", "int8_channel", "group64", "group64_fp8", "block64"))
@pytest.mark.parametrize("use_warp_spec", (False, True))
@pytest.mark.parametrize("multicast", (1, 2, 4))
@pytest.mark.parametrize(
    "block_m,block_n,block_k,warp_n",
    ((16, 256, 256, 32), (16, 512, 128, 64), (8, 128, 512, 32), (16, 128, 256, 16)),
)
def test_wgmma_tma_b_general(
    mode, use_warp_spec, multicast, block_m, block_n, block_k, warp_n, monkeypatch,
):
    """One B TMA covers all K slabs and N warpgroups, including multicast."""
    from humming.config import WeightScale2Type, WeightScaleType

    dtype = "int8" if mode == "int8_channel" else "float8e4m3"
    skip_if_unsupported(a_dtype=dtype, mma_type="wgmma")
    has_group = mode in ("group64", "group64_fp8", "block64")
    scale_values = {}
    if mode == "group64_fp8":
        scale_values.update(bs_dtype="float8e4m3", weight_scale_2_type=WeightScale2Type.CHANNEL)
    elif mode == "block64":
        scale_values.update(
            bs_dtype="float32", weight_scale_type=WeightScaleType.BLOCK, weight_scale_group_size_n=64,
        )
    config = LayerConfig(
        shape_n=1024, shape_k=2048, a_dtype=dtype, b_dtype=dtype, c_dtype="bfloat16", has_bias=True,
        input_scale_group_size=64 if has_group else 0, weight_scale_group_size=64 if has_group else 0,
        use_int_weight_scale=False, use_fused_e8m0_scale=False, **scale_values,
    )
    tuning = dict(
        mma_type="wgmma", block_shape=(block_m, block_n, block_k),
        warp_shape=(block_m, warp_n, 128), num_stages=3, use_tma=True,
        use_warp_spec=use_warp_spec, use_stream_k=False, multi_cast_size_b=multicast,
    )
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: dict(tuning))
    case = KernelTestCase(
        name=f"tma-b-general-{mode}", layer_config=config,
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE), seed=2026,
    )
    if multicast > 1 and not use_warp_spec:
        with pytest.raises(AssertionError, match="multicast requires warp specialization"):
            KernelTestRunner(case).run((1,))
        return
    results = KernelTestRunner(case).run((1, 17, 129))
    assert_kernel_test_shape_coverage(results, (1, 17, 129))


@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    (("float8e4m3", "float8e4m3"), ("int8", "int8"), ("float16", "uint4"), ("bfloat16", "uint4")),
)
@pytest.mark.parametrize("shape_k", (2048, 1920, 1984, 2016))
@pytest.mark.parametrize("multicast", (1, 2, 4))
@pytest.mark.parametrize("use_warp_spec", (False, True))
def test_wgmma_tma_a_general(a_dtype, b_dtype, shape_k, multicast, use_warp_spec, monkeypatch):
    """Cover 8/16-bit slabs, aligned padding and the partial-slab fallback."""
    skip_if_unsupported(a_dtype=a_dtype, mma_type="wgmma")
    is_16bit = a_dtype in ("float16", "bfloat16")
    tuning = dict(
        mma_type="wgmma", block_shape=(16, 128, 256),
        warp_shape=(16, 32, 64 if is_16bit else 128), num_stages=3,
        use_tma=True, use_warp_spec=use_warp_spec, use_stream_k=False,
        multi_cast_size_a=multicast,
    )
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: dict(tuning))
    case = KernelTestCase(
        name="tma-a-general", seed=2026,
        layer_config=LayerConfig(
            shape_n=1024, shape_k=2048, pad_shape_k=2048 - shape_k, a_dtype=a_dtype, b_dtype=b_dtype,
            c_dtype=a_dtype if is_16bit else "bfloat16", has_bias=True,
            use_int_weight_scale=False, use_fused_e8m0_scale=False,
        ),
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
    )
    if multicast > 1 and not use_warp_spec:
        with pytest.raises(AssertionError, match="multicast requires warp specialization"):
            KernelTestRunner(case).run((1,))
        return
    results = KernelTestRunner(case).run((1, 17, 129))
    assert_kernel_test_shape_coverage(results, (1, 17, 129))


@pytest.mark.parametrize(
    "a_dtype,b_dtype,block_k",
    (("float16", "uint4", 128), ("bfloat16", "uint4", 512),
     ("float8e4m3", "uint4", 256), ("int8", "uint4", 256),
     ("float8e4m3", "float8e4m3", 64), ("float16", "uint4", 64)),
)
@pytest.mark.parametrize("use_warp_spec", (False, True))
def test_wgmma_tma_a_general_geometry(a_dtype, b_dtype, block_k, use_warp_spec, monkeypatch):
    if block_k == 512 and use_warp_spec:
        pytest.skip("16-bit BlockK512 already uses 1024 math threads; no room for producer warps")
    skip_if_unsupported(a_dtype=a_dtype, mma_type="wgmma")
    is_16bit = a_dtype in ("float16", "bfloat16")
    tuning = dict(
        mma_type="wgmma", block_shape=(16, 128, block_k),
        warp_shape=(16, 32, min(block_k, 64 if is_16bit else 128)),
        num_stages=3, use_tma=True, use_warp_spec=use_warp_spec, use_stream_k=False,
    )
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: dict(tuning))
    case = KernelTestCase(
        name="tma-a-general-geometry", seed=2026,
        layer_config=LayerConfig(
            shape_n=512, shape_k=1024, a_dtype=a_dtype, b_dtype=b_dtype,
            c_dtype=a_dtype if is_16bit else "bfloat16", has_bias=True,
            weight_scale_group_size=64, use_int_weight_scale=False, use_fused_e8m0_scale=False,
        ),
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
    )
    results = KernelTestRunner(case).run((1, 17, 129))
    assert_kernel_test_shape_coverage(results, (1, 17, 129))
