"""Cold-L2 process_input roofline and semantically matched vLLM comparisons."""

import argparse
import dataclasses
import statistics
from collections.abc import Callable

import torch

from humming.ops.input import process_input

ACTIVATIONS = {
    "relu": ("unary", "a > 0.f ? a : 0.f"),
    "gelu": ("unary", "0.5f * a * (1.f + erff(a * 0.7071067811865475f))"),
    "silu_split": ("binary_split", "a / (1.f + expf(-a)) * b"),
    "silu_interleaved": ("binary_interleaved", "a / (1.f + expf(-a)) * b"),
}


@dataclasses.dataclass
class Case:
    name: str
    humming: Callable[[], object]
    competitor: Callable[[], object]
    semantic_bytes: int
    competitor_kernels: int = 1


@dataclasses.dataclass
class FeatureCase:
    name: str
    baseline: Callable[[], object]
    variant: Callable[[], object]
    baseline_bytes: int
    variant_bytes: int


def _capture(call: Callable[[], object], batch: int) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(batch):
            call()
    return graph


def _measure_functions(
    left: Callable[[], object],
    right: Callable[[], object],
    flush: torch.Tensor,
    repeats: int,
    cache_mode: str,
) -> tuple[float, float]:
    if cache_mode == "hot":
        batch = 100
        graphs = (_capture(left, batch), _capture(right, batch))
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        samples = [[], []]
        events = []
        for repeat in range(repeats):
            order = (0, 1) if not repeat & 1 else (1, 0)
            for index in order:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                graphs[index].replay()
                end.record()
                events.append((index, begin, end))
        torch.cuda.synchronize()
        for index, begin, end in events:
            samples[index].append(begin.elapsed_time(end) * 1000 / batch)
        return statistics.median(samples[0]), statistics.median(samples[1])

    for _ in range(3):
        left()
        right()
    torch.cuda.synchronize()
    samples = [[], []]
    events = []
    calls = (left, right)
    for repeat in range(repeats):
        order = (0, 1) if not repeat & 1 else (1, 0)
        for index in order:
            flush.zero_()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            calls[index]()
            end.record()
            events.append((index, begin, end))
    torch.cuda.synchronize()
    for index, begin, end in events:
        samples[index].append(begin.elapsed_time(end) * 1000)
    return statistics.median(samples[0]), statistics.median(samples[1])


def _profile_cold_features(
    cases: list[FeatureCase], flush: torch.Tensor, repeats: int
) -> list[tuple[float, float]]:
    from torch.profiler import ProfilerActivity, profile

    for case in cases:
        case.baseline()
        case.variant()
    torch.cuda.synchronize()
    order = []
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        for repeat in range(repeats):
            for case_index, case in enumerate(cases):
                sides = (0, 1) if not repeat & 1 else (1, 0)
                calls = (case.baseline, case.variant)
                for side in sides:
                    flush.zero_()
                    calls[side]()
                    order.append((case_index, side))
    torch.cuda.synchronize()

    durations = []
    current = None
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if "FillFunctor<unsigned char>" in event.name:
            if current is not None:
                durations.append(current)
            current = 0.0
        elif current is not None:
            current += event.self_device_time_total
    if current is not None:
        durations.append(current)
    assert len(durations) == len(order), (len(durations), len(order))

    samples = [[[], []] for _ in cases]
    for (case_index, side), duration in zip(order, durations, strict=True):
        samples[case_index][side].append(duration)
    return [(statistics.median(baseline), statistics.median(variant)) for baseline, variant in samples]


def _measure_pair(case: Case, flush: torch.Tensor, repeats: int, cache_mode: str) -> tuple[float, float]:
    return _measure_functions(case.humming, case.competitor, flush, repeats, cache_mode)


def _measure_hbm(repeats: int) -> tuple[float, float]:
    size = 256 * 1024 * 1024
    sources = [torch.empty(size, dtype=torch.uint8, device="cuda") for _ in range(3)]
    targets = [torch.empty_like(source) for source in sources]
    for source, target in zip(sources, targets, strict=True):
        target.copy_(source)
    events = []
    for repeat in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        targets[repeat % len(targets)].copy_(sources[repeat % len(sources)])
        end.record()
        events.append((begin, end))
    torch.cuda.synchronize()
    samples = [begin.elapsed_time(end) * 1000 for begin, end in events]
    latency_us = statistics.median(samples)
    return latency_us, 2 * size / (latency_us * 1e-6) / 1e9


def _raw_activation_case(m: int, k: int, activation: str) -> Case:
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(x)
    if activation == "relu":
        expression = "a > 0.f ? a : 0.f"
    else:
        expression = "0.5f * a * (1.f + erff(a * 0.7071067811865475f))"

    def competitor():
        if activation == "relu":
            return torch.ops.aten.relu.out(x, out=output)
        return torch.ops.aten.gelu.out(x, approximate="none", out=output)

    def humming():
        process_input(
            x,
            outputs=output,
            activation_type="unary",
            activation_impl=expression,
        )

    return Case(activation, humming, competitor, m * k * 4)


def _silu_case(m: int, k: int) -> Case:
    x = torch.randn((m, 2 * k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)

    def humming():
        process_input(
            x,
            outputs=output,
            activation_type="binary_split",
            activation_impl="a / (1.f + expf(-a)) * b",
        )

    return Case("silu", humming, lambda: torch.ops._C.silu_and_mul(output, x), m * k * 6)


def _silu_group_case(m: int, k: int, group_size: int, dtype: torch.dtype) -> Case:
    x = torch.randn((m, 2 * k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, k), device="cuda", dtype=dtype)
    scales = torch.empty((m, k // group_size), device="cuda", dtype=torch.float32)
    quant_dtype = "int8" if dtype == torch.int8 else "float8e4m3"

    def humming():
        process_input(
            x,
            outputs=output,
            quant_mode="dynamic_group",
            quant_dtype=quant_dtype,
            quant_group_size=group_size,
            group_scales=scales,
            activation_type="binary_split",
            activation_impl="a / (1.f + expf(-a)) * b",
        )

    def vllm():
        torch.ops._C.silu_and_mul_per_block_quant(output, x, scales, group_size, None, False)

    scale_bytes = scales.numel() * scales.element_size()
    return Case(f"silu_group_{quant_dtype}", humming, vllm, x.nbytes + output.nbytes + scale_bytes)


def _token_case(m: int, k: int, dtype: torch.dtype) -> Case:
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, k), device="cuda", dtype=dtype)
    scales = torch.empty((m, 1), device="cuda", dtype=torch.float32)
    quant_dtype = "int8" if dtype == torch.int8 else "float8e4m3"

    def humming():
        process_input(
            x,
            outputs=output,
            quant_mode="dynamic_token",
            quant_dtype=quant_dtype,
            token_scales=scales.view(m),
        )

    def competitor():
        if dtype == torch.int8:
            return torch.ops._C.dynamic_scaled_int8_quant(output, x, scales, None)
        return torch.ops._C.dynamic_per_token_scaled_fp8_quant(output, x, scales, None)

    return Case(f"token_{quant_dtype}", humming, competitor, x.nbytes + output.nbytes + scales.nbytes)


def _group_case(m: int, k: int, group_size: int, dtype: torch.dtype) -> Case:
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, k), device="cuda", dtype=dtype)
    scales = torch.empty((m, k // group_size), device="cuda", dtype=torch.float32)
    quant_dtype = "int8" if dtype == torch.int8 else "float8e4m3"

    def humming():
        process_input(
            x,
            outputs=output,
            quant_mode="dynamic_group",
            quant_dtype=quant_dtype,
            quant_group_size=group_size,
            group_scales=scales,
        )

    def vllm():
        if dtype == torch.int8:
            torch.ops._C.per_token_group_quant_int8(x, output, scales, group_size, 1e-10, -128.0, 127.0)
        else:
            torch.ops._C.per_token_group_fp8_quant(
                x,
                output,
                scales,
                group_size,
                1e-10,
                -448.0,
                448.0,
                False,
                False,
                False,
            )

    return Case(
        f"group_{quant_dtype}_g{group_size}",
        humming,
        vllm,
        x.nbytes + output.nbytes + scales.nbytes,
    )


def _static_tensor_case(m: int, k: int, group_size: int, dtype: torch.dtype) -> Case:
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, k), device="cuda", dtype=dtype)
    scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
    quant_dtype = "int8" if dtype == torch.int8 else "float8e4m3"

    def humming():
        process_input(
            x,
            outputs=output,
            quant_mode="static_tensor",
            quant_dtype=quant_dtype,
            quant_group_size=group_size,
            token_scales=scale,
        )

    def vllm():
        if dtype == torch.int8:
            torch.ops._C.static_scaled_int8_quant(output, x, scale, None)
        else:
            torch.ops._C.static_scaled_fp8_quant(output, x, scale, None)

    return Case(f"static_tensor_{quant_dtype}", humming, vllm, x.nbytes + output.nbytes)


def _fp4_case(m: int, k: int) -> Case:
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.empty((m, k // 16), device="cuda", dtype=torch.float8_e4m3fn)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

    def humming():
        process_input(
            x,
            outputs=output,
            quant_mode="static_tensor_dynamic_group",
            quant_dtype="float4e2m1",
            quant_group_size=16,
            group_scales=scales,
            token_scales=global_scale,
        )

    def vllm():
        torch.ops._C.scaled_fp4_quant.out(x, global_scale, False, output=output, output_scale=scales)

    return Case("group_fp4", humming, vllm, x.nbytes + output.nbytes + scales.nbytes)


def _silu_fp4_expert_case(m: int, k: int) -> Case:
    from vllm import envs

    x = torch.randn((m, 2 * k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
    stride = (m + 3) // 4 * 4
    scales = torch.empty(((k // 16 + 3) // 4, stride, 4), device="cuda", dtype=torch.float8_e4m3fn)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    expert_offsets = torch.tensor([0, m], device="cuda", dtype=torch.int32)
    blockscale_offsets = torch.tensor([0, m], device="cuda", dtype=torch.int32)
    vllm_scales = torch.empty(
        (envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE, (k // 16 + 3) // 4),
        device="cuda",
        dtype=torch.int32,
    )

    def humming():
        process_input(
            x,
            outputs=output,
            quant_mode="static_tensor_dynamic_group",
            quant_dtype="float4e2m1",
            quant_group_size=16,
            group_scales=scales,
            token_scales=global_scale,
            activation_type="binary_split",
            activation_impl="a / (1.f + expf(-a)) * b",
            layout="grouped",
            expert_layout=expert_offsets,
            group_scale_layout="mx_packed",
        )

    def vllm():
        torch.ops._C.silu_and_mul_scaled_fp4_experts_quant(
            output, vllm_scales, x, global_scale, expert_offsets, blockscale_offsets
        )

    useful_scales = m * k // 16
    return Case(
        "silu_group_fp4_expert",
        humming,
        vllm,
        x.nbytes + output.nbytes + useful_scales,
    )


def _mxfp4_expert_case(m: int, k: int, silu: bool) -> Case:
    from vllm import envs

    input_columns = 2 * k if silu else k
    x = torch.randn((m, input_columns), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
    stride = (m + 3) // 4 * 4
    scales = torch.empty(
        ((k // 32 + 3) // 4, stride, 4),
        device="cuda",
        dtype=torch.float8_e8m0fnu,
    )
    expert_offsets = torch.tensor([0, m], device="cuda", dtype=torch.int32)
    blockscale_offsets = torch.tensor([0, m], device="cuda", dtype=torch.int32)
    vllm_scales = torch.empty(
        (envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE, (k // 32 + 3) // 4),
        device="cuda",
        dtype=torch.int32,
    )

    def humming():
        process_input(
            x,
            outputs=output,
            quant_mode="dynamic_group",
            quant_dtype="float4e2m1",
            quant_group_size=32,
            group_scales=scales,
            activation_type="binary_split" if silu else "none",
            activation_impl="a / (1.f + expf(-a)) * b" if silu else None,
            layout="grouped",
            expert_layout=expert_offsets,
            group_scale_layout="mx_packed",
        )

    def vllm():
        op = torch.ops._C.silu_and_mul_mxfp4_experts_quant if silu else torch.ops._C.mxfp4_experts_quant
        op(output, vllm_scales, x, expert_offsets, blockscale_offsets, 1)

    useful_scales = m * k // 32
    prefix = "silu_" if silu else ""
    return Case(
        f"{prefix}group_mxfp4_expert",
        humming,
        vllm,
        x.nbytes + output.nbytes + useful_scales,
    )


def _hadamard_case(m: int, k: int, block_size: int, silu: bool) -> Case:
    input_columns = 2 * k if silu else k
    x = torch.randn((m, input_columns), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
    humming_buffer = None if silu else x.clone()
    vllm_buffer = torch.empty_like(output) if silu else x.clone()

    def humming():
        if not silu:
            return process_input(humming_buffer, inplace=True, hadamard_block_size=block_size)
        process_input(
            x,
            outputs=output,
            activation_type="binary_split",
            activation_impl="a / (1.f + expf(-a)) * b",
            hadamard_block_size=block_size,
        )

    def vllm():
        if silu:
            torch.ops._C.silu_and_mul(vllm_buffer, x)
        torch.ops._C.hadacore_transform(vllm_buffer.view(-1, block_size), True)

    name = f"{'silu_' if silu else ''}hadamard_h{block_size}"
    return Case(name, humming, vllm, x.nbytes + output.nbytes, 2 if silu else 1)


def _hadamard_fp4_case(m: int, k: int, block_size: int, silu: bool) -> Case:
    input_columns = 2 * k if silu else k
    x = torch.randn((m, input_columns), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.empty((m, k // 16), device="cuda", dtype=torch.float8_e4m3fn)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    vllm_buffer = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
    if not silu:
        vllm_buffer.copy_(x)

    def humming():
        process_input(
            x,
            outputs=output,
            quant_mode="static_tensor_dynamic_group",
            quant_dtype="float4e2m1",
            quant_group_size=16,
            group_scales=scales,
            token_scales=global_scale,
            activation_type="binary_split" if silu else "none",
            activation_impl="a / (1.f + expf(-a)) * b" if silu else None,
            hadamard_block_size=block_size,
        )

    def vllm():
        if silu:
            torch.ops._C.silu_and_mul(vllm_buffer, x)
        torch.ops._C.hadacore_transform(vllm_buffer.view(-1, block_size), True)
        torch.ops._C.scaled_fp4_quant.out(
            vllm_buffer, global_scale, False, output=output, output_scale=scales
        )

    prefix = "silu_" if silu else ""
    return Case(
        f"{prefix}hadamard_fp4_h{block_size}",
        humming,
        vllm,
        x.nbytes + output.nbytes + scales.nbytes,
        3 if silu else 2,
    )


def _make_humming_call(
    m: int,
    k: int,
    config: tuple[str, str, str | None, int, torch.dtype | None],
    activation: str | None,
    block_size: int | None,
) -> tuple[Callable[[], object], int]:
    _, quant_mode, quant_dtype, group_size, scale_dtype = config
    activation_type, activation_impl = ACTIVATIONS.get(activation, ("none", None))
    binary = activation_type.startswith("binary")
    x = torch.randn((m, k * (2 if binary else 1)), device="cuda", dtype=torch.bfloat16)
    if quant_dtype in ("int4", "float4e2m1"):
        output = torch.empty((m, k // 2), device="cuda", dtype=torch.uint8)
    elif quant_dtype == "int8":
        output = torch.empty((m, k), device="cuda", dtype=torch.int8)
    elif quant_dtype == "float8e4m3":
        output = torch.empty((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)

    uses_group = quant_mode in ("dynamic_group", "dynamic_group_token")
    uses_token = quant_mode in ("dynamic_token", "dynamic_group_token")
    group_scales = None
    if uses_group:
        group_scales = torch.empty((m, k // group_size), device="cuda", dtype=scale_dtype)
    token_scales = torch.empty((m,), device="cuda", dtype=torch.float32) if uses_token else None

    def call():
        return process_input(
            x,
            outputs=output,
            quant_mode=quant_mode,
            quant_dtype=quant_dtype,
            quant_group_size=group_size,
            group_scales=group_scales,
            token_scales=token_scales,
            activation_type=activation_type,
            activation_impl=activation_impl,
            hadamard_block_size=block_size,
        )

    semantic_bytes = x.nbytes + output.nbytes
    if group_scales is not None:
        semantic_bytes += group_scales.nbytes
    if token_scales is not None:
        semantic_bytes += token_scales.nbytes
    return call, semantic_bytes


def _feature_cases(
    m: int,
    k: int,
    block_sizes: tuple[int, ...],
    filters: list[str] | None = None,
) -> list[FeatureCase]:
    configs = (
        ("raw", "none", None, 128, None),
        ("group_int8_g128", "dynamic_group", "int8", 128, torch.float32),
        ("group_fp8_g128", "dynamic_group", "float8e4m3", 128, torch.float32),
        ("group_fp4_g16", "dynamic_group", "float4e2m1", 16, torch.float8_e4m3fn),
        ("token_int8", "dynamic_token", "int8", 128, None),
        ("token_fp8", "dynamic_token", "float8e4m3", 128, None),
        ("group_token_fp8_g128", "dynamic_group_token", "float8e4m3", 128, torch.float8_e4m3fn),
        ("group_token_fp4_g128", "dynamic_group_token", "float4e2m1", 128, torch.float8_e4m3fn),
    )
    cases = []
    for config in configs:
        config_name = config[0]
        features = [(activation, activation, None) for activation in ACTIVATIONS]
        features += [(f"h{block_size}", None, block_size) for block_size in block_sizes]
        features += [
            (f"{activation}_h{block_size}", activation, block_size)
            for activation in ACTIVATIONS
            for block_size in block_sizes
        ]
        selected = [
            feature
            for feature in features
            if filters is None or any(pattern in f"{config_name}+{feature[0]}" for pattern in filters)
        ]
        if not selected:
            continue
        baseline, baseline_bytes = _make_humming_call(m, k, config, None, None)
        for feature_name, activation, block_size in selected:
            variant, variant_bytes = _make_humming_call(m, k, config, activation, block_size)
            cases.append(
                FeatureCase(
                    f"{config_name}+{feature_name}",
                    baseline,
                    variant,
                    baseline_bytes,
                    variant_bytes,
                )
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, action="append")
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--hadamard-block-size", type=int, action="append")
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--suite", choices=("internal", "external"), default="internal")
    parser.add_argument("--cache-mode", choices=("cold", "hot"), default="cold")
    parser.add_argument("--top", type=int, help="only print the worst N internal cases")
    parser.add_argument("--filter", action="append", help="only run matching internal case names")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    props = torch.cuda.get_device_properties(args.device)
    flush = torch.empty(4 * props.L2_cache_size, dtype=torch.uint8, device="cuda")
    copy_us, hbm_gbps = _measure_hbm(args.repeats)
    print(f"device={props.name} L2={props.L2_cache_size} HBM={hbm_gbps:.1f}GB/s copy={copy_us:.1f}us")
    for m in args.m or (1, 16, 128, 1024, 4096):
        block_sizes = tuple(args.hadamard_block_size or (128, 512, args.k))
        block_sizes = tuple(
            size for size in block_sizes if size <= 512 and args.k % size == 0 and not size & (size - 1)
        )
        if args.suite == "internal":
            feature_cases = _feature_cases(m, args.k, block_sizes, args.filter)
            if args.cache_mode == "cold":
                timings = _profile_cold_features(feature_cases, flush, args.repeats)
            else:
                timings = [
                    _measure_functions(
                        case.baseline,
                        case.variant,
                        flush,
                        args.repeats,
                        args.cache_mode,
                    )
                    for case in feature_cases
                ]
            rows = []
            for case, (baseline_us, variant_us) in zip(feature_cases, timings, strict=True):
                latency_ratio = variant_us / baseline_us
                traffic_ratio = case.variant_bytes / case.baseline_bytes
                rows.append(
                    (
                        latency_ratio / traffic_ratio,
                        case,
                        baseline_us,
                        variant_us,
                        latency_ratio,
                        traffic_ratio,
                    )
                )
            if args.top is not None:
                rows = sorted(rows, reverse=True, key=lambda row: row[0])[: args.top]
            for excess, case, baseline_us, variant_us, latency_ratio, traffic_ratio in rows:
                print(
                    f"M={m} K={args.k} {case.name}: Q={baseline_us:.4f}us "
                    f"variant={variant_us:.4f}us latency={latency_ratio:.3f}x "
                    f"traffic={traffic_ratio:.3f}x excess={excess:.3f}x"
                )
            continue

        from vllm import _custom_ops  # noqa: F401

        cases = [
            _raw_activation_case(m, args.k, "relu"),
            _raw_activation_case(m, args.k, "gelu"),
            _silu_case(m, args.k),
            _static_tensor_case(m, args.k, args.group_size, torch.float8_e4m3fn),
            _static_tensor_case(m, args.k, args.group_size, torch.int8),
            _group_case(m, args.k, args.group_size, torch.float8_e4m3fn),
            _group_case(m, args.k, args.group_size, torch.int8),
            _silu_group_case(m, args.k, args.group_size, torch.float8_e4m3fn),
            _silu_group_case(m, args.k, args.group_size, torch.int8),
            _token_case(m, args.k, torch.float8_e4m3fn),
            _token_case(m, args.k, torch.int8),
            _fp4_case(m, args.k),
            _silu_fp4_expert_case(m, args.k),
        ]
        capability = props.major * 10 + props.minor
        if torch.ops._C.mxfp4_experts_quant_supported(capability):
            cases.extend((_mxfp4_expert_case(m, args.k, False), _mxfp4_expert_case(m, args.k, True)))
        for block_size in block_sizes:
            cases.extend(
                (
                    _hadamard_case(m, args.k, block_size, False),
                    _hadamard_case(m, args.k, block_size, True),
                    _hadamard_fp4_case(m, args.k, block_size, False),
                    _hadamard_fp4_case(m, args.k, block_size, True),
                )
            )
        if args.filter:
            cases = [case for case in cases if any(pattern in case.name for pattern in args.filter)]
        if args.cache_mode == "cold":
            timings = _profile_cold_features(
                [
                    FeatureCase(
                        case.name,
                        case.humming,
                        case.competitor,
                        case.semantic_bytes,
                        case.semantic_bytes,
                    )
                    for case in cases
                ],
                flush,
                args.repeats,
            )
        else:
            timings = [_measure_pair(case, flush, args.repeats, args.cache_mode) for case in cases]
        for case, (humming_us, competitor_us) in zip(cases, timings, strict=True):
            effective_gbps = case.semantic_bytes / (humming_us * 1e-6) / 1e9
            baseline = f" competitor_kernels={case.competitor_kernels}" if case.competitor_kernels > 1 else ""
            print(
                f"M={m} K={args.k} {case.name}: humming={humming_us:.4f}us "
                f"competitor={competitor_us:.4f}us speedup={competitor_us / humming_us:.3f}x "
                f"effective={effective_gbps:.1f}GB/s hbm_eff={effective_gbps / hbm_gbps:.1%}{baseline}"
            )


if __name__ == "__main__":
    main()
