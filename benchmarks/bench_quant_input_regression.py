"""Regression benchmark for the legacy quant-input feature set.

Run this file from both a baseline checkout and a candidate checkout.  It
detects whether the checkout provides the unified CUDA API and uses
preallocated outputs/scales in both cases.  CUDA graphs remove Python launch
and allocation noise from the primary ``graph_us`` metric.
"""

import argparse
import dataclasses
import json
import statistics
import sys
from pathlib import Path

import torch

# Executing ``python benchmarks/...`` otherwise searches the benchmarks
# directory before the checkout root and may silently import another installed
# humming tree.  A regression comparison must benchmark the requested checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Case:
    family: str
    m: int
    k: int
    group_size: int
    block_size: int = 0
    source_dtype: str = "bfloat16"
    quant_dtype: str = "float8e4m3"
    scale_dtype: str = "float32"
    scale_mode: str = "dynamic"
    m_major_scale: bool = False

    @property
    def name(self) -> str:
        fields = [
            self.family,
            f"M{self.m}",
            f"K{self.k}",
            f"G{self.group_size}",
        ]
        if self.block_size:
            fields.append(f"H{self.block_size}")
        fields.extend((self.source_dtype, self.quant_dtype, self.scale_dtype, self.scale_mode))
        if self.m_major_scale:
            fields.append("mmajor")
        return "-".join(fields)


SOURCE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

OUTPUT_DTYPES = {
    "int8": torch.int8,
    "int4": torch.uint8,
    "float4e2m1": torch.uint8,
    "float8e4m3": torch.float8_e4m3fn,
    "float8e5m2": torch.float8_e5m2,
}

SCALE_DTYPES = {
    "float32": torch.float32,
    "float8e4m3": torch.float8_e4m3fn,
    "float8e8m0": torch.uint8,
}

LEGACY_FAMILIES = {
    "quant_input",
    "hadamard_quant_input",
    "hadamard_quant_input_wide",
}


def build_cases(preset: str) -> list[Case]:
    cases: list[Case] = []

    # M sweep: exposes launch-bound, saturation, and large memory-bound regions.
    for family, block_size, group_size in (
        ("quant_input", 0, 128),
        ("hadamard_quant_input", 128, 128),
        ("hadamard_quant_input_wide", 128, 512),
    ):
        for m in (1, 16, 128, 1024, 4096):
            cases.append(
                Case(
                    family=family,
                    m=m,
                    k=4096,
                    group_size=group_size,
                    block_size=block_size,
                )
            )

    # Saturated-grid E4 scale sweep: exercises the token-centric direct-scale
    # schedule while retaining the old per-group E4 result semantics.
    for family, block_size, group_size in (
        ("quant_input", 0, 128),
        ("hadamard_quant_input", 128, 128),
        ("hadamard_quant_input_wide", 128, 512),
    ):
        for m in (512, 1024, 2048, 4096):
            cases.append(
                Case(
                    family=family,
                    m=m,
                    k=4096,
                    group_size=group_size,
                    block_size=block_size,
                    scale_dtype="float8e4m3",
                )
            )

    # K sweep at a representative batch size.
    for family, block_size, group_size in (
        ("quant_input", 0, 128),
        ("hadamard_quant_input", 128, 128),
        ("hadamard_quant_input_wide", 128, 512),
    ):
        for k in (4096, 7168, 8192):
            cases.append(
                Case(
                    family=family,
                    m=128,
                    k=k,
                    group_size=group_size,
                    block_size=block_size,
                )
            )

    # Group/transform scheduling regimes.
    for group_size in (64, 128, 256, 512, 4096):
        cases.append(Case(family="quant_input", m=128, k=4096, group_size=group_size))
    for block_size, group_size in (
        (64, 64),
        (128, 64),
        (128, 128),
        (256, 128),
        (256, 256),
        (512, 128),
        (1024, 128),
    ):
        cases.append(
            Case(
                family="hadamard_quant_input",
                m=128,
                k=4096,
                group_size=group_size,
                block_size=block_size,
            )
        )
    for block_size, group_size in (
        (64, 256),
        (128, 256),
        (128, 512),
        (128, 1024),
        (256, 512),
    ):
        cases.append(
            Case(
                family="hadamard_quant_input_wide",
                m=128,
                k=4096,
                group_size=group_size,
                block_size=block_size,
            )
        )

    # Source/target/scale codecs supported by the old paths.
    for family, block_size, group_size in (
        ("quant_input", 0, 128),
        ("hadamard_quant_input", 128, 128),
        ("hadamard_quant_input_wide", 128, 512),
    ):
        for source_dtype in ("float16", "bfloat16", "float32"):
            for quant_dtype in ("int8", "float8e4m3", "int4"):
                cases.append(
                    Case(
                        family=family,
                        m=128,
                        k=4096,
                        group_size=group_size,
                        block_size=block_size,
                        source_dtype=source_dtype,
                        quant_dtype=quant_dtype,
                    )
                )
        for scale_dtype in ("float8e4m3", "float8e8m0"):
            cases.append(
                Case(
                    family=family,
                    m=128,
                    k=4096,
                    group_size=group_size,
                    block_size=block_size,
                    scale_dtype=scale_dtype,
                )
            )
        for scale_dtype in ("float32", "float8e4m3", "float8e8m0"):
            cases.append(
                Case(
                    family=family,
                    m=128,
                    k=4096,
                    group_size=group_size,
                    block_size=block_size,
                    scale_dtype=scale_dtype,
                    m_major_scale=True,
                )
            )

    # Static quant and global-scale normalization are separate old code paths.
    cases.extend(
        (
            Case(
                family="quant_input",
                m=128,
                k=4096,
                group_size=128,
                scale_mode="static",
            ),
            Case(
                family="quant_input",
                m=4096,
                k=4096,
                group_size=128,
                scale_mode="static",
            ),
            Case(
                family="hadamard_quant_input",
                m=128,
                k=4096,
                group_size=128,
                block_size=128,
                scale_mode="global",
            ),
            Case(
                family="hadamard_quant_input_wide",
                m=128,
                k=4096,
                group_size=512,
                block_size=128,
                scale_mode="global",
            ),
        )
    )

    # Diagnostic for the expected benefit of fusing Hadamard with quantization.
    for m in (128, 1024, 4096):
        cases.append(
            Case(
                family="hadamard_unfused",
                m=m,
                k=4096,
                group_size=128,
                block_size=128,
            )
        )

    # Stable de-duplication keeps the matrix readable.
    cases = list(dict.fromkeys(cases))
    if preset == "core":
        return [
            case
            for case in cases
            if case.source_dtype == "bfloat16"
            and case.quant_dtype == "float8e4m3"
            and case.scale_dtype == "float32"
        ]
    return cases


def build_group_token_cases(preset: str) -> list[Case]:
    cases = []
    for family, block_size, group_size in (
        ("quant_input_group_token", 0, 128),
        ("hadamard_quant_input_group_token", 128, 128),
        ("hadamard_quant_input_wide_group_token", 128, 512),
    ):
        for m in (1, 16, 128, 1024, 4096):
            cases.append(
                Case(
                    family=family,
                    m=m,
                    k=4096,
                    group_size=group_size,
                    block_size=block_size,
                    scale_dtype="float8e4m3",
                    scale_mode="group_token",
                )
            )
    if preset == "core":
        return cases

    for group_size in (64, 128, 256, 512, 1024, 4096):
        cases.append(
            Case(
                family="quant_input_group_token",
                m=128,
                k=4096,
                group_size=group_size,
                scale_dtype="float8e4m3",
                scale_mode="group_token",
            )
        )
    for block_size, group_size in (
        (64, 64),
        (128, 64),
        (128, 128),
        (256, 128),
        (256, 256),
        (512, 128),
        (1024, 128),
        (64, 256),
        (128, 256),
        (128, 512),
        (128, 1024),
        (256, 512),
    ):
        family = (
            "hadamard_quant_input_group_token"
            if group_size <= block_size
            else "hadamard_quant_input_wide_group_token"
        )
        cases.append(
            Case(
                family=family,
                m=128,
                k=4096,
                group_size=group_size,
                block_size=block_size,
                scale_dtype="float8e4m3",
                scale_mode="group_token",
            )
        )
    return list(dict.fromkeys(cases))


def build_dynamic_token_cases(preset: str) -> list[Case]:
    cases = []
    for family, block_size, group_size in (
        ("quant_input_dynamic_token", 0, 128),
        ("hadamard_quant_input_dynamic_token", 128, 128),
        ("hadamard_quant_input_wide_dynamic_token", 128, 512),
    ):
        for m in (1, 16, 128, 1024, 4096):
            cases.append(
                Case(
                    family=family,
                    m=m,
                    k=4096,
                    group_size=group_size,
                    block_size=block_size,
                    scale_mode="dynamic_token",
                )
            )
        if preset == "full":
            for k in (7168, 8192):
                cases.append(
                    Case(
                        family=family,
                        m=128,
                        k=k,
                        group_size=group_size,
                        block_size=block_size,
                        scale_mode="dynamic_token",
                    )
                )
            cases.append(
                Case(
                    family=family,
                    m=128,
                    k=4096,
                    group_size=group_size,
                    block_size=block_size,
                    quant_dtype="int8",
                    scale_mode="dynamic_token",
                )
            )
    return cases


def allocate_output(case: Case, inputs: torch.Tensor) -> torch.Tensor:
    packing = 2 if case.quant_dtype in ("int4", "float4e2m1") else 1
    return torch.empty(
        (case.m, case.k // packing),
        device=inputs.device,
        dtype=OUTPUT_DTYPES[case.quant_dtype],
    )


def allocate_group_scales(case: Case, inputs: torch.Tensor) -> torch.Tensor:
    num_groups = case.k // case.group_size
    m_stride = (case.m + 3) // 4 * 4 if case.m_major_scale else case.m
    mx_pack = (
        case.m_major_scale
        and case.scale_dtype in ("float8e4m3", "float8e8m0")
        and (case.family == "quant_input" or case.scale_dtype == "float8e8m0")
    )
    if mx_pack:
        return torch.empty(
            ((num_groups + 3) // 4, m_stride),
            device=inputs.device,
            dtype=torch.int32,
        )
    shape = (num_groups, m_stride) if case.m_major_scale else (case.m, num_groups)
    return torch.empty(shape, device=inputs.device, dtype=SCALE_DTYPES[case.scale_dtype])


def process_group_scale_view(case: Case, scales: torch.Tensor) -> torch.Tensor:
    if scales.dtype != torch.int32:
        return scales
    groups = case.k // case.group_size
    stride = (case.m + 3) // 4 * 4
    return scales.view(SCALE_DTYPES[case.scale_dtype]).reshape((groups + 3) // 4, stride, 4)


def make_baseline_quant_call(case: Case, inputs: torch.Tensor, outputs: torch.Tensor):
    """Launch the old Triton kernel directly so dynamic scales are preallocated."""
    import triton

    import humming.ops.input as input_ops

    group_scales = allocate_group_scales(case, inputs)
    static_scales = None
    is_dynamic = case.scale_mode != "static"
    if not is_dynamic:
        static_scales = torch.full(
            (case.k // case.group_size,), 0.025, device=inputs.device, dtype=torch.float32
        )
        scale_arg = static_scales
    else:
        scale_arg = group_scales

    num_blocks = inputs.numel() // case.group_size
    block = triton.next_power_of_2(case.group_size)
    groups_per_block = 1
    if case.group_size <= 256 and num_blocks >= 131072:
        groups_per_block = min(1024 // case.group_size, num_blocks)
    grid_blocks = (num_blocks + groups_per_block - 1) // groups_per_block
    packed = case.quant_dtype in ("int4", "float4e2m1")
    effective_block = block // 2 if packed else block
    num_warps = min(max(effective_block // 256, 1), 8)
    global_scale = None
    m_stride = (case.m + 3) // 4 * 4 if case.m_major_scale else case.m
    mx_pack = case.m_major_scale and case.scale_dtype in ("float8e4m3", "float8e8m0")
    launch_args = (
        inputs,
        outputs,
        scale_arg,
        inputs.stride(0),
        num_blocks,
        is_dynamic,
        case.k,
        case.group_size,
        block,
        groups_per_block,
        case.quant_dtype,
        m_stride,
        case.m_major_scale,
        case.scale_dtype,
        False,
        global_scale,
        mx_pack,
    )
    # USE_PDL has a constexpr default of False.  Triton 3.7 no longer accepts
    # explicitly passing defaulted constexprs through the launch keyword pack,
    # so rely on the kernel default while still disabling launch-side PDL.
    launch_kwargs = dict(num_warps=num_warps, num_stages=1, launch_pdl=False)

    def call():
        input_ops._quant_tensor_kernel[(grid_blocks,)](*launch_args, **launch_kwargs)

    return call


def make_current_quant_call(case: Case, inputs: torch.Tensor, outputs: torch.Tensor):
    from humming.ops.input import process_input

    if case.scale_mode == "static":
        static_scales = torch.full(
            (case.k // case.group_size,), 0.025, device=inputs.device, dtype=torch.float32
        )

        def call():
            process_input(
                inputs,
                outputs=outputs,
                quant_mode="static_group",
                quant_dtype=case.quant_dtype,
                quant_group_size=case.group_size,
                group_scales=static_scales,
            )

    else:
        group_scales = allocate_group_scales(case, inputs)
        process_scales = process_group_scale_view(case, group_scales)
        group_scale_layout = (
            "row_major"
            if not case.m_major_scale
            else ("mx_packed" if group_scales.dtype == torch.int32 else "m_major")
        )

        def call():
            process_input(
                inputs,
                outputs=outputs,
                quant_mode="dynamic_group",
                quant_dtype=case.quant_dtype,
                group_scales=process_scales,
                quant_group_size=case.group_size,
                group_scale_layout=group_scale_layout,
            )

    return call


def make_hadamard_call(case: Case, inputs: torch.Tensor, outputs: torch.Tensor):
    from humming.ops.input import hadamard_quant_input

    scales = allocate_group_scales(case, inputs)
    global_scale = None
    if case.scale_mode == "global":
        global_scale = torch.tensor([0.025], device=inputs.device, dtype=torch.float32)

    def call():
        hadamard_quant_input(
            inputs,
            block_size=case.block_size,
            quant_dtype=case.quant_dtype,
            group_size=case.group_size,
            outputs=outputs,
            scales=scales,
            scale_dtype=case.scale_dtype,
            global_scale=global_scale,
            m_major_scale=case.m_major_scale,
        )

    return call


def make_group_token_call(case: Case, inputs: torch.Tensor, outputs: torch.Tensor):
    from humming.ops.input import process_input

    group_scales = allocate_group_scales(case, inputs)
    process_scales = process_group_scale_view(case, group_scales)
    group_scale_layout = (
        "row_major" if not case.m_major_scale else ("mx_packed" if process_scales.ndim == 3 else "m_major")
    )
    token_scales = torch.empty((case.m,), device=inputs.device, dtype=torch.float32)
    block_size = case.block_size or None

    def call():
        process_input(
            inputs,
            outputs=outputs,
            quant_mode="dynamic_group_token",
            quant_dtype=case.quant_dtype,
            group_scales=process_scales,
            token_scales=token_scales,
            quant_group_size=case.group_size,
            hadamard_block_size=block_size,
            group_scale_layout=group_scale_layout,
        )

    return call


def make_dynamic_token_call(case: Case, inputs: torch.Tensor, outputs: torch.Tensor):
    from humming.ops.input import process_input

    token_scales = torch.empty((case.m,), device=inputs.device, dtype=torch.float32)
    block_size = case.block_size or None

    def call():
        process_input(
            inputs,
            outputs=outputs,
            quant_mode="dynamic_token",
            quant_dtype=case.quant_dtype,
            token_scales=token_scales,
            quant_group_size=case.group_size,
            hadamard_block_size=block_size,
        )

    return call


def make_unfused_call(case: Case, inputs: torch.Tensor, outputs: torch.Tensor):
    import humming.ops.input as input_ops
    from humming.ops.input import hadamard_transform

    transformed = torch.empty_like(inputs)
    if hasattr(input_ops, "process_input"):
        quant_call = make_current_quant_call(case, transformed, outputs)
    else:
        quant_call = make_baseline_quant_call(case, transformed, outputs)

    def call():
        hadamard_transform(inputs, case.block_size, outputs=transformed)
        quant_call()

    return call


def make_call(case: Case):
    inputs = torch.randn((case.m, case.k), device="cuda", dtype=SOURCE_DTYPES[case.source_dtype])
    outputs = allocate_output(case, inputs)
    if case.family.endswith("_group_token"):
        call = make_group_token_call(case, inputs, outputs)
    elif case.family.endswith("_dynamic_token"):
        call = make_dynamic_token_call(case, inputs, outputs)
    elif case.family == "quant_input":
        import humming.ops.input as input_ops

        if hasattr(input_ops, "process_input"):
            call = make_current_quant_call(case, inputs, outputs)
        else:
            call = make_baseline_quant_call(case, inputs, outputs)
    elif case.family == "hadamard_unfused":
        call = make_unfused_call(case, inputs, outputs)
    else:
        call = make_hadamard_call(case, inputs, outputs)
    return call, inputs, outputs


def benchmark_calls(calls, *, graph_batch: int, repeats: int) -> tuple[float, float]:
    for call in calls:
        call()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if len(calls) == 1:
            for _ in range(graph_batch):
                calls[0]()
        else:
            assert graph_batch == len(calls)
            for call in calls:
                call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    event_pairs = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        event_pairs.append((begin, end))
    torch.cuda.synchronize()
    samples = [begin.elapsed_time(end) * 1000.0 / graph_batch for begin, end in event_pairs]
    return statistics.median(samples), min(samples)


def compare_with_baseline(records: list[dict], baseline_path: Path, max_regression: float) -> None:
    baseline_records = [json.loads(line) for line in baseline_path.read_text().splitlines()]
    baseline_by_name = {
        record["name"]: record for record in baseline_records if record.get("family") in LEGACY_FAMILIES
    }
    current_records = [record for record in records if record.get("family") in LEGACY_FAMILIES]
    missing = [record["name"] for record in current_records if record["name"] not in baseline_by_name]
    if missing:
        raise ValueError(f"baseline is missing {len(missing)} cases: {missing}")

    limit = 1.0 + max_regression
    violations = []
    family_ratios: dict[str, list[float]] = {}
    for record in current_records:
        baseline = baseline_by_name[record["name"]]
        if baseline["cache_mode"] != record["cache_mode"]:
            raise ValueError(
                f"cache mode mismatch for {record['name']}: "
                f"{baseline['cache_mode']} != {record['cache_mode']}"
            )
        ratio = record["graph_us"] / baseline["graph_us"]
        family_ratios.setdefault(record["family"], []).append(ratio)
        if ratio > limit:
            violations.append((record["name"], baseline["graph_us"], record["graph_us"], ratio))

    for family, ratios in family_ratios.items():
        print(
            f"{family}: n={len(ratios)}, median={statistics.median(ratios):.4f}x, "
            f"max={max(ratios):.4f}x, violations={sum(ratio > limit for ratio in ratios)}"
        )
    for name, baseline_us, current_us, ratio in violations:
        print(f"REGRESSION {name}: {baseline_us:.4f} -> {current_us:.4f} us ({ratio:.4f}x)")
    if violations:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("core", "full"), default="full")
    parser.add_argument("--cache-mode", choices=("hot", "streaming"), default="hot")
    parser.add_argument("--family", action="append")
    parser.add_argument("--min-m", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.01)
    parser.add_argument("--include-group-token", action="store_true")
    parser.add_argument("--include-dynamic-token", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(2026)
    cases = [case for case in build_cases(args.preset) if case.m >= args.min_m]
    if args.include_group_token:
        cases.extend(case for case in build_group_token_cases(args.preset) if case.m >= args.min_m)
    if args.include_dynamic_token:
        cases.extend(case for case in build_dynamic_token_cases(args.preset) if case.m >= args.min_m)
    if args.family:
        cases = [case for case in cases if case.family in args.family]
    records = [
        {
            "metadata": {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "preset": args.preset,
                "cache_mode": args.cache_mode,
                "num_cases": len(cases),
            }
        }
    ]
    if not args.quiet:
        print(json.dumps(records[0]), flush=True)
    for case in cases:
        numel = case.m * case.k
        source_bytes = torch.tensor([], dtype=SOURCE_DTYPES[case.source_dtype]).element_size()
        output_bytes = 0.5 if case.quant_dtype in ("int4", "float4e2m1") else 1.0
        bytes_per_slot = int(numel * (source_bytes + output_bytes))
        if args.cache_mode == "streaming":
            l2_bytes = torch.cuda.get_device_properties().L2_cache_size
            num_slots = max(2, (3 * l2_bytes + bytes_per_slot - 1) // bytes_per_slot)
            num_slots = min(num_slots, 64)
            calls_and_tensors = [make_call(case) for _ in range(num_slots)]
            calls = [item[0] for item in calls_and_tensors]
            graph_batch = len(calls)
        else:
            calls_and_tensors = [make_call(case)]
            calls = [calls_and_tensors[0][0]]
            graph_batch = 20 if numel >= 8 * 1024 * 1024 else 100
        median_us, min_us = benchmark_calls(calls, graph_batch=graph_batch, repeats=args.repeats)
        logical_gb = numel * (source_bytes + output_bytes) / 1e9
        result = dataclasses.asdict(case)
        result.update(
            name=case.name,
            graph_us=median_us,
            min_us=min_us,
            logical_gbps=logical_gb / (median_us * 1e-6),
            cache_mode=args.cache_mode,
            working_set_slots=len(calls),
        )
        records.append(result)
        if not args.quiet:
            print(json.dumps(result), flush=True)
        del calls, calls_and_tensors
    if args.output is not None:
        args.output.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    if args.baseline is not None:
        compare_with_baseline(records, args.baseline, args.max_regression)


if __name__ == "__main__":
    main()
