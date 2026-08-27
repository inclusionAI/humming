"""Same-process paired benchmark for a baseline and candidate checkout.

Importing both implementations into one process keeps clocks, inputs, and
measurement order shared.  This is intended for resolving sub-microsecond
regressions which are otherwise easily hidden by cross-process GPU clocks.
"""

import argparse
import importlib
import inspect
import statistics
import sys
from pathlib import Path

import torch

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
}
SCALE_DTYPES = {
    "float32": torch.float32,
    "float8e4m3": torch.float8_e4m3fn,
    "float8e8m0": torch.uint8,
}
ACTIVATIONS = {
    "none": ("none", None),
    "relu": ("unary", "a > 0.f ? a : 0.f"),
    "gelu": ("unary", "0.5f * a * (1.f + erff(a * 0.7071067811865475f))"),
    "silu_split": ("binary_split", "a / (1.f + expf(-a)) * b"),
    "silu_interleaved": ("binary_interleaved", "a / (1.f + expf(-a)) * b"),
}


class _IsolatedFunction:
    """Keep delayed imports bound to the checkout that created the function."""

    def __init__(self, function, modules):
        self.function = function
        self.modules = modules

    def __call__(self, *args, **kwargs):
        for name in tuple(sys.modules):
            if name == "humming" or name.startswith("humming."):
                del sys.modules[name]
        sys.modules.update(self.modules)
        result = self.function(*args, **kwargs)
        # process_input imports its kernel lazily. Retain modules discovered
        # by the call so subsequent launches reuse the same runtime instance.
        self.modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "humming" or name.startswith("humming.")
        }
        return result


def _import_process_input(root: Path, *, skip_op_registration: bool = False):
    for name in tuple(sys.modules):
        if name == "humming" or name.startswith("humming."):
            del sys.modules[name]
    sys.path.insert(0, str(root))
    saved_library_methods = None
    if skip_op_registration:
        saved_library_methods = (
            torch.library.Library.define,
            torch.library.Library.impl,
            torch.library.Library._register_fake,
        )
        torch.library.Library.define = lambda *_args, **_kwargs: None
        torch.library.Library.impl = lambda *_args, **_kwargs: None
        torch.library.Library._register_fake = lambda *_args, **_kwargs: None
    try:
        input_ops = importlib.import_module("humming.ops.input")
        if hasattr(input_ops, "process_input"):
            process = input_ops.process_input
            if "quant_mode" not in inspect.signature(process).parameters:
                function = process
            else:

                def function(
                    inputs,
                    quant_dtype,
                    *,
                    outputs,
                    group_scales,
                    token_scales,
                    group_size,
                    dynamic_group_dtype,
                    dynamic_scale_mode,
                    hadamard_block_size,
                    m_major_scale,
                    static_tensor_scale=None,
                    activation_type="none",
                    activation_impl=None,
                ):
                    group_scale_layout = "m_major" if m_major_scale else "row_major"
                    if m_major_scale and group_scales is not None and group_scales.dtype == torch.int32:
                        rows = inputs.numel() // inputs.size(-1)
                        stride = (rows + 3) // 4 * 4
                        packing = 2 if quant_dtype in ("int4", "float4e2m1") else 1
                        groups = outputs.size(-1) * packing // group_size
                        group_scales = group_scales.view(SCALE_DTYPES[dynamic_group_dtype]).reshape(
                            (groups + 3) // 4, stride, 4
                        )
                        group_scale_layout = "mx_packed"
                    quant_mode = (
                        "none"
                        if quant_dtype is None
                        else {
                            "token": "dynamic_token",
                            "group": "dynamic_group",
                            "group_token": "dynamic_group_token",
                        }[dynamic_scale_mode]
                    )
                    if static_tensor_scale is not None:
                        assert quant_mode == "dynamic_group"
                        quant_mode = "static_tensor_dynamic_group"
                        token_scales = static_tensor_scale
                    return process(
                        inputs,
                        outputs=outputs,
                        quant_mode=quant_mode,
                        quant_dtype=quant_dtype,
                        quant_group_size=group_size,
                        group_scales=group_scales,
                        token_scales=token_scales,
                        activation_type=activation_type,
                        activation_impl=activation_impl,
                        hadamard_block_size=hadamard_block_size,
                        group_scale_layout=group_scale_layout,
                    )
        else:
            hadamard_ops = importlib.import_module("humming.ops.hadamard")

            def function(
                inputs,
                quant_dtype,
                *,
                outputs,
                group_scales,
                token_scales,
                group_size,
                dynamic_group_dtype,
                dynamic_scale_mode,
                hadamard_block_size,
                m_major_scale,
                static_tensor_scale=None,
                activation_type="none",
                activation_impl=None,
            ):
                assert activation_type == "none" and activation_impl is None
                assert dynamic_scale_mode == "group" and token_scales is None
                if hadamard_block_size:
                    return hadamard_ops.hadamard_quant_input(
                        inputs,
                        hadamard_block_size,
                        quant_dtype,
                        group_size,
                        outputs=outputs,
                        scales=group_scales,
                        m_major_scale=m_major_scale,
                        scale_dtype=dynamic_group_dtype,
                        global_scale=static_tensor_scale,
                    )
                return input_ops.quant_input(
                    inputs,
                    quant_dtype,
                    scales=group_scales,
                    outputs=outputs,
                    group_size=group_size,
                    m_major_scale=m_major_scale,
                    scale_dtype=dynamic_group_dtype,
                    global_scale=static_tensor_scale,
                )

        modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "humming" or name.startswith("humming.")
        }
        return _IsolatedFunction(function, modules)
    finally:
        if saved_library_methods is not None:
            (
                torch.library.Library.define,
                torch.library.Library.impl,
                torch.library.Library._register_fake,
            ) = saved_library_methods
        sys.path.remove(str(root))


def _make_call(
    function,
    inputs: torch.Tensor,
    *,
    block_size: int,
    group_size: int,
    quant_dtype: str,
    scale_dtype: str,
    scale_mode: str,
    m_major_scale: bool,
    activation: str,
):
    activation_type, activation_impl = ACTIVATIONS[activation]
    hidden_size = inputs.shape[-1] // (2 if activation_type.startswith("binary") else 1)
    packing = 2 if quant_dtype in ("int4", "float4e2m1") else 1
    output_dtype = inputs.dtype if quant_dtype is None else OUTPUT_DTYPES[quant_dtype]
    outputs = torch.empty(
        (*inputs.shape[:-1], hidden_size // packing),
        device=inputs.device,
        dtype=output_dtype,
    )
    num_groups = hidden_size // group_size
    rows = inputs.numel() // inputs.shape[-1]
    padded_rows = (rows + 3) // 4 * 4
    mx_pack = m_major_scale and scale_dtype == "float8e8m0"
    if m_major_scale:
        scale_shape = ((num_groups + 3) // 4, padded_rows) if mx_pack else (num_groups, padded_rows)
    else:
        scale_shape = (*inputs.shape[:-1], num_groups)
    uses_group_scale = quant_dtype is not None and scale_mode in ("group", "group_token", "global")
    uses_token_scale = quant_dtype is not None and scale_mode in ("token", "group_token")
    group_scales = (
        torch.empty(
            scale_shape,
            device=inputs.device,
            dtype=torch.int32 if mx_pack else SCALE_DTYPES[scale_dtype],
        )
        if uses_group_scale
        else None
    )
    token_scales = (
        torch.empty((rows,), device=inputs.device, dtype=torch.float32) if uses_token_scale else None
    )
    static_tensor_scale = (
        torch.tensor([0.025], device=inputs.device, dtype=torch.float32) if scale_mode == "global" else None
    )

    def call():
        function(
            inputs,
            quant_dtype,
            group_size=group_size,
            outputs=outputs,
            group_scales=group_scales,
            token_scales=token_scales,
            dynamic_group_dtype=scale_dtype if uses_group_scale else None,
            dynamic_scale_mode="group" if scale_mode == "global" else scale_mode,
            hadamard_block_size=block_size or None,
            m_major_scale=m_major_scale,
            static_tensor_scale=static_tensor_scale,
            activation_type=activation_type,
            activation_impl=activation_impl,
        )

    return call, outputs, group_scales, token_scales


def _capture(call, batch: int) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(batch):
            call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def _measure_pair(
    baseline: torch.cuda.CUDAGraph,
    candidate: torch.cuda.CUDAGraph,
    *,
    batch: int,
    repeats: int,
) -> tuple[float, float]:
    # Bring the GPU into the steady state using both workloads.  Warming each
    # graph in isolation biases sub-microsecond comparisons when their power
    # profiles trigger different boost-clock transitions.
    for _ in range(10):
        baseline.replay()
        candidate.replay()
    torch.cuda.synchronize()
    samples = {"baseline": [], "candidate": []}
    pairs = []
    for repeat in range(repeats):
        order = (
            (("baseline", baseline), ("candidate", candidate))
            if repeat % 2 == 0
            else (("candidate", candidate), ("baseline", baseline))
        )
        for name, graph in order:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            pairs.append((name, begin, end))
    torch.cuda.synchronize()
    for name, begin, end in pairs:
        samples[name].append(begin.elapsed_time(end) * 1000.0 / batch)
    return statistics.median(samples["baseline"]), statistics.median(samples["candidate"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--source-dtype", choices=SOURCE_DTYPES, default="bfloat16")
    parser.add_argument("--quant-dtype", choices=("none", *OUTPUT_DTYPES), default="float8e4m3")
    parser.add_argument("--scale-dtype", choices=SCALE_DTYPES, default="float8e4m3")
    parser.add_argument(
        "--scale-mode",
        choices=("none", "group", "token", "group_token", "global"),
        default="group",
    )
    parser.add_argument("--activation", choices=ACTIVATIONS, default="none")
    parser.add_argument("--m-major-scale", action="store_true")
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    quant_dtype = None if args.quant_dtype == "none" else args.quant_dtype
    assert (quant_dtype is None) == (args.scale_mode == "none")

    torch.cuda.set_device(args.device)
    torch.manual_seed(2026)
    activation_type, _ = ACTIVATIONS[args.activation]
    input_columns = args.k * (2 if activation_type.startswith("binary") else 1)
    inputs = torch.randn(
        (args.m, input_columns),
        device="cuda",
        dtype=SOURCE_DTYPES[args.source_dtype],
    )
    baseline_fn = _import_process_input(args.baseline_root.resolve())
    candidate_fn = _import_process_input(args.candidate_root.resolve(), skip_op_registration=True)
    baseline_call, baseline_output, baseline_group_scale, baseline_token_scale = _make_call(
        baseline_fn,
        inputs,
        block_size=args.block_size,
        group_size=args.group_size,
        quant_dtype=quant_dtype,
        scale_dtype=args.scale_dtype,
        scale_mode=args.scale_mode,
        m_major_scale=args.m_major_scale,
        activation=args.activation,
    )
    candidate_call, candidate_output, candidate_group_scale, candidate_token_scale = _make_call(
        candidate_fn,
        inputs,
        block_size=args.block_size,
        group_size=args.group_size,
        quant_dtype=quant_dtype,
        scale_dtype=args.scale_dtype,
        scale_mode=args.scale_mode,
        m_major_scale=args.m_major_scale,
        activation=args.activation,
    )
    baseline_call()
    candidate_call()
    torch.cuda.synchronize()
    print(
        f"M={args.m} K={args.k} G={args.group_size} "
        f"H={args.block_size} mode={args.scale_mode} "
        f"activation={args.activation} source={args.source_dtype} target={args.quant_dtype} "
        f"scale={args.scale_dtype}"
    )
    exact_group_scale = baseline_group_scale is None or torch.equal(
        baseline_group_scale, candidate_group_scale
    )
    exact_token_scale = baseline_token_scale is None or torch.equal(
        baseline_token_scale, candidate_token_scale
    )
    print(
        f"exact_output={torch.equal(baseline_output, candidate_output)} "
        f"exact_group_scale={exact_group_scale} "
        f"exact_token_scale={exact_token_scale}"
    )
    baseline_graph = _capture(baseline_call, args.batch)
    candidate_graph = _capture(candidate_call, args.batch)
    baseline_us, candidate_us = _measure_pair(
        baseline_graph,
        candidate_graph,
        batch=args.batch,
        repeats=args.repeats,
    )
    print(
        f"baseline={baseline_us:.6f} us candidate={candidate_us:.6f} us "
        f"ratio={candidate_us / baseline_us:.6f}x"
    )


if __name__ == "__main__":
    main()
