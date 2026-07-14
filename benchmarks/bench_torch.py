import argparse

import torch
import triton
from tqdm import tqdm

from humming.utils.test import save_benchmark_result

# Block-scaled scaled_mm (cuBLAS) helpers. Available on recent PyTorch builds.
try:
    from torch.nn.functional import ScalingType, SwizzleType, scaled_mm
    from torch.testing._internal.common_quantized import (
        _bfloat16_to_float4_e2m1fn_x2,
        to_blocked,
    )

    _HAS_SCALED_MM_V2 = True
    _SW = SwizzleType.SWIZZLE_32_4_4
    _NO = SwizzleType.NO_SWIZZLE
except Exception:  # pragma: no cover - depends on torch version
    _HAS_SCALED_MM_V2 = False


DTYPE_CHOICES = [
    "float16",
    "bfloat16",
    "fp8",  # e4m3 x e4m3, per-tensor scale
    "int8",  # int8 x int8 -> int32
    "mxfp8",  # e4m3 x e4m3, e8m0 block32
    "nvfp4",  # e2m1 x e2m1, e4m3 block16 + per-tensor global
    "mxfp4",  # e2m1 x e2m1, e8m0 block32
    "mxfp8_mxfp4",  # mixed: e4m3 x e2m1 (probe)
    "mxfp8_mxfp6",  # mixed: e4m3 x e3m2 (probe)
]


def _e8m0_scale(rows: int, cols: int) -> torch.Tensor:
    raw = torch.randint(120, 130, (rows, cols), device="cuda:0", dtype=torch.uint8)
    return to_blocked(raw.view(torch.float8_e8m0fnu))


def _e4m3_scale(rows: int, cols: int) -> torch.Tensor:
    return to_blocked((torch.rand(rows, cols, device="cuda:0") * 0.5 + 0.5).to(torch.float8_e4m3fn))


def make_runner(dtype: str, shape_m: int, shape_n: int, shape_k: int):
    m, n, k = shape_m, shape_n, shape_k
    dev = "cuda:0"

    if dtype in ("float16", "bfloat16"):
        td = torch.float16 if dtype == "float16" else torch.bfloat16
        a = torch.randn(m, k, dtype=td, device=dev)
        b = torch.randn(n, k, dtype=td, device=dev).t()
        out = a.matmul(b)
        return (lambda: a.matmul(b)), a.nbytes + b.nbytes + out.nbytes

    if dtype == "int8":
        a = torch.randint(-8, 8, (m, k), device=dev, dtype=torch.int8)
        b = torch.randint(-8, 8, (n, k), device=dev, dtype=torch.int8).t()
        out = torch._int_mm(a, b)
        return (lambda: torch._int_mm(a, b)), a.nbytes + b.nbytes + out.nbytes

    if dtype == "fp8":
        a = torch.randn(m, k, dtype=torch.bfloat16, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, dtype=torch.bfloat16, device=dev).to(torch.float8_e4m3fn).t()
        sa = torch.tensor(1.0, device=dev)
        sb = torch.tensor(1.0, device=dev)

        def run():
            return torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16)

        out = run()
        return run, a.nbytes + b.nbytes + out.nbytes

    if not _HAS_SCALED_MM_V2:
        raise RuntimeError("scaled_mm (v2) not available in this torch build")

    def fp4(x_bf16: torch.Tensor) -> torch.Tensor:
        return _bfloat16_to_float4_e2m1fn_x2(x_bf16)

    if dtype == "mxfp8":
        a = torch.randn(m, k, dtype=torch.bfloat16, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, dtype=torch.bfloat16, device=dev).to(torch.float8_e4m3fn).t()
        sa, sb = _e8m0_scale(m, k // 32), _e8m0_scale(n, k // 32)
        recipe = [ScalingType.BlockWise1x32]

        def run():
            return scaled_mm(
                a,
                b,
                [sa],
                recipe,
                [sb],
                recipe,
                swizzle_a=[_SW],
                swizzle_b=[_SW],
                output_dtype=torch.bfloat16,
            )

    elif dtype == "mxfp4":
        a = fp4(torch.randn(m, k, dtype=torch.bfloat16, device=dev))
        b = fp4(torch.randn(n, k, dtype=torch.bfloat16, device=dev)).t()
        sa, sb = _e8m0_scale(m, k // 32), _e8m0_scale(n, k // 32)
        recipe = [ScalingType.BlockWise1x32]

        def run():
            return scaled_mm(
                a,
                b,
                [sa],
                recipe,
                [sb],
                recipe,
                swizzle_a=[_SW],
                swizzle_b=[_SW],
                output_dtype=torch.bfloat16,
            )

    elif dtype == "nvfp4":
        a = fp4(torch.randn(m, k, dtype=torch.bfloat16, device=dev))
        b = fp4(torch.randn(n, k, dtype=torch.bfloat16, device=dev)).t()
        sa, sb = _e4m3_scale(m, k // 16), _e4m3_scale(n, k // 16)
        ga = torch.tensor([1.0], device=dev)
        gb = torch.tensor([1.0], device=dev)
        recipe = [ScalingType.BlockWise1x16, ScalingType.TensorWise]

        def run():
            return scaled_mm(
                a,
                b,
                [sa, ga],
                recipe,
                [sb, gb],
                recipe,
                swizzle_a=[_SW, _NO],
                swizzle_b=[_SW, _NO],
                output_dtype=torch.bfloat16,
            )

    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    out = run()
    return run, a.nbytes + b.nbytes + out.nbytes


def bench_torch(
    shape_n: int,
    shape_k: int,
    dtype: str,
    use_f16_accum: bool = False,
    shape_m_list: list[int] | None = None,
) -> list[dict[str, int | float]]:
    if use_f16_accum:
        assert dtype == "float16", "fp16 accumulation only applies to float16"
        torch.backends.cuda.matmul.allow_fp16_accumulation = True

    default_shape_m_list = [2**i for i in range(15)]
    benchmark_result: list[dict[str, int | float]] = []
    for shape_m in tqdm(shape_m_list or default_shape_m_list):
        try:
            run, nbytes = make_runner(dtype, shape_m, shape_n, shape_k)
            run()
            torch.cuda.synchronize()
            t = triton.testing.do_bench(run, warmup=100, rep=1000)
            res: dict[str, int | float] = {
                "shape_m": shape_m,
                "time": t,
                "memory_gbps": nbytes / t / 1e6,
                "compute_tops": shape_m * shape_n * shape_k * 2 / t / 1e9,
            }
        except Exception as e:  # unsupported dtype/shape on this device
            res = {
                "shape_m": shape_m,
                "time": float("nan"),
                "memory_gbps": float("nan"),
                "compute_tops": float("nan"),
                "error": str(e).splitlines()[-1][:80],
            }
        benchmark_result.append(res)

    return benchmark_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape_n", type=int, required=True)
    parser.add_argument("--shape_k", type=int, required=True)
    parser.add_argument("--dtype", type=str, choices=DTYPE_CHOICES, required=True)
    parser.add_argument("--use_f16_accum", default=False, action="store_true")
    parser.add_argument("--shape_m_list", type=int, default=None, nargs="+")
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()
    benchmark_result = bench_torch(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
        dtype=args.dtype,
        use_f16_accum=args.use_f16_accum,
        shape_m_list=args.shape_m_list,
    )

    save_benchmark_result(benchmark_result, args)

    from tabulate import tabulate

    table = tabulate(
        benchmark_result,
        headers="keys",
        tablefmt="grid",
        numalign="right",
        floatfmt=".4f",
    )

    print(table)


if __name__ == "__main__":
    main()
