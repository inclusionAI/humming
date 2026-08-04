import argparse

import torch
import triton
import vllm._custom_ops as vllm_ops
from tqdm import tqdm

from humming.testing import save_benchmark_result


def bench_cutlass_w8a8(
    shape_n: int,
    shape_k: int,
    dtype: str,
    out_dtype: str,
    shape_m_list: list[int] | None = None,
) -> list[dict[str, int | float]]:
    assert dtype in ["int8", "float8e4m3"]

    torch_dtype = torch.int8 if dtype == "int8" else torch.float8_e4m3fn
    weight = torch.randint(-120, 120, (shape_n, shape_k), dtype=torch.int8, device="cuda:0")
    weight = weight.view(torch_dtype)
    weight_scale = torch.randn((shape_n,), dtype=torch.float32, device="cuda:0")

    default_shape_m_list = [2**i for i in range(15)]
    benchmark_result: list[dict[str, int | float]] = []
    for shape_m in tqdm(shape_m_list or default_shape_m_list):
        inputs = torch.randint(-120, 120, (shape_m, shape_k), dtype=torch.int8, device="cuda:0")
        inputs = inputs.view(torch_dtype)
        input_scale = torch.randn((shape_m,), dtype=torch.float32, device="cuda:0")

        def run():
            return vllm_ops.cutlass_scaled_mm(
                a=inputs,  # noqa
                b=weight.T,
                scale_a=input_scale,  # noqa
                scale_b=weight_scale,
                out_dtype=getattr(torch, out_dtype),
            )

        outputs = run()
        torch.cuda.synchronize()
        t = triton.testing.do_bench(run, warmup=100, rep=1000)

        nbytes = inputs.nbytes + outputs.nbytes + input_scale.nbytes
        nbytes += weight.nbytes + weight_scale.nbytes

        res = {
            "shape_m": shape_m,
            "time": t,
            "memory_gbps": nbytes / t / 1e6,
            "compute_tops": shape_m * shape_n * shape_k * 2 / t / 1e9,
        }
        benchmark_result.append(res)

    return benchmark_result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--shape_n", type=int, required=True)
    parser.add_argument("--shape_k", type=int, required=True)
    parser.add_argument("--dtype", type=str, choices=["int8", "float8e4m3"], required=True)
    parser.add_argument("--out_dtype", type=str, choices=["float16", "bfloat16"], required=True)
    parser.add_argument("--shape_m_list", type=int, default=None, nargs="+")
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()
    benchmark_result = bench_cutlass_w8a8(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
        dtype=args.dtype,
        out_dtype=args.out_dtype,
        shape_m_list=args.shape_m_list,
    )

    save_benchmark_result(benchmark_result, args, ["vllm"])

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
