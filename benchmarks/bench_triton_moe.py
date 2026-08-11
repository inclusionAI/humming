import argparse

import torch
import triton
import triton.language as tl
from tqdm import tqdm
from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_default_config,
    get_moe_configs,
    invoke_fused_moe_triton_kernel,
)

from humming.testing import generate_random_moe_tensors, save_benchmark_result


def generate_random_tensor(dtype, shape: tuple[int, ...]):
    if dtype in [torch.int8, torch.float8_e4m3fn]:
        tensor = torch.randint(-120, 120, shape, dtype=torch.int8, device="cuda:0")
        return tensor.view(dtype)
    else:
        return torch.randn(shape, dtype=dtype, device="cuda:0")


def get_triton_moe_config(
    num_experts: int,
    shape_n: int,
    shape_k: int,
    shape_m: int,
    top_k: int,
    is_moe_down: bool,
    block_shape: list[int] | None,
    torch_dtype: torch.dtype,
):
    if torch_dtype == torch.float8_e4m3fn:
        dtype = "fp8_w8a8"
    elif torch_dtype == torch.int8:
        dtype = "int8_w8a8"
    else:
        dtype = None

    if not is_moe_down:
        shape_n = shape_n // 2
    else:
        shape_n, shape_k = shape_k, shape_n

    if block_shape is None:
        configs = get_moe_configs(num_experts, shape_n, dtype)
    else:
        configs = get_moe_configs(num_experts, shape_n, dtype, block_shape[0], block_shape[1])

    if configs is None and block_shape is None:
        configs = get_moe_configs(num_experts, shape_n, dtype, 128, 128)
    if configs is not None:
        config = configs[min(configs.keys(), key=lambda x: abs(x - shape_m))]
        return config
    else:
        return get_default_config(shape_m, num_experts, shape_n, shape_k, top_k, dtype)


def bench_triton_moe(
    shape_n: int,
    shape_k: int,
    num_experts: int,
    top_k: int,
    is_moe_down: bool,
    dtype: str,
    out_dtype: str,
    block_shape: str | list[int] | None = None,
    balanced: bool = False,
    shape_m_list: list[int] | None = None,
) -> list[dict[str, int | float]]:
    if isinstance(block_shape, str):
        block_shape = [int(x) for x in block_shape.split("x")]

    dtype_map = {
        "int8": torch.int8,
        "float8e4m3": torch.float8_e4m3fn,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    torch_dtype = dtype_map[dtype]
    weight = generate_random_tensor(torch_dtype, (num_experts, shape_n, shape_k))
    if torch_dtype.itemsize == 2:
        weight_scale: torch.Tensor | None = None
    elif block_shape is None:
        weight_scale = torch.randn((num_experts, shape_n), dtype=torch.float32, device="cuda:0")
    else:
        weight_scale = torch.randn(
            (num_experts, shape_n // block_shape[0], shape_k // block_shape[1]),
            dtype=torch.float32,
            device="cuda:0",
        )

    default_shape_m_list = [2**i for i in range(15)]
    benchmark_result: list[dict[str, int | float]] = []
    for shape_m in tqdm(shape_m_list or default_shape_m_list):
        input_scale: torch.Tensor | None
        num_groups = shape_k // block_shape[1] if block_shape is not None else 1
        if is_moe_down:
            inputs = generate_random_tensor(torch_dtype, (shape_m * top_k, shape_k))
            input_scale = torch.randn(
                (shape_m * top_k, num_groups),
                dtype=torch.float32,
                device="cuda:0",
            )
        else:
            inputs = generate_random_tensor(torch_dtype, (shape_m, shape_k))
            input_scale = torch.randn(
                (shape_m, num_groups),
                dtype=torch.float32,
                device="cuda:0",
            )

        if torch_dtype.itemsize == 2:
            input_scale = None

        torch.cuda.manual_seed(shape_m)
        config = get_triton_moe_config(
            num_experts=num_experts,
            shape_n=shape_n,
            shape_k=shape_k,
            shape_m=shape_m,
            top_k=top_k,
            torch_dtype=torch_dtype,
            is_moe_down=is_moe_down,
            block_shape=block_shape,
        )
        moe_tensors = generate_random_moe_tensors(
            shape_m=shape_m,
            num_experts=num_experts,
            top_k=top_k,
            balanced=balanced,
            block_size_config=config["BLOCK_SIZE_M"],
        )
        _, _, sorted_ids, expert_ids, num_tokens_padded = moe_tensors

        def run():
            outputs = torch.randn(
                (shape_m, top_k, shape_n),  # noqa
                dtype=torch.float16,
                device="cuda:0",
            )
            invoke_fused_moe_triton_kernel(
                A=inputs,  # noqa
                B=weight,
                C=outputs,
                A_scale=input_scale,  # noqa
                B_scale=weight_scale,
                topk_weights=None,  # noqa
                sorted_token_ids=sorted_ids,  # noqa
                expert_ids=expert_ids,  # noqa
                num_tokens_post_padded=num_tokens_padded,  # noqa
                mul_routed_weight=False,
                top_k=1 if is_moe_down else top_k,
                config=config,  # noqa
                compute_type=getattr(tl, out_dtype),
                use_fp8_w8a8=dtype == "float8e4m3",
                use_int8_w8a8=dtype == "int8",
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=block_shape is None,
                block_shape=block_shape,
            )

            return outputs

        torch.cuda.synchronize()
        outputs = run()
        t = triton.testing.do_bench(run, warmup=100, rep=1000)

        assert expert_ids is not None
        num_actived_experts = len(set(expert_ids.tolist()))
        nbytes = inputs.nbytes + outputs.nbytes
        if input_scale is not None:
            nbytes += input_scale.nbytes
        nbytes += weight.nbytes // num_experts * num_actived_experts
        if weight_scale is not None:
            nbytes += weight_scale.nbytes // num_experts * num_actived_experts

        res = {
            "shape_m": shape_m,
            "time": t,
            "memory_gbps": nbytes / t / 1e6,
            "compute_tops": shape_m * shape_n * shape_k * top_k * 2 / t / 1e9,
            "config": config,
        }
        benchmark_result.append(res)

    return benchmark_result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--shape_n", type=int, required=True)
    parser.add_argument("--shape_k", type=int, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["int8", "float8e4m3", "float16", "bfloat16"],
        required=True,
    )
    parser.add_argument("--out_dtype", type=str, choices=["float16", "bfloat16"], required=True)
    parser.add_argument("--num_experts", type=int, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--is_moe_down", default=False, action="store_true")
    parser.add_argument("--balanced", default=False, action="store_true")
    parser.add_argument("--shape_m_list", type=int, default=None, nargs="+")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--block_shape", type=str, default=None)

    args = parser.parse_args()
    benchmark_result = bench_triton_moe(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
        num_experts=args.num_experts,
        top_k=args.top_k,
        is_moe_down=args.is_moe_down,
        dtype=args.dtype,
        out_dtype=args.out_dtype,
        block_shape=args.block_shape,
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
