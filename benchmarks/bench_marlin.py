import argparse

import torch
import triton
import vllm._custom_ops as vllm_ops
from tqdm import tqdm
from vllm.scalar_type import scalar_types

from humming import dtypes, ops
from humming.layer import HummingLayer
from humming.utils.test import (
    generate_random_moe_tensors,
    random_fill_tensor,
    save_benchmark_result,
)


def bench_marlin(
    shape_n: int,
    shape_k: int,
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
    bs_dtype: str,
    weight_scale_group_size: int = 0,
    num_experts: int | None = None,
    top_k: int = 0,
    has_zero_point: bool = False,
    shape_m_list: list[int] | None = None,
    is_moe_down: bool = False,
    balanced: bool = False,
) -> list[dict[str, int | float]]:
    torch_dtype = dtypes.torch_dtype_map[dtypes.DataType.from_str(c_dtype)]
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        weight_config={
            "dtype": b_dtype,
            "group_size": weight_scale_group_size,
            "scale_dtype": bs_dtype,
            "has_zero_point": has_zero_point,
        },
        input_config={"dtype": a_dtype},
        torch_dtype=torch_dtype,
    ).to("cuda:0")

    for tensor in layer.parameters():
        random_fill_tensor(tensor)
    layer.transform()

    if b_dtype in ["int4", "uint4"] and has_zero_point:
        vllm_scalar_type = scalar_types.uint4
    elif b_dtype in ["int4", "uint4"] and not has_zero_point:
        vllm_scalar_type = scalar_types.uint4b8
    elif b_dtype in ["int8", "uint8"] and not has_zero_point:
        vllm_scalar_type = scalar_types.uint8b128
    elif b_dtype == "float8e4m3":
        vllm_scalar_type = scalar_types.float8_e4m3fn
    elif b_dtype == "float4e2m1":
        vllm_scalar_type = scalar_types.float4_e2m1f
    else:
        raise ValueError(f"unsupported dtype for marlin: {b_dtype}")

    assert isinstance(layer.weight, torch.Tensor)
    weight = layer.weight.data
    if a_dtype not in ["float16", "bfloat16"] and num_experts is None:
        weight = weight.view(weight.size(0) * 2, -1)
    elif a_dtype not in ["float16", "bfloat16"] and num_experts is not None:
        weight = weight.view(num_experts, weight.size(1) * 2, -1)

    default_shape_m_list = [2**i for i in range(15)]
    benchmark_result: list[dict[str, int | float]] = []
    for shape_m in tqdm(shape_m_list or default_shape_m_list):
        if num_experts is not None and is_moe_down:
            inputs = torch.randn((shape_m * top_k, shape_k), dtype=torch_dtype, device="cuda:0")
        else:
            inputs = torch.randn((shape_m, shape_k), dtype=torch_dtype, device="cuda:0")

        input_scale: torch.Tensor | None = None
        if a_dtype not in ["float16", "bfloat16"]:
            inputs, input_scale = ops.quant_input(inputs, a_dtype)

        if num_experts is not None:
            for block_size_m in [8, 16, 32, 48, 64]:
                if block_size_m == 8 and a_dtype not in ["float16", "bfloat16"]:
                    continue
                if shape_m * top_k / num_experts / block_size_m < 0.9:
                    break

            torch.cuda.manual_seed(shape_m)
            moe_tensors = generate_random_moe_tensors(
                shape_m=shape_m,
                num_experts=num_experts,
                top_k=top_k,
                block_size_config=block_size_m,
                balanced=balanced,
            )
            _, _, sorted_ids, expert_ids, num_tokens_padded = moe_tensors

            # The marlin kernel always dereferences topk_weights (it does not
            # check for a null pointer), so pass a real tensor even though
            # mul_topk_weights=False means the values are unused.
            topk_weights = torch.ones(
                (shape_m, top_k),
                dtype=torch.float32,
                device="cuda:0",
            )

        def run_dense():
            return vllm_ops.marlin_gemm(
                a=inputs,  # noqa
                c=None,
                b_q_weight=weight,
                b_bias=None,
                b_scales=layer.weight_scale,
                a_scales=input_scale,  # noqa
                global_scale=None,
                b_zeros=getattr(layer, "zero_point", None),
                g_idx=None,
                perm=None,
                workspace=layer.locks,
                b_q_type=vllm_scalar_type,
                size_m=inputs.size(0),  # noqa
                size_n=shape_n,
                size_k=shape_k,
                use_atomic_add=True,
            )

        def run_moe():
            return vllm_ops.moe_wna16_marlin_gemm(
                input=inputs,  # noqa
                output=None,
                b_qweight=weight,
                b_bias=None,
                b_scales=layer.weight_scale,
                a_scales=input_scale,  # noqa
                global_scale=None,
                b_qzeros=getattr(layer, "zero_point", None),
                g_idx=None,
                perm=None,
                workspace=layer.locks,
                sorted_token_ids=sorted_ids,  # noqa
                expert_ids=expert_ids,  # noqa
                num_tokens_past_padded=num_tokens_padded,  # noqa
                topk_weights=topk_weights,  # noqa
                mul_topk_weights=False,
                b_q_type=vllm_scalar_type,
                size_m=inputs.size(0),  # noqa
                size_n=shape_n,
                size_k=shape_k,
                top_k=top_k,
                moe_block_size=block_size_m,  # noqa
                is_k_full=True,
                use_atomic_add=False,
                use_fp32_reduce=True,
                is_zp_float=False,
            )

        run = run_dense if num_experts is None else run_moe
        outputs = run()
        torch.cuda.synchronize()
        t = triton.testing.do_bench(run, warmup=100, rep=1000)

        nbytes = inputs.nbytes + outputs.nbytes
        if input_scale is not None:
            nbytes += input_scale.nbytes
        for tensor in layer.state_dict().values():
            if num_experts is None:
                nbytes += tensor.nbytes
            else:
                assert expert_ids is not None
                num_actived_experts = len(set(expert_ids.tolist()))
                nbytes += tensor.nbytes // num_experts * num_actived_experts

        res = {
            "shape_m": shape_m,
            "time": t,
            "memory_gbps": nbytes / t / 1e6,
            "compute_tops": shape_m * shape_n * shape_k * (top_k or 1) * 2 / t / 1e9,
        }
        benchmark_result.append(res)

    return benchmark_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape_n", type=int, required=True)
    parser.add_argument("--shape_k", type=int, required=True)
    activation_dtypes = ["float16", "bfloat16", "float8e4m3", "int8"]
    scale_dtypes = ["float16", "bfloat16", "float8e4m3", "float8e8m0"]
    f16_dtypes = ["float16", "bfloat16"]
    parser.add_argument("--a_dtype", type=str, choices=activation_dtypes, required=True)
    parser.add_argument("--b_dtype", type=str, required=True)
    parser.add_argument("--bs_dtype", type=str, choices=scale_dtypes, required=True)
    parser.add_argument("--c_dtype", type=str, choices=f16_dtypes, required=True)
    parser.add_argument("--weight_scale_group_size", type=int, default=0)
    parser.add_argument("--zero_point", default=False, action="store_true")
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--is_moe_down", default=False, action="store_true")
    parser.add_argument("--balanced", default=False, action="store_true")
    parser.add_argument("--shape_m_list", type=int, default=None, nargs="+")
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()
    benchmark_result = bench_marlin(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
        a_dtype=args.a_dtype,
        b_dtype=args.b_dtype,
        c_dtype=args.c_dtype,
        bs_dtype=args.bs_dtype,
        num_experts=args.num_experts,
        top_k=args.top_k,
        weight_scale_group_size=args.weight_scale_group_size,
        has_zero_point=args.zero_point,
        shape_m_list=args.shape_m_list,
        is_moe_down=args.is_moe_down,
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
