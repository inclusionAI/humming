import argparse
import json

import torch
import triton
from tqdm import tqdm

from humming import dtypes, ops
from humming.config import GemmType
from humming.layer import HummingLayer
from humming.tune import get_heuristics_config
from humming.utils.test import (
    generate_random_moe_tensors,
    random_fill_tensor,
    save_benchmark_result,
)


def bench_humming(
    shape_n: int,
    shape_k: int,
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
    bs_dtype: str,
    input_scale_group_size: int = 0,
    weight_scale_group_size: int = 0,
    num_experts: int = 0,
    top_k: int = 0,
    has_zero_point: bool = False,
    is_fp_zero_point: bool = False,
    use_f16_accum: bool = False,
    is_moe_down: bool = False,
    balanced: bool = False,
    expert_max_tokens: int | None = None,
    shape_m_list: list[int] | None = None,
    gemm_type: GemmType = GemmType.DENSE,
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
            "is_fp_zero_point": is_fp_zero_point,
        },
        input_config={"dtype": a_dtype, "group_size": input_scale_group_size},
        torch_dtype=torch_dtype,
    ).to("cuda:0")

    for tensor in layer.parameters():
        random_fill_tensor(tensor)
    layer.transform()
    meta = layer.humming_metas[""]

    default_shape_m_list = [2**i for i in range(15)]
    benchmark_result: list[dict[str, int | float]] = []
    for shape_m in tqdm(shape_m_list or default_shape_m_list):
        if gemm_type == GemmType.DENSE:
            actual_shape_m = shape_m
        elif gemm_type == GemmType.INDEXED:
            actual_shape_m = shape_m * (top_k if is_moe_down else 1)
        elif gemm_type == GemmType.GROUPED_CONTIGUOUS:
            actual_shape_m = shape_m * top_k
        else:
            assert num_experts is not None and expert_max_tokens is not None
            actual_shape_m = num_experts * expert_max_tokens

        inputs = torch.randn((actual_shape_m, shape_k), dtype=torch_dtype, device="cuda:0")
        input_scale: torch.Tensor | None = None
        if a_dtype not in ["float16", "bfloat16"]:
            inputs, input_scale = ops.quant_input(
                inputs,
                a_dtype,
                None,
                group_size=input_scale_group_size,
            )

        tuning_config = get_heuristics_config(
            meta=meta,
            use_f16_accum=use_f16_accum,
            gemm_type=gemm_type,
        )

        block_size_config: int | None = None
        if gemm_type == GemmType.INDEXED:
            routed_shape_m = shape_m * top_k
            for min_shape_m, max_shape_m, config in tuning_config:
                if routed_shape_m > min_shape_m and routed_shape_m <= max_shape_m:
                    block_size_config = config["block_shape"][0]
                    break
            assert block_size_config is not None

        torch.cuda.manual_seed(shape_m)
        if gemm_type == GemmType.DENSE:
            expert_layout = None
            sorted_ids = None
            expert_ids = None
            num_tokens_padded = None
        else:
            moe_tensors = generate_random_moe_tensors(
                shape_m=shape_m,
                num_experts=num_experts,
                top_k=top_k,
                gemm_type=gemm_type,
                balanced=balanced,
                block_size_config=block_size_config,
                expert_max_tokens=expert_max_tokens,
            )
            _, expert_layout, sorted_ids, expert_ids, num_tokens_padded = moe_tensors

        def run():
            valid_shape_m = 0
            if gemm_type == GemmType.GROUPED_MASKED:
                valid_shape_m = shape_m * top_k
            return layer(
                inputs=inputs,  # noqa
                input_scale=input_scale,  # noqa
                sorted_ids=sorted_ids,  # noqa
                expert_ids=expert_ids,  # noqa
                num_tokens_padded=num_tokens_padded,  # noqa
                expert_layout=expert_layout,  # noqa
                valid_shape_m=valid_shape_m,
                compute_config=json.dumps(
                    {
                        "use_f16_accum": use_f16_accum,
                        "gemm_type": gemm_type.value,
                    }
                ),
                tuning_config=tuning_config,  # noqa
                top_k=top_k if not is_moe_down else 1,
            )

        outputs = run()
        torch.cuda.synchronize()
        t = triton.testing.do_bench(run, warmup=100, rep=1000)

        nbytes = inputs.nbytes + outputs.nbytes
        if gemm_type == GemmType.GROUPED_MASKED:
            nbytes = nbytes // actual_shape_m * shape_m * top_k
        if input_scale is not None:
            nbytes += input_scale.nbytes
        for tensor in layer.state_dict().values():
            if gemm_type == GemmType.DENSE:
                nbytes += tensor.nbytes
            elif gemm_type == GemmType.GROUPED_MASKED:
                assert expert_layout is not None
                num_actived_experts = int((expert_layout > 0).sum().item())
                nbytes += tensor.nbytes // num_experts * num_actived_experts
            elif gemm_type == GemmType.GROUPED_CONTIGUOUS:
                assert expert_layout is not None
                num_actived_experts = int((expert_layout[1:] > expert_layout[:-1]).sum().item())
                nbytes += tensor.nbytes // num_experts * num_actived_experts
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
    activation_dtypes = [
        "float16",
        "bfloat16",
        "float8e4m3",
        "float8e5m2",
        "float4e2m1",
        "int8",
        "int4",
    ]
    scale_dtypes = ["float16", "bfloat16", "float8e4m3", "float8e5m2", "float8e8m0"]
    f16_dtypes = ["float16", "bfloat16"]
    parser.add_argument("--a_dtype", type=str, choices=activation_dtypes, required=True)
    parser.add_argument("--b_dtype", type=str, required=True)
    parser.add_argument("--bs_dtype", type=str, choices=scale_dtypes, required=True)
    parser.add_argument("--c_dtype", type=str, choices=f16_dtypes, required=True)
    parser.add_argument("--input_scale_group_size", type=int, default=0)
    parser.add_argument("--weight_scale_group_size", type=int, default=0)
    parser.add_argument("--zero_point", default=False, action="store_true")
    parser.add_argument("--use_fp_zero_point", default=False, action="store_true")
    parser.add_argument("--use_f16_accum", default=False, action="store_true")
    parser.add_argument("--num_experts", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--is_moe_down", default=False, action="store_true")
    parser.add_argument("--balanced", default=False, action="store_true")
    parser.add_argument("--expert_max_tokens", type=int, default=None)
    parser.add_argument("--shape_m_list", type=int, default=None, nargs="+")
    parser.add_argument(
        "--gemm_type",
        type=GemmType,
        choices=list(GemmType),
        default=GemmType.DENSE,
    )
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()
    benchmark_result = bench_humming(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
        a_dtype=args.a_dtype,
        b_dtype=args.b_dtype,
        c_dtype=args.c_dtype,
        bs_dtype=args.bs_dtype,
        num_experts=args.num_experts,
        top_k=args.top_k,
        input_scale_group_size=args.input_scale_group_size,
        weight_scale_group_size=args.weight_scale_group_size,
        has_zero_point=args.zero_point or args.use_fp_zero_point,
        is_fp_zero_point=args.use_fp_zero_point,
        use_f16_accum=args.use_f16_accum,
        shape_m_list=args.shape_m_list,
        is_moe_down=args.is_moe_down,
        balanced=args.balanced,
        expert_max_tokens=args.expert_max_tokens,
        gemm_type=args.gemm_type,
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
