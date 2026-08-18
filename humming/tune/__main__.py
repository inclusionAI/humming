import argparse
import json
import os

from humming.config import GemmType


def parse_args():
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
    parser.add_argument("--weight_scale_group_size", type=int, required=True)
    parser.add_argument("--input_scale_group_size", type=int, default=0)

    parser.add_argument(
        "--gemm_type",
        type=str,
        choices=[GemmType.DENSE.value, GemmType.GROUPED_MASKED.value, GemmType.INDEXED.value],
        required=True,
    )
    parser.add_argument("--num_experts", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--expert_max_tokens", type=int, default=None)
    parser.add_argument("--balanced", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--is_moe_down", default=False, action="store_true")
    parser.add_argument("--shape_m_list", type=int, nargs="+", default=None)
    parser.add_argument("--no-save", default=False, action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fast", default=False, action="store_true")
    return parser.parse_args()


def validate_args(args, gemm_type: GemmType) -> None:
    default_grid = args.shape_m_list is None
    if default_grid:
        if gemm_type == GemmType.DENSE:
            args.shape_m_list = [256, 4096]
        else:
            args.shape_m_list = [64, 256, 1024, 4096]

    if gemm_type in (GemmType.GROUPED_MASKED, GemmType.INDEXED):
        if args.num_experts <= 0:
            raise ValueError("--num_experts must be positive for MoE gemm types")
        if args.top_k <= 0:
            raise ValueError("--top_k must be positive for MoE gemm types")

    if gemm_type == GemmType.INDEXED:
        # indexed shape_m is routed rows (tokens * top_k); a grid value that
        # top_k does not divide would fail inside every worker bench.
        misaligned = [m for m in args.shape_m_list if m % args.top_k]
        if misaligned and default_grid:
            args.shape_m_list = sorted(
                {-(-m // args.top_k) * args.top_k for m in args.shape_m_list}
            )
        elif misaligned:
            raise ValueError(
                f"--shape_m_list values {misaligned} are not divisible by "
                f"--top_k={args.top_k}; indexed shape_m is routed rows"
            )


def config_summary(config: dict) -> str:
    parts = [
        f"block={config.get('block_shape')}",
        f"warp={config.get('warp_shape')}",
        f"stages={config.get('num_stages')}",
        f"ctas={config.get('num_ctas_per_sm')}",
        f"sms={config.get('num_sms')}",
    ]
    if config.get("use_stream_k"):
        parts.append("stream_k")
    if config.get("use_tma"):
        parts.append("tma")
    if config.get("use_warp_spec"):
        parts.append("warp_spec")
    return " ".join(parts)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def format_range(min_shape_m: int, max_shape_m: int) -> str:
    max_text = "inf" if max_shape_m == 1 << 30 else str(max_shape_m)
    return f"({min_shape_m}, {max_text}]"


def make_summary_rows(table: list, per_m: list[dict]) -> list[dict]:
    rows = []
    for min_shape_m, max_shape_m, config in table:
        samples = [
            item
            for item in per_m
            if item["shape_m"] > min_shape_m and item["shape_m"] <= max_shape_m
        ]
        rows.append(
            {
                "m_range": format_range(min_shape_m, max_shape_m),
                "sample_m": ", ".join(str(item["shape_m"]) for item in samples),
                "fine_ms": ", ".join(format_float(item["fine_ms"]) for item in samples),
                "speedup": ", ".join(format_float(item["speedup"]) for item in samples),
                "fallback": any(item["used_fallback"] for item in samples),
                "config": config_summary(config),
            }
        )
    return rows


def print_summary(table: list, per_m: list[dict]) -> None:
    from tabulate import tabulate

    print(
        tabulate(
            make_summary_rows(table, per_m),
            headers="keys",
            tablefmt="grid",
            numalign="right",
            floatfmt=".4f",
        )
    )


def write_output(path: str, payload: dict) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> None:
    from humming.tune import cache
    from humming.tune._worker import create_layer
    from humming.tune.measured import _make_flags, _run_search, _shape_m_values

    args = parse_args()
    gemm_type = GemmType(args.gemm_type)
    validate_args(args, gemm_type)

    layer, _torch_dtype, _weight_ref = create_layer(args, gemm_type)
    sublayer_name = ""
    meta = layer.humming_metas[sublayer_name]

    shape_m_values = _shape_m_values(args.shape_m_list)
    table, per_m = _run_search(
        meta,
        gemm_type,
        shape_m_values,
        top_k=args.top_k,
        is_moe_down=args.is_moe_down,
        balanced=args.balanced,
        expert_max_tokens=args.expert_max_tokens,
        input_scale_group_size=args.input_scale_group_size,
        fast=args.fast,
    )

    cache_path = None
    if not args.no_save:
        flags = _make_flags(False, False, False)
        fingerprint = cache.current_fingerprint()
        cache_path = cache.save_table(
            meta,
            gemm_type,
            flags,
            fingerprint,
            table,
            shape_m_list=shape_m_values,
        )
        print(f"saved: {cache_path}")

    if args.output is not None:
        write_output(
            args.output,
            {
                "cache_path": cache_path,
                "table": table,
                "per_m": per_m,
            },
        )

    print_summary(table, per_m)


if __name__ == "__main__":
    main()
