"""Sweep the measured-config tuner across an m grid and report vs heuristics.

Runs `python -m humming.tune` per spec, collects the per-m winner/baseline
timings from --output payloads, and renders a markdown report. Intended for
producing tuning-effect evidence (PR tables) and for re-running after
heuristic or kernel changes.

Usage:
    python tools/tune_sweep.py --out-dir /root/tune_sweep [--fast]
    python tools/tune_sweep.py --spec my_specs.jsonl --report report.md

Each spec line is a JSON object of `python -m humming.tune` CLI arguments
plus a `name`; see DEFAULT_SPECS.
"""

import argparse
import json
import math
import os
import subprocess
import sys

_M_GRID = (
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
    1024, 1536, 2048, 3072, 4096, 6144, 8192,
)

DEFAULT_SPECS = [
    {
        "name": "dense-6144x7168",
        "shape_n": 6144, "shape_k": 7168,
        "a_dtype": "float8e4m3", "b_dtype": "float4e2m1",
        "bs_dtype": "float8e8m0", "c_dtype": "bfloat16",
        "weight_scale_group_size": 32, "input_scale_group_size": 128,
        "gemm_type": "dense", "num_spares": 0,
    },
    {
        "name": "indexed-w13-tp8",
        "shape_n": 1536, "shape_k": 7168, "num_experts": 384, "top_k": 8,
        "a_dtype": "float8e4m3", "b_dtype": "float4e2m1",
        "bs_dtype": "float8e8m0", "c_dtype": "bfloat16",
        "weight_scale_group_size": 32, "input_scale_group_size": 128,
        "gemm_type": "indexed", "num_spares": 0,
    },
    {
        "name": "indexed-w2-tp8",
        "shape_n": 7168, "shape_k": 768, "num_experts": 384, "top_k": 8,
        "is_moe_down": True,
        "a_dtype": "float8e4m3", "b_dtype": "float4e2m1",
        "bs_dtype": "float8e8m0", "c_dtype": "bfloat16",
        "weight_scale_group_size": 32, "input_scale_group_size": 128,
        "gemm_type": "indexed", "num_spares": 0,
    },
]


def _m_grid_for(spec, m_grid):
    if spec["gemm_type"] != "indexed":
        return list(m_grid)
    top_k = spec["top_k"]
    aligned = sorted({-(-m // top_k) * top_k for m in m_grid})
    return aligned


def run_spec(spec, out_dir, m_grid, fast, python=sys.executable):
    name = spec["name"]
    output = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(output):
        os.remove(output)  # never let stale evidence survive a failed run
    cmd = [python, "-m", "humming.tune", "--output", output]
    for key, value in spec.items():
        if key == "name":
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])
    cmd.append("--shape_m_list")
    cmd.extend(str(m) for m in _m_grid_for(spec, m_grid))
    if fast:
        cmd.append("--fast")
    log_path = os.path.join(out_dir, f"{name}.log")
    print(f"[tune_sweep] {name}: {' '.join(cmd)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"[tune_sweep] {name}: FAILED (exit {result.returncode}), "
              f"see {log_path}", flush=True)
        return None
    return output


def render_report(payloads):
    lines = []
    overall = []
    for name, payload in payloads:
        per_m = payload.get("per_m") or []
        lines.append(f"\n## {name}\n")
        lines.append("| m | heuristic ms | tuned ms | speedup | winner |")
        lines.append("|---:|---:|---:|---:|:---|")
        for row in per_m:
            heur = row.get("heuristic_ms")
            fine = row.get("fine_ms")
            speedup = row.get("speedup")
            same = row.get("same_as_heuristic")
            fallback = row.get("used_fallback")
            if fallback:
                winner = "heuristic (fallback)"
            elif same:
                winner = "heuristic (tie)"
            else:
                winner = "tuned"
            if speedup is not None and not same and not fallback:
                overall.append(speedup)
            lines.append(
                "| {m} | {h} | {t} | {s} | {w} |".format(
                    m=row.get("shape_m"),
                    h="-" if heur is None else f"{heur:.4f}",
                    t="-" if fine is None else f"{fine:.4f}",
                    s="-" if speedup is None else f"{speedup:.3f}x",
                    w=winner,
                )
            )
        wins = sum(
            1 for row in per_m
            if not row.get("same_as_heuristic") and not row.get("used_fallback")
        )
        ties = len(per_m) - wins
        lines.append(f"\nwins: {wins} / ties(=heuristic): {ties} / total: {len(per_m)}")
    if overall:
        geomean = math.exp(sum(math.log(s) for s in overall) / len(overall))
        lines.insert(0, (
            f"# Tune sweep report\n\n"
            f"{len(overall)} tuned wins across all specs; "
            f"geomean speedup of wins: {geomean:.3f}x; "
            f"remaining points tie with the heuristic (the measurement gate "
            f"never selects a config slower than the heuristic baseline)."
        ))
    else:
        lines.insert(0, "# Tune sweep report\n\n(no tuned wins recorded)")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, default=None,
                        help="JSONL file of specs; default: built-in specs")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--report", type=str, default=None,
                        help="default: <out-dir>/report.md")
    parser.add_argument("--m-grid", type=int, nargs="+", default=list(_M_GRID))
    parser.add_argument("--fast", default=False, action="store_true")
    parser.add_argument("--report-only", default=False, action="store_true",
                        help="skip tuning; rebuild the report from out-dir")
    args = parser.parse_args()

    if args.spec:
        with open(args.spec, encoding="utf-8") as f:
            specs = [json.loads(line) for line in f if line.strip()]
    else:
        specs = DEFAULT_SPECS

    os.makedirs(args.out_dir, exist_ok=True)
    payloads = []
    failed = []
    for spec in specs:
        output = os.path.join(args.out_dir, f"{spec['name']}.json")
        if not args.report_only:
            output = run_spec(spec, args.out_dir, args.m_grid, args.fast)
        if output and os.path.exists(output):
            with open(output, encoding="utf-8") as f:
                payloads.append((spec["name"], json.load(f)))
        else:
            failed.append(spec["name"])

    report = render_report(payloads)
    if failed:
        report = (
            "# !! INCOMPLETE SWEEP: %d spec(s) FAILED or missing: %s\n\n"
            % (len(failed), ", ".join(failed))
        ) + report
    report_path = args.report or os.path.join(args.out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[tune_sweep] report: {report_path}", flush=True)
    print(report, flush=True)
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
