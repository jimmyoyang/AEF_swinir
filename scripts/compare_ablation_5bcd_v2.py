#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import csv
import re
from pathlib import Path


VAL_RE = re.compile(
    r"Validation Metrics \| PSNR: (?P<psnr>[0-9.]+) \| SSIM: (?P<ssim>[0-9.]+) \| ERGAS: (?P<ergas>[0-9.]+) \| SAM: (?P<sam>[0-9.]+)"
)
BEST_RE = re.compile(r"New best metric: (?P<best>[0-9.]+)")


def latest_run_dir(exp_dir):
    if not exp_dir.exists():
        return None
    dirs = [x for x in exp_dir.iterdir() if x.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda x: x.stat().st_mtime)


def parse_training_log(log_path):
    if not log_path.exists():
        return None
    best = None
    last = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = BEST_RE.search(line)
            if m:
                best = float(m.group("best"))
            v = VAL_RE.search(line)
            if v:
                last = {
                    "psnr": float(v.group("psnr")),
                    "ssim": float(v.group("ssim")),
                    "ergas": float(v.group("ergas")),
                    "sam": float(v.group("sam")),
                }
    if last is None:
        return None
    last["best_psnr"] = best if best is not None else last["psnr"]
    return last


def parse_post_stats(csv_path):
    if not csv_path.exists():
        return {}
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    tail = rows[-1]
    return {
        "post_tau": float(tail.get("post_tau", 0.0)),
        "routing_entropy": float(tail.get("routing_entropy", 0.0)),
        "active_groups": float(tail.get("active_groups", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="training_logs/experiments_v2")
    parser.add_argument("--output", type=str, default="ablation_v2_comparison.csv")
    args = parser.parse_args()

    root = Path(args.root)
    schemes = [
        ("5a_nopost_bestpre", root / "ablation_v2_5a_nopost_bestpre"),
        ("5b_scheme1_gumbel", root / "ablation_v2_5b_scheme1_gumbel"),
        ("5c_scheme3_dual_branch", root / "ablation_v2_5c_scheme3_dual_branch"),
        ("5d_scheme6_matrix", root / "ablation_v2_5d_scheme6_matrix"),
    ]

    results = []
    for name, exp_dir in schemes:
        run_dir = latest_run_dir(exp_dir)
        if run_dir is None:
            continue
        metric = parse_training_log(run_dir / "training.log")
        if metric is None:
            continue
        post_stats = parse_post_stats(run_dir / "post_v2_stats.csv")
        row = {
            "scheme": name,
            "run_dir": str(run_dir),
            **metric,
            **post_stats,
        }
        results.append(row)

    if not results:
        print("No results found.")
        return

    fields = sorted({k for r in results for k in r.keys()})
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {args.output}")
    print("Summary:")
    for r in results:
        print(
            f"- {r['scheme']}: best_psnr={r.get('best_psnr', 0):.4f}, "
            f"last_psnr={r.get('psnr', 0):.4f}, last_ssim={r.get('ssim', 0):.4f}"
        )


if __name__ == "__main__":
    main()
