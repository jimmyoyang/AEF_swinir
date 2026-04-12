#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from pathlib import Path


ITER_RE = re.compile(r"(?:Val\s+)?Iter\s+(?P<iter>\d+)")
VAL_RE = re.compile(
    r"Validation Metrics \| PSNR: (?P<psnr>[0-9.]+) \| SSIM: (?P<ssim>[0-9.]+)"
)


def parse_log_metrics(log_path: Path):
    if not log_path.exists():
        return None

    current_iter = None
    rows = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_iter = ITER_RE.search(line)
            if m_iter:
                current_iter = int(m_iter.group("iter"))

            m_val = VAL_RE.search(line)
            if m_val:
                rows.append(
                    {
                        "iter": current_iter,
                        "psnr": float(m_val.group("psnr")),
                        "ssim": float(m_val.group("ssim")),
                    }
                )

    if not rows:
        return None

    best_row = max(rows, key=lambda x: x["psnr"])
    last_row = rows[-1]

    return {
        "best_psnr": best_row["psnr"],
        "best_ssim": best_row["ssim"],
        "best_iter": best_row["iter"],
        "last_psnr": last_row["psnr"],
        "last_ssim": last_row["ssim"],
        "last_iter": last_row["iter"],
        "val_count": len(rows),
    }


def latest_run_dir(exp_dir: Path):
    if not exp_dir.exists() or not exp_dir.is_dir():
        return None
    run_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda x: x.stat().st_mtime)


def collect_group_results(group_root: Path, group_name: str):
    rows = []
    if not group_root.exists() or not group_root.is_dir():
        return rows

    for exp_dir in sorted([d for d in group_root.iterdir() if d.is_dir()]):
        run_dir = latest_run_dir(exp_dir)
        if run_dir is None:
            continue

        metrics = parse_log_metrics(run_dir / "training.log")
        if metrics is None:
            continue

        rows.append(
            {
                "scheme": exp_dir.name,
                "group": group_name,
                "run_dir": str(run_dir),
                **metrics,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Rank recovered 3b against all v2/v3 experiments using training.log metrics."
    )
    parser.add_argument(
        "--recovered-log",
        type=str,
        required=True,
        help="Recovered 3b training.log path",
    )
    parser.add_argument(
        "--recovered-name",
        type=str,
        default="ablation_3b_recovered",
        help="Display name for recovered 3b row",
    )
    parser.add_argument(
        "--v2-root",
        type=str,
        default="training_logs/experiments_v2",
        help="Root directory of v2 experiments",
    )
    parser.add_argument(
        "--v3-root",
        type=str,
        default="training_logs/experiments_v3",
        help="Root directory of v3 experiments",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="debug_output/ranking_recovered_3b_vs_all_v2_v3.csv",
        help="Output ranking CSV",
    )
    args = parser.parse_args()

    recovered_log = Path(args.recovered_log)
    v2_root = Path(args.v2_root)
    v3_root = Path(args.v3_root)
    output = Path(args.output)

    recovered_metrics = parse_log_metrics(recovered_log)
    if recovered_metrics is None:
        raise RuntimeError(f"No validation metrics found in recovered log: {recovered_log}")

    results = [
        {
            "scheme": args.recovered_name,
            "group": "recovered_3b",
            "run_dir": str(recovered_log.parent),
            **recovered_metrics,
        }
    ]

    results.extend(collect_group_results(v2_root, "v2"))
    results.extend(collect_group_results(v3_root, "v3"))

    if not results:
        raise RuntimeError("No results collected.")

    ref_best_psnr = recovered_metrics["best_psnr"]
    for row in results:
        row["delta_vs_recovered_best_psnr"] = row["best_psnr"] - ref_best_psnr

    results.sort(key=lambda x: x["best_psnr"], reverse=True)
    for idx, row in enumerate(results, start=1):
        row["rank"] = idx

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "scheme",
        "group",
        "best_psnr",
        "best_ssim",
        "best_iter",
        "last_psnr",
        "last_ssim",
        "last_iter",
        "val_count",
        "delta_vs_recovered_best_psnr",
        "run_dir",
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("===== Ranking: Recovered 3b vs All v2/v3 =====")
    for row in results:
        print(
            f"#{row['rank']:<2} {row['scheme']:<50} "
            f"best_psnr={row['best_psnr']:.4f} "
            f"best_ssim={row['best_ssim']:.4f} "
            f"delta_vs_rec={row['delta_vs_recovered_best_psnr']:+.4f}"
        )
    print(f"CSV saved: {output}")


if __name__ == "__main__":
    main()
