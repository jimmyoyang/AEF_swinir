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
        raise FileNotFoundError(f"Log file not found: {log_path}")

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
        raise RuntimeError(f"No validation metrics found in: {log_path}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Compare recovered/current runs using training-log validation metrics (weekly-report consistent scale)."
    )
    parser.add_argument(
        "--recovered-log",
        type=str,
        required=True,
        help="Path to recovered run training.log",
    )
    parser.add_argument(
        "--current-log",
        type=str,
        required=True,
        help="Path to current run training.log",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="debug_output/compare_training_log_metrics_recovered_vs_current.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    rec_log = Path(args.recovered_log)
    cur_log = Path(args.current_log)
    out_csv = Path(args.output)

    rec = parse_log_metrics(rec_log)
    cur = parse_log_metrics(cur_log)

    delta_best_psnr = rec["best_psnr"] - cur["best_psnr"]
    delta_best_ssim = rec["best_ssim"] - cur["best_ssim"]
    delta_last_psnr = rec["last_psnr"] - cur["last_psnr"]
    delta_last_ssim = rec["last_ssim"] - cur["last_ssim"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run",
            "log_path",
            "best_psnr",
            "best_ssim",
            "best_iter",
            "last_psnr",
            "last_ssim",
            "last_iter",
            "val_count",
        ])
        w.writerow([
            "recovered",
            str(rec_log),
            f"{rec['best_psnr']:.4f}",
            f"{rec['best_ssim']:.4f}",
            rec["best_iter"],
            f"{rec['last_psnr']:.4f}",
            f"{rec['last_ssim']:.4f}",
            rec["last_iter"],
            rec["val_count"],
        ])
        w.writerow([
            "current",
            str(cur_log),
            f"{cur['best_psnr']:.4f}",
            f"{cur['best_ssim']:.4f}",
            cur["best_iter"],
            f"{cur['last_psnr']:.4f}",
            f"{cur['last_ssim']:.4f}",
            cur["last_iter"],
            cur["val_count"],
        ])
        w.writerow([
            "delta_recovered_minus_current",
            "-",
            f"{delta_best_psnr:+.4f}",
            f"{delta_best_ssim:+.4f}",
            "-",
            f"{delta_last_psnr:+.4f}",
            f"{delta_last_ssim:+.4f}",
            "-",
            "-",
        ])

    print("===== Weekly-Report Scale Comparison (from training.log) =====")
    print(
        f"Recovered | best={rec['best_psnr']:.4f}/{rec['best_ssim']:.4f} @iter {rec['best_iter']} | "
        f"last={rec['last_psnr']:.4f}/{rec['last_ssim']:.4f} @iter {rec['last_iter']}"
    )
    print(
        f"Current   | best={cur['best_psnr']:.4f}/{cur['best_ssim']:.4f} @iter {cur['best_iter']} | "
        f"last={cur['last_psnr']:.4f}/{cur['last_ssim']:.4f} @iter {cur['last_iter']}"
    )
    print(
        f"Delta(rec-current) | best_psnr={delta_best_psnr:+.4f}, best_ssim={delta_best_ssim:+.4f}, "
        f"last_psnr={delta_last_psnr:+.4f}, last_ssim={delta_last_ssim:+.4f}"
    )
    print(f"CSV saved: {out_csv}")


if __name__ == "__main__":
    main()
