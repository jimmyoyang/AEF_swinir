#!/usr/bin/env python3
"""
Generate a unified v2/v3 comparison table with dual deltas:
1) delta vs group no-post baseline (v2 or v3)
2) delta vs anchor PSNR (user-specified)

Default input expects the schema in ablation_v3_full_vs_v2.csv.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _to_float(value: str) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "":
        return float("nan")
    return float(text)


def _infer_group(scheme: str) -> str:
    s = (scheme or "").strip().lower()
    if s.startswith("v2_"):
        return "v2"
    if s.startswith("v3_"):
        return "v3"
    return "unknown"


def _find_nopost_baselines(rows: List[Dict[str, str]]) -> Dict[str, float]:
    baselines: Dict[str, float] = {}
    for row in rows:
        scheme = (row.get("scheme") or "").lower()
        if "nopost" not in scheme:
            continue
        group = _infer_group(row.get("scheme", ""))
        best_psnr = _to_float(row.get("best_psnr", ""))
        if group not in baselines or best_psnr > baselines[group]:
            baselines[group] = best_psnr
    return baselines


_VAL_RE = re.compile(r"Validation Metrics\s*\|\s*PSNR:\s*([0-9]+\.[0-9]+)\s*\|\s*SSIM:\s*([0-9]+\.[0-9]+)")


def _parse_training_log(log_path: Path) -> Dict[str, float]:
    best_psnr = -1.0
    best_ssim = float("nan")
    last_psnr = float("nan")
    last_ssim = float("nan")
    val_count = 0

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _VAL_RE.search(line)
            if not m:
                continue
            p = float(m.group(1))
            s = float(m.group(2))
            val_count += 1
            last_psnr, last_ssim = p, s
            if p > best_psnr:
                best_psnr, best_ssim = p, s

    if val_count == 0:
        raise RuntimeError(f"No validation metrics found in: {log_path}")

    return {
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "last_psnr": last_psnr,
        "last_ssim": last_ssim,
        "val_count": float(val_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2/v3 dual-delta comparison report")
    parser.add_argument(
        "--input",
        default="ablation_v3_full_vs_v2.csv",
        help="Input CSV that includes v2_* and v3_* schemes",
    )
    parser.add_argument(
        "--output",
        default="debug_output/v2_v3_dual_delta_report.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--anchor-psnr",
        type=float,
        default=14.6694,
        help="Anchor best PSNR for delta comparison",
    )
    parser.add_argument(
        "--anchor-name",
        default="anchor_14.6694",
        help="Label for anchor in output metadata column",
    )
    parser.add_argument(
        "--from-logs",
        action="store_true",
        help="Parse best/last metrics from each run_dir/training.log instead of using CSV metric columns",
    )
    parser.add_argument(
        "--anchor-log",
        default="",
        help="Optional anchor training.log path; when provided, --anchor-psnr is ignored",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Input CSV has no rows: {input_path}")

    if args.anchor_log:
        anchor_log = Path(args.anchor_log)
        if not anchor_log.exists():
            raise FileNotFoundError(f"Anchor log not found: {anchor_log}")
        anchor_info = _parse_training_log(anchor_log)
        args.anchor_psnr = anchor_info["best_psnr"]

    if args.from_logs:
        for row in rows:
            run_dir = (row.get("run_dir") or "").strip()
            log_path = Path(run_dir) / "training.log"
            row["log_path"] = str(log_path)
            if not run_dir or not log_path.exists():
                row["log_status"] = "log_missing"
                continue
            try:
                info = _parse_training_log(log_path)
                row["best_psnr"] = f"{info['best_psnr']:.4f}"
                row["ssim"] = f"{info['last_ssim']:.4f}"
                row["psnr"] = f"{info['last_psnr']:.4f}"
                row["best_ssim_from_log"] = f"{info['best_ssim']:.4f}"
                row["val_count_from_log"] = str(int(info["val_count"]))
                row["log_status"] = "ok"
            except Exception:
                row["log_status"] = "parse_failed"

    baselines = _find_nopost_baselines(rows)
    if "v2" not in baselines or "v3" not in baselines:
        raise RuntimeError(
            f"Could not find both v2/v3 nopost baselines. Found: {baselines}"
        )

    out_rows: List[Dict[str, str]] = []
    for row in rows:
        scheme = row.get("scheme", "")
        group = _infer_group(scheme)
        best_psnr = _to_float(row.get("best_psnr", ""))
        last_psnr = _to_float(row.get("psnr", ""))

        group_baseline = baselines.get(group, float("nan"))
        delta_vs_group_nopost = best_psnr - group_baseline
        delta_vs_anchor = best_psnr - args.anchor_psnr

        out = dict(row)
        out["group"] = group
        out["group_nopost_best_psnr"] = f"{group_baseline:.4f}"
        out["delta_best_psnr_vs_group_nopost"] = f"{delta_vs_group_nopost:+.4f}"
        out["anchor_name"] = args.anchor_name
        out["anchor_best_psnr"] = f"{args.anchor_psnr:.4f}"
        out["delta_best_psnr_vs_anchor"] = f"{delta_vs_anchor:+.4f}"
        out["best_psnr_rank_desc"] = ""  # fill later
        out["last_psnr_for_reference"] = f"{last_psnr:.4f}"
        out_rows.append(out)

    # ranking by best_psnr desc
    ranked: List[Tuple[int, Dict[str, str]]] = sorted(
        enumerate(out_rows),
        key=lambda x: _to_float(x[1].get("best_psnr", "")),
        reverse=True,
    )
    for rank, (_, row) in enumerate(ranked, start=1):
        row["best_psnr_rank_desc"] = str(rank)

    # stable sort for readability: group, then rank
    out_rows.sort(
        key=lambda r: (
            0 if r.get("group") == "v2" else 1 if r.get("group") == "v3" else 2,
            int(r.get("best_psnr_rank_desc", "9999")),
        )
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(out_rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print("===== Dual-Delta Report Generated =====")
    print(f"input   : {input_path}")
    print(f"output  : {output_path}")
    print(f"anchor  : {args.anchor_name} ({args.anchor_psnr:.4f})")
    print(
        "baselines: "
        f"v2_nopost={baselines['v2']:.4f}, "
        f"v3_nopost={baselines['v3']:.4f}"
    )


if __name__ == "__main__":
    main()
