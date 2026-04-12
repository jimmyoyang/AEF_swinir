#!/usr/bin/env python3
"""
Build a unified dual-delta report across four suites:
- legacy_v2_full
- legacy_v3_all_full
- mainline_4c_all_full
- mainline_4d_all_full

Dual-delta metrics:
1) delta_best_psnr_vs_group_5a: module incremental gain within same suite+track(v2/v3)
2) delta_best_psnr_vs_mainline_4c / _4d: gap to mainline anchor ceilings
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VAL_RE = re.compile(
    r"Validation Metrics \| PSNR: (?P<psnr>[0-9.]+) \| SSIM: (?P<ssim>[0-9.]+) \| ERGAS: (?P<ergas>[0-9.]+) \| SAM: (?P<sam>[0-9.]+)"
)
BEST_RE = re.compile(r"New best metric: (?P<best>[0-9.]+)")
ITER_RE = re.compile(r"Iter\s+(?P<iter>\d+)")


@dataclass
class ExpSpec:
    suite: str
    track: str
    exp_name: str
    root_dir: str


def expected_specs() -> List[ExpSpec]:
    specs: List[ExpSpec] = []

    legacy_v2 = [
        "ablation_v2_5a_nopost_bestpre",
        "ablation_v2_5b_scheme1_gumbel",
        "ablation_v2_5c_scheme3_dual_branch",
        "ablation_v2_5d_scheme6_matrix",
    ]
    for name in legacy_v2:
        specs.append(ExpSpec("legacy_v2_full", "v2", name, "training_logs/experiments_v2"))

    legacy_v3 = [
        "ablation_v3_5a_nopost_bestpre",
        "ablation_v3_5e_scheme1_plus6",
        "ablation_v3_5f_scheme3_plus6",
        "ablation_v3_5e_scheme1_plus6_stage",
        "ablation_v3_5f_scheme3_plus6_stage",
        "ablation_v3_5e_scheme1_plus6_gated_lightweight_full",
        "ablation_v3_5f_scheme3_plus6_gated_lightweight_full",
    ]
    for name in legacy_v3:
        specs.append(ExpSpec("legacy_v3_all_full", "v3", name, "training_logs/experiments_v3"))

    mainline_4c_v2 = [
        "ablation_4c_v2_5a_nopost_bestpre_full",
        "ablation_4c_v2_5b_scheme1_gumbel_full",
        "ablation_4c_v2_5c_scheme3_dual_branch_full",
        "ablation_4c_v2_5d_scheme6_matrix_full",
    ]
    for name in mainline_4c_v2:
        specs.append(ExpSpec("mainline_4c_all_full", "v2", name, "training_logs/experiments_mainline_4c_v2"))

    mainline_4c_v3 = [
        "ablation_4c_v3_5a_nopost_bestpre_full",
        "ablation_4c_v3_5e_scheme1_plus6_full",
        "ablation_4c_v3_5f_scheme3_plus6_full",
        "ablation_4c_v3_5e_scheme1_plus6_stage_full",
        "ablation_4c_v3_5f_scheme3_plus6_stage_full",
        "ablation_4c_v3_5e_scheme1_plus6_gated_lightweight_full",
        "ablation_4c_v3_5f_scheme3_plus6_gated_lightweight_full",
        "ablation_4c_v3_5e_scheme1_plus6_stage_gated_full",
        "ablation_4c_v3_5f_scheme3_plus6_stage_gated_full",
    ]
    for name in mainline_4c_v3:
        specs.append(ExpSpec("mainline_4c_all_full", "v3", name, "training_logs/experiments_mainline_4c_v3"))

    mainline_4d_v2 = [
        "ablation_4d_v2_5a_nopost_bestpre_full",
        "ablation_4d_v2_5b_scheme1_gumbel_full",
        "ablation_4d_v2_5c_scheme3_dual_branch_full",
        "ablation_4d_v2_5d_scheme6_matrix_full",
    ]
    for name in mainline_4d_v2:
        specs.append(ExpSpec("mainline_4d_all_full", "v2", name, "training_logs/experiments_mainline_4d_v2"))

    mainline_4d_v3 = [
        "ablation_4d_v3_5a_nopost_bestpre_full",
        "ablation_4d_v3_5e_scheme1_plus6_full",
        "ablation_4d_v3_5f_scheme3_plus6_full",
        "ablation_4d_v3_5e_scheme1_plus6_stage_full",
        "ablation_4d_v3_5f_scheme3_plus6_stage_full",
        "ablation_4d_v3_5e_scheme1_plus6_gated_lightweight_full",
        "ablation_4d_v3_5f_scheme3_plus6_gated_lightweight_full",
        "ablation_4d_v3_5e_scheme1_plus6_stage_gated_full",
        "ablation_4d_v3_5f_scheme3_plus6_stage_gated_full",
    ]
    for name in mainline_4d_v3:
        specs.append(ExpSpec("mainline_4d_all_full", "v3", name, "training_logs/experiments_mainline_4d_v3"))

    return specs


def latest_run_dir(exp_dir: Path) -> Optional[Path]:
    if not exp_dir.exists():
        return None
    dirs = [x for x in exp_dir.iterdir() if x.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda x: x.stat().st_mtime)


def parse_training_log(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None

    best_from_marker = None
    best_psnr = None
    best_iter = None
    last = None
    cur_iter = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            it = ITER_RE.search(line)
            if it:
                cur_iter = int(it.group("iter"))

            m = BEST_RE.search(line)
            if m:
                best_from_marker = float(m.group("best"))

            v = VAL_RE.search(line)
            if v:
                cur_psnr = float(v.group("psnr"))
                cur_ssim = float(v.group("ssim"))
                cur_ergas = float(v.group("ergas"))
                cur_sam = float(v.group("sam"))

                last = {
                    "last_psnr": cur_psnr,
                    "last_ssim": cur_ssim,
                    "last_ergas": cur_ergas,
                    "last_sam": cur_sam,
                }
                if best_psnr is None or cur_psnr > best_psnr:
                    best_psnr = cur_psnr
                    best_iter = cur_iter

    if last is None:
        return None

    if best_psnr is None:
        best_psnr = last["last_psnr"]
    if best_from_marker is not None:
        best_psnr = max(best_psnr, best_from_marker)

    return {
        **last,
        "best_psnr": best_psnr,
        "best_iter": best_iter if best_iter is not None else "",
    }


def infer_display_name(exp_name: str) -> str:
    return exp_name


def find_group_baselines(rows: List[Dict[str, object]]) -> Dict[Tuple[str, str], float]:
    baselines: Dict[Tuple[str, str], float] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        exp_name = str(r["exp_name"])
        if "5a_nopost_bestpre" in exp_name:
            key = (str(r["suite"]), str(r["track"]))
            baselines[key] = float(r["best_psnr"])
    return baselines


def write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "suite",
        "track",
        "exp_name",
        "display_name",
        "status",
        "run_dir",
        "log_path",
        "best_psnr",
        "best_iter",
        "last_psnr",
        "last_ssim",
        "last_ergas",
        "last_sam",
        "group_5a_baseline_psnr",
        "delta_best_psnr_vs_group_5a",
        "delta_best_psnr_vs_mainline_4c",
        "delta_best_psnr_vs_mainline_4d",
        "global_rank_by_best_psnr",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, object]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ok_rows.sort(key=lambda x: float(x["best_psnr"]), reverse=True)

    lines = []
    lines.append("# Unified Dual-Delta Report")
    lines.append("")
    lines.append("| rank | suite | track | exp_name | best_psnr | vs group 5a | vs 4c | vs 4d |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|")
    for idx, r in enumerate(ok_rows, start=1):
        lines.append(
            f"| {idx} | {r['suite']} | {r['track']} | {r['exp_name']} | {float(r['best_psnr']):.4f} | "
            f"{float(r['delta_best_psnr_vs_group_5a']):+.4f} | "
            f"{float(r['delta_best_psnr_vs_mainline_4c']):+.4f} | "
            f"{float(r['delta_best_psnr_vs_mainline_4d']):+.4f} |"
        )

    missing = [r for r in rows if r.get("status") != "ok"]
    if missing:
        lines.append("")
        lines.append("## Missing Runs")
        lines.append("")
        for r in missing:
            lines.append(f"- {r['suite']} | {r['track']} | {r['exp_name']} | status={r['status']}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified dual-delta report across four suites")
    parser.add_argument("--project-root", default=".", help="Project root path")
    parser.add_argument("--anchor-4c", type=float, default=14.7357)
    parser.add_argument("--anchor-4d", type=float, default=14.7634)
    parser.add_argument("--output-csv", default="debug_output/unified_dual_delta_report.csv")
    parser.add_argument("--output-md", default="debug_output/unified_dual_delta_report.md")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    rows: List[Dict[str, object]] = []

    for spec in expected_specs():
        exp_dir = root / spec.root_dir / spec.exp_name
        run_dir = latest_run_dir(exp_dir)

        if run_dir is None:
            rows.append(
                {
                    "suite": spec.suite,
                    "track": spec.track,
                    "exp_name": spec.exp_name,
                    "display_name": infer_display_name(spec.exp_name),
                    "status": "missing_run_dir",
                    "run_dir": "",
                    "log_path": "",
                    "best_psnr": "",
                    "best_iter": "",
                    "last_psnr": "",
                    "last_ssim": "",
                    "last_ergas": "",
                    "last_sam": "",
                    "group_5a_baseline_psnr": "",
                    "delta_best_psnr_vs_group_5a": "",
                    "delta_best_psnr_vs_mainline_4c": "",
                    "delta_best_psnr_vs_mainline_4d": "",
                    "global_rank_by_best_psnr": "",
                }
            )
            continue

        log_path = run_dir / "training.log"
        metric = parse_training_log(log_path)
        if metric is None:
            rows.append(
                {
                    "suite": spec.suite,
                    "track": spec.track,
                    "exp_name": spec.exp_name,
                    "display_name": infer_display_name(spec.exp_name),
                    "status": "missing_or_invalid_log",
                    "run_dir": str(run_dir),
                    "log_path": str(log_path),
                    "best_psnr": "",
                    "best_iter": "",
                    "last_psnr": "",
                    "last_ssim": "",
                    "last_ergas": "",
                    "last_sam": "",
                    "group_5a_baseline_psnr": "",
                    "delta_best_psnr_vs_group_5a": "",
                    "delta_best_psnr_vs_mainline_4c": "",
                    "delta_best_psnr_vs_mainline_4d": "",
                    "global_rank_by_best_psnr": "",
                }
            )
            continue

        rows.append(
            {
                "suite": spec.suite,
                "track": spec.track,
                "exp_name": spec.exp_name,
                "display_name": infer_display_name(spec.exp_name),
                "status": "ok",
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "best_psnr": metric["best_psnr"],
                "best_iter": metric["best_iter"],
                "last_psnr": metric["last_psnr"],
                "last_ssim": metric["last_ssim"],
                "last_ergas": metric["last_ergas"],
                "last_sam": metric["last_sam"],
                "group_5a_baseline_psnr": "",
                "delta_best_psnr_vs_group_5a": "",
                "delta_best_psnr_vs_mainline_4c": "",
                "delta_best_psnr_vs_mainline_4d": "",
                "global_rank_by_best_psnr": "",
            }
        )

    baselines = find_group_baselines(rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ok_rows.sort(key=lambda x: float(x["best_psnr"]), reverse=True)
    rank_map = {id(r): i for i, r in enumerate(ok_rows, start=1)}

    for r in rows:
        if r.get("status") != "ok":
            continue
        best = float(r["best_psnr"])
        key = (str(r["suite"]), str(r["track"]))
        base_psnr = baselines.get(key)

        if base_psnr is None:
            r["group_5a_baseline_psnr"] = ""
            r["delta_best_psnr_vs_group_5a"] = ""
        else:
            r["group_5a_baseline_psnr"] = base_psnr
            r["delta_best_psnr_vs_group_5a"] = best - base_psnr

        r["delta_best_psnr_vs_mainline_4c"] = best - float(args.anchor_4c)
        r["delta_best_psnr_vs_mainline_4d"] = best - float(args.anchor_4d)
        r["global_rank_by_best_psnr"] = rank_map[id(r)]

    out_csv = root / args.output_csv
    out_md = root / args.output_md
    write_csv(rows, out_csv)
    write_markdown(rows, out_md)

    total = len(rows)
    ok = len([r for r in rows if r.get("status") == "ok"])
    print(f"Saved CSV: {out_csv}")
    print(f"Saved MD : {out_md}")
    print(f"Summary  : total={total}, ok={ok}, missing={total-ok}")


if __name__ == "__main__":
    main()
