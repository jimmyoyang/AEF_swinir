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
ITER_RE = re.compile(r"Iter\s+(?P<iter>\d+)")


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
    best_from_marker = None
    best_psnr = None
    best_iter = None
    last = None
    cur_iter = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
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
                    "psnr": cur_psnr,
                    "ssim": cur_ssim,
                    "ergas": cur_ergas,
                    "sam": cur_sam,
                }
                if best_psnr is None or cur_psnr > best_psnr:
                    best_psnr = cur_psnr
                    best_iter = cur_iter
    if last is None:
        return None

    # Prefer max observed validation PSNR; keep compatibility with legacy marker if present.
    if best_psnr is None:
        best_psnr = last["psnr"]
    if best_from_marker is not None:
        best_psnr = max(best_psnr, best_from_marker)

    last["best_psnr"] = best_psnr
    last["best_iter"] = best_iter if best_iter is not None else ""
    return last


def infer_group(name):
    if name.startswith("v2_"):
        return "v2"
    if name.startswith("v3_"):
        return "v3"
    if "3b" in name:
        return "3b"
    return "other"


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
    out = {}
    for k, v in tail.items():
        if k in ["iter", "stage", ""]:
            continue
        if v is None or v == "":
            continue
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=str, default="training_logs/experiments_v2")
    parser.add_argument("--v3-root", type=str, default="training_logs/experiments_v3")
    parser.add_argument("--main-root", type=str, default="training_logs/experiments")
    parser.add_argument("--output", type=str, default="ablation_v3_vs_v2_comparison.csv")
    args = parser.parse_args()

    v2 = Path(args.v2_root)
    v3 = Path(args.v3_root)
    main_root = Path(args.main_root)

    schemes = [
        ("v2_5a_nopost_bestpre", v2 / "ablation_v2_5a_nopost_bestpre"),
        ("v2_5b_scheme1_gumbel", v2 / "ablation_v2_5b_scheme1_gumbel"),
        ("v2_5c_scheme3_dual_branch", v2 / "ablation_v2_5c_scheme3_dual_branch"),
        ("v2_5d_scheme6_matrix", v2 / "ablation_v2_5d_scheme6_matrix"),
        ("v3_5a_nopost_bestpre", v3 / "ablation_v3_5a_nopost_bestpre"),
        ("v3_5e_scheme1_plus6", v3 / "ablation_v3_5e_scheme1_plus6"),
        ("v3_5f_scheme3_plus6", v3 / "ablation_v3_5f_scheme3_plus6"),
        ("v3_5e_scheme1_plus6_stage", v3 / "ablation_v3_5e_scheme1_plus6_stage"),
        ("v3_5f_scheme3_plus6_stage", v3 / "ablation_v3_5f_scheme3_plus6_stage"),
        ("v3_5e_scheme1_plus6_stage_gated", v3 / "ablation_v3_5e_scheme1_plus6_stage_gated"),
        ("v3_5f_scheme3_plus6_stage_gated", v3 / "ablation_v3_5f_scheme3_plus6_stage_gated"),
        ("v3_5e_gated_lightweight", v3 / "ablation_v3_5e_scheme1_plus6_gated_lightweight"),
        ("v3_5f_gated_lightweight", v3 / "ablation_v3_5f_scheme3_plus6_gated_lightweight_full"),
        (
            "ablation_3b_with_cross_attention_posenc_learnable",
            main_root / "ablation_3b_with_cross_attention_posenc_learnable",
        ),
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
            "group": infer_group(name),
            "run_dir": str(run_dir),
            **metric,
            **post_stats,
        }
        results.append(row)

    if not results:
        print("No results found.")
        return

    # Add ranking and delta to 3b for easier direct comparison.
    ref_3b = next(
        (r for r in results if r["scheme"] == "ablation_3b_with_cross_attention_posenc_learnable"),
        None,
    )
    ref_best = ref_3b.get("best_psnr", 0.0) if ref_3b else None

    for r in results:
        if ref_best is None:
            r["delta_vs_3b"] = ""
        else:
            r["delta_vs_3b"] = r.get("best_psnr", 0.0) - ref_best

    results.sort(key=lambda x: x.get("best_psnr", 0.0), reverse=True)
    for idx, r in enumerate(results, start=1):
        r["rank"] = idx

    fields = sorted({k for r in results for k in r.keys()})
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {args.output}")
    print("Summary:")
    for r in results:
        print(
            f"- #{r.get('rank', '-'):<2} {r['scheme']}: "
            f"best_psnr={r.get('best_psnr', 0):.4f}, "
            f"best_iter={r.get('best_iter', '')}, "
            f"last_psnr={r.get('psnr', 0):.4f}, "
            f"last_ssim={r.get('ssim', 0):.4f}, "
            f"delta_vs_3b={r.get('delta_vs_3b', '') if isinstance(r.get('delta_vs_3b', ''), str) else f'{r.get('delta_vs_3b', 0):+.4f}'}"
        )


if __name__ == "__main__":
    main()
