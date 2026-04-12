#!/usr/bin/env python3
"""
Compare latest run results for selected experiments and generate:
1) Summary table (best/last metrics + delta vs baseline)
2) Full validation-round table (all rounds)
3) Markdown report for midterm update
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VAL_RE = re.compile(
    r"Validation Metrics\s*\|\s*PSNR:\s*(?P<psnr>[0-9.]+)\s*\|\s*SSIM:\s*(?P<ssim>[0-9.]+)\s*\|\s*ERGAS:\s*(?P<ergas>[0-9.]+)\s*\|\s*SAM:\s*(?P<sam>[0-9.]+)"
)
ITER_RE = re.compile(r"Iter\s+(?P<iter>\d+)")
SAVE_DIR_RE = re.compile(r"save_dir:\s*[\"'](?P<save_dir>[^\"']+)[\"']")


@dataclass
class ExpItem:
    config_path: str
    alias: str


def target_experiments() -> List[ExpItem]:
    return [
        ExpItem("configs/ablation/config_srcnn.yaml", "srcnn"),
        ExpItem("configs/ablation/config_true_baseline.yaml", "config_true_baseline"),
        ExpItem("configs/ablation/ablation_3b_with_cross_attention_posenc_learnable.yaml", "ablation_3b_with_cross_attention_posenc_learnable"),
        ExpItem("configs/ablation/ablation_4c_mask_reduce_prob_or.yaml", "ablation_4c_mask_reduce_prob_or"),
        ExpItem("configs/ablation/ablation_4f_mask_prob_or_learnable_pos_no_cross.yaml", "ablation_4f_mask_prob_or_learnable_pos_no_cross"),
        ExpItem("configs/ablation/ablation_4e_mask_prob_or_learnable_pos.yaml", "ablation_4e_mask_prob_or_learnable_pos"),
        ExpItem("configs/ablation/ablation_4d_mask_temporal_loss.yaml", "ablation_4d_mask_temporal_loss"),
    ]


def preferred_local_save_dirs() -> Dict[str, Path]:
    """Canonical local paths for this selected comparison task.

    These paths match the directories user checks via shell (training_logs/...).
    """
    return {
        "srcnn": (PROJECT_ROOT / "training_logs/srcnn_b2_3bands_run_1").resolve(),
        "config_true_baseline": (PROJECT_ROOT / "training_logs/experiments/config_true_baseline").resolve(),
        "ablation_3b_with_cross_attention_posenc_learnable": (PROJECT_ROOT / "training_logs/experiments/ablation_3b_with_cross_attention_posenc_learnable").resolve(),
        "ablation_4c_mask_reduce_prob_or": (PROJECT_ROOT / "training_logs/experiments/ablation_4c_mask_reduce_prob_or").resolve(),
        "ablation_4f_mask_prob_or_learnable_pos_no_cross": (PROJECT_ROOT / "training_logs/experiments/ablation_4f_mask_prob_or_learnable_pos_no_cross").resolve(),
        "ablation_4e_mask_prob_or_learnable_pos": (PROJECT_ROOT / "training_logs/experiments/ablation_4e_mask_prob_or_learnable_pos").resolve(),
        "ablation_4d_mask_temporal_loss": (PROJECT_ROOT / "training_logs/experiments/ablation_4d_mask_temporal_loss").resolve(),
    }


def parse_save_dir_from_config(config_file: Path) -> Optional[Path]:
    if not config_file.exists():
        return None
    text = config_file.read_text(encoding="utf-8", errors="ignore")
    m = SAVE_DIR_RE.search(text)
    if not m:
        return None
    return (PROJECT_ROOT / m.group("save_dir")).resolve()


def latest_run_dir(save_dir: Path) -> Optional[Path]:
    if save_dir is None or not save_dir.exists():
        return None
    run_dirs = [d for d in save_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def choose_best_source_dir(local_dir: Optional[Path], cfg_dir: Optional[Path]) -> tuple[Optional[Path], Optional[Path], str]:
    """Choose source directory by latest available run timestamp.

    Priority logic:
    - Compare latest runs from local_dir and cfg_dir when both exist.
    - Pick whichever has newer run mtime.
    - Return (chosen_save_dir, chosen_run_dir, source_label).
    """
    candidates: list[tuple[str, Path, Path]] = []

    if local_dir is not None:
        local_run = latest_run_dir(local_dir)
        if local_run is not None:
            candidates.append(("local_training_logs", local_dir, local_run))

    if cfg_dir is not None:
        cfg_run = latest_run_dir(cfg_dir)
        if cfg_run is not None:
            candidates.append(("config_save_dir", cfg_dir, cfg_run))

    if not candidates:
        if local_dir is not None:
            return local_dir, None, "local_training_logs"
        if cfg_dir is not None:
            return cfg_dir, None, "config_save_dir"
        return None, None, "none"

    source, save_dir, run_dir = max(candidates, key=lambda x: x[2].stat().st_mtime)
    return save_dir, run_dir, source


def parse_validation_rounds(log_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not log_path.exists():
        return rows

    cur_iter: Optional[int] = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_iter = ITER_RE.search(line)
            if m_iter:
                cur_iter = int(m_iter.group("iter"))

            m_val = VAL_RE.search(line)
            if not m_val:
                continue

            rows.append(
                {
                    "iter": cur_iter if cur_iter is not None else -1,
                    "psnr": float(m_val.group("psnr")),
                    "ssim": float(m_val.group("ssim")),
                    "ergas": float(m_val.group("ergas")),
                    "sam": float(m_val.group("sam")),
                }
            )
    return rows


def summarize_rounds(rounds: List[Dict[str, float]]) -> Dict[str, float]:
    if not rounds:
        return {
            "val_rounds": 0,
            "best_iter": -1,
            "best_psnr": float("nan"),
            "best_ssim": float("nan"),
            "best_ergas": float("nan"),
            "best_sam": float("nan"),
            "last_iter": -1,
            "last_psnr": float("nan"),
            "last_ssim": float("nan"),
            "last_ergas": float("nan"),
            "last_sam": float("nan"),
        }

    best = max(rounds, key=lambda x: x["psnr"])
    last = rounds[-1]
    return {
        "val_rounds": len(rounds),
        "best_iter": int(best["iter"]),
        "best_psnr": float(best["psnr"]),
        "best_ssim": float(best["ssim"]),
        "best_ergas": float(best["ergas"]),
        "best_sam": float(best["sam"]),
        "last_iter": int(last["iter"]),
        "last_psnr": float(last["psnr"]),
        "last_ssim": float(last["ssim"]),
        "last_ergas": float(last["ergas"]),
        "last_sam": float(last["sam"]),
    }


def fmt_num(v: float, digits: int = 4) -> str:
    if v != v:
        return "N/A"
    return f"{v:.{digits}f}"


def to_display_path(p: Path) -> str:
    """Return project-relative path when possible, otherwise absolute path."""
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(p.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare latest selected experiment results")
    parser.add_argument("--output-dir", default="debug_output/latest_selected_compare")
    parser.add_argument("--report-md", default="debug_output/latest_selected_compare/midterm_selected_compare.md")
    parser.add_argument(
        "--prefer-local-training-logs",
        action="store_true",
        default=True,
        help="Kept for compatibility; script now auto-selects latest between local and config paths.",
    )
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary_latest_selected.csv"
    rounds_csv = out_dir / "all_validation_rounds_latest_selected.csv"
    report_md = (PROJECT_ROOT / args.report_md).resolve()
    report_md.parent.mkdir(parents=True, exist_ok=True)

    exp_rows: List[Dict[str, object]] = []
    full_round_rows: List[Dict[str, object]] = []
    local_dir_map = preferred_local_save_dirs()

    for item in target_experiments():
        cfg_path = (PROJECT_ROOT / item.config_path).resolve()
        cfg_save_dir = parse_save_dir_from_config(cfg_path)

        local_save_dir = local_dir_map.get(item.alias)
        save_dir, run_dir, source = choose_best_source_dir(local_save_dir, cfg_save_dir)

        if run_dir is None:
            exp_rows.append(
                {
                    "alias": item.alias,
                    "config_path": item.config_path,
                    "save_dir": str(save_dir) if save_dir else "",
                    "latest_run_dir": "",
                    "log_path": "",
                    "source": source,
                    "status": "missing_run_dir",
                    "val_rounds": 0,
                    "best_iter": "",
                    "best_psnr": "",
                    "best_ssim": "",
                    "best_ergas": "",
                    "best_sam": "",
                    "last_iter": "",
                    "last_psnr": "",
                    "last_ssim": "",
                    "last_ergas": "",
                    "last_sam": "",
                    "delta_best_psnr_vs_baseline": "",
                }
            )
            continue

        log_path = run_dir / "training.log"
        rounds = parse_validation_rounds(log_path)
        summ = summarize_rounds(rounds)

        for idx, r in enumerate(rounds, start=1):
            full_round_rows.append(
                {
                    "alias": item.alias,
                    "config_path": item.config_path,
                    "latest_run_dir": to_display_path(run_dir),
                    "val_round_index": idx,
                    "iter": r["iter"],
                    "psnr": r["psnr"],
                    "ssim": r["ssim"],
                    "ergas": r["ergas"],
                    "sam": r["sam"],
                }
            )

        exp_rows.append(
            {
                "alias": item.alias,
                "config_path": item.config_path,
                "save_dir": to_display_path(save_dir) if save_dir else "",
                "latest_run_dir": to_display_path(run_dir),
                "log_path": to_display_path(log_path),
                "source": source,
                "status": "ok" if rounds else "missing_or_invalid_log",
                **summ,
                "delta_best_psnr_vs_baseline": "",
            }
        )

    baseline_row = next((r for r in exp_rows if r["alias"] == "config_true_baseline" and r["status"] == "ok"), None)
    baseline_psnr = float(baseline_row["best_psnr"]) if baseline_row else float("nan")

    for r in exp_rows:
        if r["status"] != "ok" or baseline_psnr != baseline_psnr:
            r["delta_best_psnr_vs_baseline"] = ""
            continue
        r["delta_best_psnr_vs_baseline"] = float(r["best_psnr"]) - baseline_psnr

    # CSV: summary
    summary_fields = [
        "alias",
        "config_path",
        "save_dir",
        "latest_run_dir",
        "log_path",
        "source",
        "status",
        "val_rounds",
        "best_iter",
        "best_psnr",
        "best_ssim",
        "best_ergas",
        "best_sam",
        "last_iter",
        "last_psnr",
        "last_ssim",
        "last_ergas",
        "last_sam",
        "delta_best_psnr_vs_baseline",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(exp_rows)

    # CSV: full rounds
    round_fields = [
        "alias",
        "config_path",
        "latest_run_dir",
        "val_round_index",
        "iter",
        "psnr",
        "ssim",
        "ergas",
        "sam",
        "delta_psnr_vs_baseline_best",
    ]
    with rounds_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=round_fields)
        w.writeheader()
        for rr in full_round_rows:
            out = dict(rr)
            if baseline_psnr == baseline_psnr:
                out["delta_psnr_vs_baseline_best"] = float(rr["psnr"]) - baseline_psnr
            else:
                out["delta_psnr_vs_baseline_best"] = ""
            w.writerow(out)

    # markdown report
    lines: List[str] = []
    lines.append("# Midterm Latest Comparison (Selected 7 Configs)")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    for e in target_experiments():
        lines.append(f"- {e.config_path}")
    lines.append("")
    lines.append("## Storage Path Check")
    lines.append("")
    lines.append("- SRCNN save_dir is independent: training_logs/srcnn_b2_3bands_run_1")
    lines.append("- Other six experiments save under: training_logs/experiments/<exp_name>")
    lines.append("")

    lines.append("## Summary Table (Latest Run + Best/Last Metrics)")
    lines.append("")
    lines.append("| alias | source | val_rounds | best_iter | best_psnr | best_ssim | best_ergas | best_sam | delta_best_psnr_vs_baseline | latest_run_dir |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

    ok_rows = [r for r in exp_rows if r["status"] == "ok"]
    ok_rows.sort(key=lambda r: float(r["best_psnr"]) if r["best_psnr"] == r["best_psnr"] else -1.0, reverse=True)

    for r in ok_rows:
        delta = r["delta_best_psnr_vs_baseline"]
        delta_txt = "N/A" if delta == "" else f"{float(delta):+.4f}"
        lines.append(
            "| "
            + f"{r['alias']} | {r['source']} | {r['val_rounds']} | {r['best_iter']} | {fmt_num(float(r['best_psnr']))} | {fmt_num(float(r['best_ssim']))} | {fmt_num(float(r['best_ergas']))} | {fmt_num(float(r['best_sam']))} | {delta_txt} | {r['latest_run_dir']} |"
        )

    bad_rows = [r for r in exp_rows if r["status"] != "ok"]
    if bad_rows:
        lines.append("")
        lines.append("## Missing/Invalid")
        lines.append("")
        for r in bad_rows:
            lines.append(f"- {r['alias']}: status={r['status']}")

    if ok_rows:
        best = ok_rows[0]
        lines.append("")
        lines.append("## Conclusion For Midterm")
        lines.append("")
        lines.append(
            f"- Current best among selected latest runs: {best['alias']} (best_psnr={fmt_num(float(best['best_psnr']))}, best_ssim={fmt_num(float(best['best_ssim']))})."
        )
        if baseline_psnr == baseline_psnr:
            delta_best = float(best["best_psnr"]) - baseline_psnr
            lines.append(
                f"- Compared with config_true_baseline (best_psnr={baseline_psnr:.4f}), delta_best_psnr={delta_best:+.4f}."
            )

    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- Summary CSV: {to_display_path(summary_csv)}")
    lines.append(f"- All-rounds CSV: {to_display_path(rounds_csv)}")
    lines.append(f"- This report: {to_display_path(report_md)}")
    lines.append("")

    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved rounds CSV: {rounds_csv}")
    print(f"Saved markdown   : {report_md}")


if __name__ == "__main__":
    main()
