#!/usr/bin/env python3
"""Generate strict 7-config figures for the midterm report.

Inputs:
- debug_output/latest_selected_compare/summary_latest_selected.csv

Outputs (same output dir by default):
- selected7_best_psnr_bar.png
- selected7_best_last_psnr_grouped.png
- selected7_best_metrics_panel.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_ORDER = [
    "srcnn",
    "config_true_baseline",
    "ablation_3b_with_cross_attention_posenc_learnable",
    "ablation_4c_mask_reduce_prob_or",
    "ablation_4f_mask_prob_or_learnable_pos_no_cross",
    "ablation_4e_mask_prob_or_learnable_pos",
    "ablation_4d_mask_temporal_loss",
]

ALIAS_SHORT = {
    "srcnn": "SRCNN",
    "config_true_baseline": "Baseline",
    "ablation_3b_with_cross_attention_posenc_learnable": "Abl-3b",
    "ablation_4c_mask_reduce_prob_or": "Abl-4c",
    "ablation_4f_mask_prob_or_learnable_pos_no_cross": "Abl-4f",
    "ablation_4e_mask_prob_or_learnable_pos": "Abl-4e",
    "ablation_4d_mask_temporal_loss": "Abl-4d",
}


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_alias = {r["alias"]: r for r in rows}
    missing = [a for a in TARGET_ORDER if a not in by_alias]
    if missing:
        raise ValueError(f"Missing aliases in summary csv: {missing}")
    return [by_alias[a] for a in TARGET_ORDER]


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_best_psnr(rows: list[dict[str, str]], out_path: Path) -> None:
    labels = [ALIAS_SHORT[r["alias"]] for r in rows]
    values = [f(r, "best_psnr") for r in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    bars = ax.bar(x, values, color=["#3d8bfd", "#6c757d", "#20c997", "#0ca678", "#2b8a3e", "#12b886", "#15aabf"])
    ax.set_title("Selected 7 Configs: Best PSNR (Latest Run)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_best_last_psnr(rows: list[dict[str, str]], out_path: Path) -> None:
    labels = [ALIAS_SHORT[r["alias"]] for r in rows]
    best_vals = [f(r, "best_psnr") for r in rows]
    last_vals = [f(r, "last_psnr") for r in rows]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    b1 = ax.bar(x - w / 2, best_vals, width=w, label="best_psnr", color="#228be6")
    b2 = ax.bar(x + w / 2, last_vals, width=w, label="last_psnr", color="#74c0fc")
    ax.set_title("Selected 7 Configs: Best vs Last PSNR")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for b, v in zip(b1, best_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for b, v in zip(b2, last_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_best_metrics_panel(rows: list[dict[str, str]], out_path: Path) -> None:
    labels = [ALIAS_SHORT[r["alias"]] for r in rows]
    x = np.arange(len(labels))

    best_psnr = np.array([f(r, "best_psnr") for r in rows])
    best_ssim = np.array([f(r, "best_ssim") for r in rows])
    best_ergas = np.array([f(r, "best_ergas") for r in rows])
    best_sam = np.array([f(r, "best_sam") for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    panels = [
        (axes[0, 0], best_psnr, "Best PSNR (higher is better)", "#1971c2"),
        (axes[0, 1], best_ssim, "Best SSIM (higher is better)", "#2f9e44"),
        (axes[1, 0], best_ergas, "Best ERGAS (lower is better)", "#e67700"),
        (axes[1, 1], best_sam, "Best SAM (lower is better)", "#c2255c"),
    ]

    for ax, vals, title, color in panels:
        bars = ax.bar(x, vals, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Selected 7 Configs: Best Metrics Panel", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot strict selected-7 experiment figures")
    parser.add_argument(
        "--summary-csv",
        default="debug_output/latest_selected_compare/summary_latest_selected.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="debug_output/latest_selected_compare",
    )
    args = parser.parse_args()

    summary_csv = (PROJECT_ROOT / args.summary_csv).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(summary_csv)
    plot_best_psnr(rows, out_dir / "selected7_best_psnr_bar.png")
    plot_best_last_psnr(rows, out_dir / "selected7_best_last_psnr_grouped.png")
    plot_best_metrics_panel(rows, out_dir / "selected7_best_metrics_panel.png")

    print(f"Saved: {out_dir / 'selected7_best_psnr_bar.png'}")
    print(f"Saved: {out_dir / 'selected7_best_last_psnr_grouped.png'}")
    print(f"Saved: {out_dir / 'selected7_best_metrics_panel.png'}")


if __name__ == "__main__":
    main()
