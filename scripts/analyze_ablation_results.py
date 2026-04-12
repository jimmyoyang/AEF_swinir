#!/usr/bin/env python3
"""
scripts/analyze_ablation_results.py
=====================================
消融实验结果自动分析脚本。

功能：
  - 扫描 training_logs/experiments/ 中所有实验的 training.log
  - 提取每个实验的最佳 PSNR / SSIM / ERGAS / SAM 及对应迭代
  - 输出 ablation_summary_latest.csv（机器可读）
  - 输出 ablation_best_results/ablation_analysis_report.md（人类可读报告）
  - 可选：生成 PSNR 增益条形图

用法：
  python scripts/analyze_ablation_results.py
  python scripts/analyze_ablation_results.py --plot --baseline config_true_baseline
  python scripts/analyze_ablation_results.py --exp_dir ./training_logs/experiments
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

# 可选绘图（不强制依赖）
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TRAIN_LOGS_DIR = PROJECT_ROOT / "training_logs" / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "ablation_best_results"
SUMMARY_CSV = PROJECT_ROOT / "ablation_summary_latest.csv"

# 日志解析正则
_RE_VAL = re.compile(
    r"Validation Metrics.*PSNR:\s*([\d.]+).*SSIM:\s*([\d.]+).*ERGAS:\s*([\d.]+).*SAM:\s*([\d.]+)"
)
_RE_ITER = re.compile(r"Iter\s+(\d+)")
_RE_BEST = re.compile(r"New best metric:\s*([\d.]+)")

# 实验配置组信息（用于报告分组）
_GROUP_MAP = {
    "config_true_baseline": "G1-基线",
    "ablation_1b": "G1-基线",
    "ablation_1c": "G1-基线",
    "ablation_1d": "G1-基线",
    "ablation_2b": "G2-交叉注意力",
    "ablation_2c": "G2-交叉注意力",
    "ablation_3a": "G3-位置编码",
    "ablation_3b": "G3-位置编码",
    "ablation_3c": "G3-位置编码",
    "ablation_3d": "G3-位置编码",
    "ablation_4a": "G4-软掩膜",
    "ablation_4b": "G4-软掩膜",
    "ablation_4c": "G4-软掩膜",
    "ablation_4d": "G4-软掩膜",
    "ablation_5a": "G5-光谱后处理",
    "ablation_5b": "G5-光谱后处理",
    "ablation_5c": "G5-光谱后处理",
    "ablation_5d": "G5-光谱后处理",
}


def _get_group(exp_name: str) -> str:
    for prefix, group in _GROUP_MAP.items():
        if exp_name.startswith(prefix):
            return group
    return "其他"


def _rank_key(r: dict) -> tuple[float, float]:
    """Ranking key: prioritize PSNR, then SSIM as tie-breaker."""
    return (r.get("best_psnr", -1.0), r.get("best_ssim", -1.0))


def build_4c_4d_conclusion(results: list[dict]) -> Optional[str]:
    """自动生成 4c vs 4d 的单变量结论文本。"""
    r4c = next((r for r in results if r["exp_name"] == "ablation_4c_mask_reduce_prob_or"), None)
    r4d = next((r for r in results if r["exp_name"] == "ablation_4d_mask_temporal_loss"), None)
    if r4c is None or r4d is None:
        return None

    d_psnr = r4d["best_psnr"] - r4c["best_psnr"]
    d_ssim = r4d["best_ssim"] - r4c["best_ssim"]

    abs_d = abs(d_psnr)
    if abs_d < 0.03:
        strength = "无明显差异"
    elif abs_d < 0.10:
        strength = "轻微差异"
    else:
        strength = "明确差异"

    if d_psnr > 0:
        trend = "4d 优于 4c"
    elif d_psnr < 0:
        trend = "4d 劣于 4c"
    else:
        if d_ssim > 0.002:
            trend = "4d 与 4c 在 PSNR 持平，但 4d 在 SSIM 略优"
        elif d_ssim < -0.002:
            trend = "4d 与 4c 在 PSNR 持平，但 4d 在 SSIM 略低"
        else:
            trend = "4d 与 4c 持平"

    return (
        f"4c vs 4d（单变量：mask 聚合策略→时相级损失）: {trend} | "
        f"ΔPSNR={d_psnr:+.4f} dB, ΔSSIM={d_ssim:+.4f} | 判定: {strength}"
    )


# ---------------------------------------------------------------------------
# 解析单个实验训练日志
# ---------------------------------------------------------------------------
def parse_training_log(log_path: Path) -> dict:
    """
    返回 {
        best_psnr, best_ssim, best_ergas, best_sam,
        best_iter, total_iters, val_curve: [(iter, psnr), ...]
    }
    """
    best_psnr = -1.0
    best_ssim = -1.0
    best_ergas = float("inf")
    best_sam = float("inf")
    best_iter = 0
    total_iters = 0
    val_curve: list[tuple[int, float]] = []
    current_iter = 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m_iter = _RE_ITER.search(line)
            if m_iter:
                current_iter = int(m_iter.group(1))
                total_iters = max(total_iters, current_iter)

            m_val = _RE_VAL.search(line)
            if m_val:
                psnr = float(m_val.group(1))
                ssim = float(m_val.group(2))
                ergas = float(m_val.group(3))
                s_am = float(m_val.group(4))
                val_curve.append((current_iter, psnr))
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_ssim = ssim
                    best_ergas = ergas
                    best_sam = s_am

            if _RE_BEST.search(line):
                best_iter = current_iter

    return {
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "best_ergas": best_ergas,
        "best_sam": best_sam,
        "best_iter": best_iter,
        "total_iters": total_iters,
        "val_curve": val_curve,
    }


# ---------------------------------------------------------------------------
# 收集所有实验
# ---------------------------------------------------------------------------
def collect_all(exp_dir: Path) -> list[dict]:
    results = []
    for exp_name_dir in sorted(exp_dir.iterdir()):
        if not exp_name_dir.is_dir():
            continue
        exp_name = exp_name_dir.name

        # 找最新的 run（多次训练时取最新）
        run_dirs = sorted(
            [d for d in exp_name_dir.iterdir() if d.is_dir() and (d / "training.log").exists()],
            key=lambda d: d.name,
            reverse=True,
        )
        if not run_dirs:
            continue

        run_dir = run_dirs[0]
        log_path = run_dir / "training.log"
        ckpt_dir = run_dir / "ckpts"
        best_ckpt = ckpt_dir / "model_best.pth" if ckpt_dir.exists() else None

        metrics = parse_training_log(log_path)
        results.append(
            {
                "exp_name": exp_name,
                "group": _get_group(exp_name),
                "run_dir": run_dir,
                "log_path": log_path,
                "best_ckpt": best_ckpt if (best_ckpt and best_ckpt.exists()) else None,
                **metrics,
            }
        )

    return results


# ---------------------------------------------------------------------------
# 保存 CSV
# ---------------------------------------------------------------------------
def save_csv(results: list[dict], out_path: Path):
    fieldnames = [
        "rank", "exp_name", "group", "best_psnr", "best_ssim",
        "best_ergas", "best_sam", "best_iter", "total_iters",
        "has_best_ckpt", "run_dir",
    ]
    sorted_results = sorted(results, key=_rank_key, reverse=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, r in enumerate(sorted_results, 1):
            row = dict(r)
            row["rank"] = rank
            row["has_best_ckpt"] = "Yes" if r["best_ckpt"] else "No"
            row["run_dir"] = str(r["run_dir"].relative_to(PROJECT_ROOT))
            row["best_psnr"] = f"{r['best_psnr']:.4f}" if r["best_psnr"] >= 0 else "N/A"
            row["best_ssim"] = f"{r['best_ssim']:.4f}" if r["best_ssim"] >= 0 else "N/A"
            row["best_ergas"] = f"{r['best_ergas']:.4f}" if r["best_ergas"] != float('inf') else "N/A"
            row["best_sam"] = f"{r['best_sam']:.4f}" if r["best_sam"] != float('inf') else "N/A"
            writer.writerow(row)
    print(f"✅ CSV 已保存: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# 生成 Markdown 报告
# ---------------------------------------------------------------------------
def save_markdown(results: list[dict], out_path: Path, baseline_name: Optional[str]):
    from datetime import datetime

    sorted_results = sorted(results, key=_rank_key, reverse=True)
    baseline = next(
        (r for r in results if r["exp_name"] == baseline_name), None
    ) if baseline_name else None
    baseline_psnr = baseline["best_psnr"] if baseline else 0.0

    lines = [
        f"# 消融实验分析报告",
        f"",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**实验总数**: {len(results)}  ",
        f"**基准实验**: `{baseline_name or '未设置'}`  "
        + (f"(PSNR={baseline_psnr:.4f} dB)" if baseline else ""),
        f"",
        f"---",
        f"",
        f"## 排行榜（按 Best PSNR）",
        f"",
        f"| Rank | 实验名称 | Group | PSNR (dB) | SSIM | Best Iter | vs Baseline |",
        f"|------|---------|-------|-----------|------|-----------|-------------|",
    ]

    medal = {1: "🥇", 2: "🥈", 3: "🥉"}
    for rank, r in enumerate(sorted_results, 1):
        psnr = r["best_psnr"]
        ssim = r["best_ssim"]
        psnr_str = f"{psnr:.4f}" if psnr >= 0 else "N/A"
        ssim_str = f"{ssim:.4f}" if ssim >= 0 else "N/A"
        delta = f"{psnr - baseline_psnr:+.4f} dB" if baseline and psnr >= 0 else "-"
        m = medal.get(rank, f"{rank}")
        lines.append(
            f"| {m} | `{r['exp_name']}` | {r['group']} "
            f"| **{psnr_str}** | {ssim_str} | {r['best_iter']} | {delta} |"
        )

    # 分组汇总
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)

    lines += ["", "---", "", "## 分组详细结果", ""]
    for group_name in sorted(groups):
        lines += [f"### {group_name}", ""]
        lines += [
            "| 实验名称 | PSNR | SSIM | ERGAS | SAM | Best Iter | vs Baseline |",
            "|---------|------|------|-------|-----|-----------|-------------|",
        ]
        for r in sorted(groups[group_name], key=_rank_key, reverse=True):
            psnr = r["best_psnr"]
            delta = f"{psnr - baseline_psnr:+.4f}" if baseline and psnr >= 0 else "-"
            ergas = f"{r['best_ergas']:.2f}" if r["best_ergas"] != float("inf") else "N/A"
            s_am = f"{r['best_sam']:.4f}" if r["best_sam"] != float("inf") else "N/A"
            lines.append(
                f"| `{r['exp_name']}` | {psnr:.4f} | {r['best_ssim']:.4f} "
                f"| {ergas} | {s_am} | {r['best_iter']} | {delta} |"
            )
        lines.append("")

    conclusion_4c_4d = build_4c_4d_conclusion(results)
    lines += ["---", "", "## 4c vs 4d 单变量结论", ""]
    if conclusion_4c_4d is None:
        lines.append("- 未同时检测到 `ablation_4c_mask_reduce_prob_or` 与 `ablation_4d_mask_temporal_loss`，跳过自动对比。")
    else:
        lines.append(f"- {conclusion_4c_4d}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"✅ Markdown 报告已保存: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# 可选绘图
# ---------------------------------------------------------------------------
def save_psnr_bar_chart(results: list[dict], out_path: Path, baseline_name: Optional[str]):
    if not _HAS_MPL:
        print("⚠️  matplotlib 未安装，跳过绘图。")
        return

    sorted_results = sorted(results, key=_rank_key, reverse=True)
    baseline_psnr = 0.0
    if baseline_name:
        base = next((r for r in results if r["exp_name"] == baseline_name), None)
        if base:
            baseline_psnr = base["best_psnr"]

    names = [r["exp_name"].replace("ablation_", "").replace("_with_", "\n") for r in sorted_results]
    psnrs = [r["best_psnr"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.8), 6))
    colors = ["#e74c3c" if n == baseline_name else "#3498db" for n in [r["exp_name"] for r in sorted_results]]
    bars = ax.bar(names, psnrs, color=colors, edgecolor="white", linewidth=0.5)

    if baseline_psnr > 0:
        ax.axhline(baseline_psnr, color="gray", linestyle="--", linewidth=1, label=f"Baseline ({baseline_psnr:.2f})")

    # 在柱顶标注数值
    for bar, psnr in zip(bars, psnrs):
        if psnr >= 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{psnr:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_ylabel("Best PSNR (dB)")
    ax.set_title("Ablation PSNR Comparison")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"✅ PSNR 柱状图已保存: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="消融实验结果自动分析")
    parser.add_argument(
        "--exp_dir",
        default=str(TRAIN_LOGS_DIR),
        help=f"实验日志根目录（默认: {TRAIN_LOGS_DIR.relative_to(PROJECT_ROOT)}）",
    )
    parser.add_argument(
        "--baseline",
        default="config_true_baseline",
        help="用于计算相对增益的基准实验名称",
    )
    parser.add_argument("--plot", action="store_true", help="生成 PSNR 柱状图")
    parser.add_argument(
        "--out_csv",
        default=str(SUMMARY_CSV),
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--out_md",
        default=str(OUTPUT_DIR / "ablation_analysis_report.md"),
        help="输出 Markdown 报告路径",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"❌ 实验目录不存在: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 扫描实验目录: {exp_dir}")
    results = collect_all(exp_dir)
    if not results:
        print("未找到任何实验日志。")
        sys.exit(0)

    print(f"   找到 {len(results)} 个实验。\n")

    # 保存 CSV
    save_csv(results, Path(args.out_csv))

    # 保存 Markdown
    save_markdown(results, Path(args.out_md), args.baseline)

    # 可选绘图
    if args.plot:
        chart_path = OUTPUT_DIR / "psnr_bar_chart.png"
        save_psnr_bar_chart(results, chart_path, args.baseline)

    # 终端打印摘要
    print("\n📊 快速摘要（Top-5）：")
    top5 = sorted(results, key=_rank_key, reverse=True)[:5]
    for rank, r in enumerate(top5, 1):
        print(f"  {rank}. {r['exp_name']:<55s}  PSNR={r['best_psnr']:.4f}  SSIM={r['best_ssim']:.4f}  iter={r['best_iter']}")

    conclusion_4c_4d = build_4c_4d_conclusion(results)
    print("\n🧪 4c vs 4d 单变量结论：")
    if conclusion_4c_4d is None:
        print("  未同时检测到 4c 与 4d 结果，无法自动输出结论。")
    else:
        print(f"  {conclusion_4c_4d}")


if __name__ == "__main__":
    import os
    os.chdir(PROJECT_ROOT)
    main()
