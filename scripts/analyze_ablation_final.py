#!/usr/bin/env python3
"""
scripts/analyze_ablation_final.py
====================================
在 analyze_ablation_results.py 基础上生成最终的论文级综合报告：

  1. 调用 analyze_ablation_results 逻辑生成/刷新 CSV + Markdown
  2. 打印分组最优实验与相对基线的 PSNR 增益
  3. 汇总进论文 Table 所需格式（Markdown/LaTeX 可选）
  4. 可选：生成 PSNR 学习曲线图（所有实验）

用法：
  python scripts/analyze_ablation_final.py
  python scripts/analyze_ablation_final.py --latex       # 附加 LaTeX 表格
  python scripts/analyze_ablation_final.py --curves      # 附加学习曲线图
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_ablation_results import (  # noqa: E402
    collect_all,
    save_csv,
    save_markdown,
    save_psnr_bar_chart,
    build_4c_4d_conclusion,
    OUTPUT_DIR,
    SUMMARY_CSV,
    TRAIN_LOGS_DIR,
    _HAS_MPL,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL_LOCAL = True
except ImportError:
    _HAS_MPL_LOCAL = False


def _rank_key(r: dict) -> tuple[float, float]:
    """Ranking key: prioritize PSNR, then SSIM as tie-breaker."""
    return (r.get("best_psnr", -1.0), r.get("best_ssim", -1.0))


def generate_latex_table(results: list[dict], baseline_name: str) -> str:
    """生成 LaTeX booktabs 格式的消融实验表格。"""
    baseline = next((r for r in results if r["exp_name"] == baseline_name), None)
    base_psnr = baseline["best_psnr"] if baseline else 0.0

    sorted_r = sorted(results, key=_rank_key, reverse=True)

    lines = [
        r"\begin{table}[h!]",
        r"  \centering",
        r"  \caption{消融实验结果对比（training iterations=5000, seed=42）}",
        r"  \label{tab:ablation}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    实验名称 & PSNR (dB) & SSIM & ERGAS & $\Delta$PSNR \\",
        r"    \midrule",
    ]
    current_group = ""
    for r in sorted_r:
        group = r["group"]
        if group != current_group:
            if current_group:
                lines.append(r"    \midrule")
            lines.append(rf"    \multicolumn{{5}}{{l}}{{\textit{{{group}}}}} \\")
            current_group = group
        psnr = r["best_psnr"]
        delta = f"{psnr - base_psnr:+.4f}" if base_psnr > 0 and psnr >= 0 else "-"
        ssim = f"{r['best_ssim']:.4f}" if r["best_ssim"] >= 0 else "-"
        ergas = f"{r['best_ergas']:.2f}" if r["best_ergas"] != float("inf") else "-"
        mark = r" \textbf" if r["exp_name"] == baseline_name else ""
        psnr_str = f"{psnr:.4f}" if psnr >= 0 else "-"
        exp_name_latex = r["exp_name"].replace("_", "\\_")
        lines.append(
            rf"    {exp_name_latex:<55s} & {mark}{{{psnr_str}}} & {ssim} & {ergas} & {delta} \\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_learning_curves(results: list[dict], out_path: Path):
    """生成所有实验的 PSNR 学习曲线图（按组着色）。"""
    if not _HAS_MPL_LOCAL:
        print("⚠️  matplotlib 不可用，跳过学习曲线图。")
        return

    group_colors = {
        "G1-基线": "#2ecc71",
        "G2-交叉注意力": "#3498db",
        "G3-位置编码": "#9b59b6",
        "G4-软掩膜": "#e67e22",
        "G5-光谱后处理": "#e74c3c",
        "其他": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    for r in results:
        curve = r["val_curve"]
        if len(curve) < 2:
            continue
        iters, psnrs = zip(*curve)
        color = group_colors.get(r["group"], "#95a5a6")
        label = r["exp_name"].replace("ablation_", "")
        ax.plot(iters, psnrs, "-o", markersize=2, linewidth=1, color=color, label=label, alpha=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("消融实验 PSNR 学习曲线")
    ax.legend(fontsize=6, ncol=3, loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"✅ 学习曲线图已保存: {out_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="消融实验综合最终分析")
    parser.add_argument(
        "--exp_dir", default=str(TRAIN_LOGS_DIR),
        help="实验日志根目录",
    )
    parser.add_argument(
        "--baseline", default="config_true_baseline",
        help="基准实验名称",
    )
    parser.add_argument("--latex", action="store_true", help="同时输出 LaTeX 表格")
    parser.add_argument("--curves", action="store_true", help="生成学习曲线图")
    parser.add_argument("--plot", action="store_true", help="生成 PSNR 柱状图")
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

    # 标准 CSV + MD 报告
    save_csv(results, SUMMARY_CSV)
    md_path = OUTPUT_DIR / "ablation_analysis_report.md"
    save_markdown(results, md_path, args.baseline)

    # 可选 PSNR 柱图
    if args.plot:
        save_psnr_bar_chart(results, OUTPUT_DIR / "psnr_bar_chart.png", args.baseline)

    # 可选学习曲线
    if args.curves:
        generate_learning_curves(results, OUTPUT_DIR / "learning_curves.png")

    # 可选 LaTeX 表格
    if args.latex:
        latex_str = generate_latex_table(results, args.baseline)
        latex_path = OUTPUT_DIR / "ablation_table.tex"
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(latex_str, encoding="utf-8")
        print(f"✅ LaTeX 表格已保存: {latex_path.relative_to(PROJECT_ROOT)}")

    # 分组最优汇总
    baseline = next((r for r in results if r["exp_name"] == args.baseline), None)
    base_psnr = baseline["best_psnr"] if baseline else 0.0

    print("\n📋 分组最优实验汇总：")
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)

    print(f"  {'Group':<22}  {'Best Exp':<55}  PSNR   ΔPSNR")
    print("  " + "-" * 100)
    for group_name in sorted(groups):
        best = max(groups[group_name], key=_rank_key)
        delta = f"{best['best_psnr'] - base_psnr:+.4f}" if base_psnr > 0 else "   N/A"
        print(f"  {group_name:<22}  {best['exp_name']:<55}  {best['best_psnr']:.4f}  {delta}")

    c4 = build_4c_4d_conclusion(results)
    print("\n🧪 4c vs 4d 单变量结论：")
    if c4 is None:
        print("  未同时检测到 4c 与 4d 结果，无法自动输出结论。")
    else:
        print(f"  {c4}")

    sorted_results = sorted(results, key=_rank_key, reverse=True)
    best = sorted_results[0]
    co_winners = [
        r for r in sorted_results
        if r["best_psnr"] == best["best_psnr"] and r["best_ssim"] == best["best_ssim"]
    ]
    if len(co_winners) > 1:
        names = ", ".join(r["exp_name"] for r in co_winners)
        print(f"\n🏆 全局并列最优: {names}")
    else:
        print(f"\n🏆 全局最优: {best['exp_name']} (按 PSNR, SSIM 排序)")


if __name__ == "__main__":
    main()
