#!/usr/bin/env python3
"""
scripts/best_ckpt_manager.py
============================
最佳权重管理器 —— 对应文档中的 BestCheckpointManager。

功能：
  list    列出所有实验的最佳权重路径与对应指标
  collect 将分散在时间戳子目录中的 model_best.pth 汇总到
           ./best_ckpts/{exp_name}/model.pth（符号链接或复制）
  clean   清理旧的非最佳检查点文件（保留 model_best.pth）

用法：
  python scripts/best_ckpt_manager.py list
  python scripts/best_ckpt_manager.py collect [--copy]
  python scripts/best_ckpt_manager.py clean   [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 目录常量（相对于项目根目录）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TRAIN_LOGS_DIR = PROJECT_ROOT / "training_logs" / "experiments"
BEST_CKPTS_DIR = PROJECT_ROOT / "best_ckpts"

# 日志中最佳 PSNR 行的正则
_RE_BEST = re.compile(r"New best metric:\s*([\d.]+)")
_RE_PSNR = re.compile(r"PSNR:\s*([\d.]+)")
_RE_SSIM = re.compile(r"SSIM:\s*([\d.]+)")
_RE_ITER = re.compile(r"Iter\s+(\d+)")


# ---------------------------------------------------------------------------
# 解析单个实验的最佳指标
# ---------------------------------------------------------------------------
def _parse_best_from_log(log_path: Path) -> dict:
    """从 training.log 中提取最佳 PSNR/SSIM 所在迭代及值。"""
    best_psnr = -1.0
    best_ssim = -1.0
    best_iter = 0
    current_iter = 0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 记录当前迭代
            m = _RE_ITER.search(line)
            if m:
                current_iter = int(m.group(1))

            # 当找到新最佳时记录迭代
            if _RE_BEST.search(line):
                best_iter = current_iter

            # 收集每行中的 PSNR/SSIM（用于统计最高值）
            m_psnr = _RE_PSNR.search(line)
            m_ssim = _RE_SSIM.search(line)
            if m_psnr and "Validation" in line:
                psnr = float(m_psnr.group(1))
                if psnr > best_psnr:
                    best_psnr = psnr
                    if m_ssim:
                        best_ssim = float(m_ssim.group(1))

    return {
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "best_iter": best_iter,
    }


# ---------------------------------------------------------------------------
# 收集所有实验信息
# ---------------------------------------------------------------------------
def collect_experiments() -> list[dict]:
    """扫描 training_logs/experiments 并返回每个实验的最新最佳模型信息。"""
    if not TRAIN_LOGS_DIR.exists():
        print(f"⚠️  training_logs 目录不存在: {TRAIN_LOGS_DIR}")
        return []

    results = []
    for exp_dir in sorted(TRAIN_LOGS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        # 可能存在多个时间戳子目录，取最新的
        run_dirs = sorted(
            [d for d in exp_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if not run_dirs:
            continue

        for run_dir in run_dirs:
            ckpt_path = run_dir / "ckpts" / "model_best.pth"
            log_path = run_dir / "training.log"
            if ckpt_path.exists():
                metrics = {}
                if log_path.exists():
                    metrics = _parse_best_from_log(log_path)
                results.append(
                    {
                        "exp_name": exp_dir.name,
                        "run_dir": run_dir,
                        "ckpt_path": ckpt_path,
                        "log_path": log_path if log_path.exists() else None,
                        **metrics,
                    }
                )
                break  # 使用最新的有模型的 run

    return results


# ---------------------------------------------------------------------------
# 子命令：list
# ---------------------------------------------------------------------------
def cmd_list(args):
    experiments = collect_experiments()
    if not experiments:
        print("未找到任何实验的最佳权重（model_best.pth）。")
        return

    # 按 PSNR 排序
    experiments.sort(key=lambda x: x.get("best_psnr", -1), reverse=True)

    # 打印表格
    col_exp = max(len(e["exp_name"]) for e in experiments) + 2
    header = f"{'Rank':<5} {'实验名称':<{col_exp}} {'Best PSNR':>10} {'Best SSIM':>10} {'Best Iter':>10}  权重路径"
    print(header)
    print("-" * len(header))
    for rank, e in enumerate(experiments, 1):
        psnr = f"{e.get('best_psnr', -1):.4f}" if e.get("best_psnr", -1) >= 0 else "N/A"
        ssim = f"{e.get('best_ssim', -1):.4f}" if e.get("best_ssim", -1) >= 0 else "N/A"
        biter = str(e.get("best_iter", 0))
        rel_ckpt = e["ckpt_path"].relative_to(PROJECT_ROOT)
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"{rank:>2}. "))
        print(f"{medal:<5} {e['exp_name']:<{col_exp}} {psnr:>10} {ssim:>10} {biter:>10}  {rel_ckpt}")

    print(f"\n共 {len(experiments)} 个实验。")


# ---------------------------------------------------------------------------
# 子命令：collect
# ---------------------------------------------------------------------------
def cmd_collect(args):
    """将各实验 model_best.pth 汇总到 best_ckpts/{exp_name}/model.pth。"""
    experiments = collect_experiments()
    if not experiments:
        print("未找到任何最佳权重，无需汇总。")
        return

    BEST_CKPTS_DIR.mkdir(parents=True, exist_ok=True)
    use_copy = getattr(args, "copy", False)

    for e in experiments:
        target_dir = BEST_CKPTS_DIR / e["exp_name"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "model.pth"

        # 删除旧的符号链接或文件
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()

        src = e["ckpt_path"]
        if use_copy:
            shutil.copy2(src, target_path)
            action = "✅ [复制]"
        else:
            # 创建相对符号链接（更节省空间）
            rel = os.path.relpath(src, target_dir)
            target_path.symlink_to(rel)
            action = "🔗 [链接]"

        psnr = e.get("best_psnr", -1)
        psnr_str = f"PSNR={psnr:.4f}" if psnr >= 0 else ""
        print(f"{action} {e['exp_name']} → {target_path.relative_to(PROJECT_ROOT)}  {psnr_str}")

    print(f"\n汇总完成。目录: {BEST_CKPTS_DIR.relative_to(PROJECT_ROOT)}/")


# ---------------------------------------------------------------------------
# 子命令：clean
# ---------------------------------------------------------------------------
def cmd_clean(args):
    """删除各实验目录中以迭代号命名的非最佳检查点（保留 model_best.pth）。"""
    dry_run = getattr(args, "dry_run", False)
    if not TRAIN_LOGS_DIR.exists():
        print("training_logs 目录不存在。")
        return

    total_freed = 0
    for exp_dir in sorted(TRAIN_LOGS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            ckpt_dir = run_dir / "ckpts"
            if not ckpt_dir.exists():
                continue
            for f in ckpt_dir.glob("model_*.pth"):
                if f.name == "model_best.pth":
                    continue
                size = f.stat().st_size
                if dry_run:
                    print(f"[dry-run] 将删除 {f.relative_to(PROJECT_ROOT)}  ({size/1e6:.1f} MB)")
                else:
                    f.unlink()
                    print(f"🗑️  删除 {f.relative_to(PROJECT_ROOT)}  ({size/1e6:.1f} MB)")
                total_freed += size

    freed_mb = total_freed / 1e6
    label = "可释放" if dry_run else "已释放"
    print(f"\n{label}空间: {freed_mb:.1f} MB")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AEF-SwinIR 最佳权重管理器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd")

    # list
    sub.add_parser("list", help="列出所有实验最佳权重与指标")

    # collect
    p_collect = sub.add_parser("collect", help="汇总最佳权重到 best_ckpts/ 目录")
    p_collect.add_argument(
        "--copy",
        action="store_true",
        help="复制文件（默认创建符号链接，更省空间）",
    )

    # clean
    p_clean = sub.add_parser("clean", help="清理非最佳检查点文件")
    p_clean.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="只打印将被删除的文件，不实际删除",
    )

    args = parser.parse_args()
    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "collect":
        cmd_collect(args)
    elif args.cmd == "clean":
        cmd_clean(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
