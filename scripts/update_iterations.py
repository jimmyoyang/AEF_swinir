#!/usr/bin/env python3
"""
scripts/update_iterations.py
==============================
批量更新消融实验配置文件中的训练迭代数。

用法：
  # 查看目前所有配置的 iterations（dry-run 预览）
  python scripts/update_iterations.py --new_val 10000

  # 实际写入
  python scripts/update_iterations.py --new_val 10000 --apply

  # 只更新部分实验（用前缀过滤）
  python scripts/update_iterations.py --new_val 10000 --apply --filter ablation_3

  # 同时更新 seed
  python scripts/update_iterations.py --new_val 10000 --seed 123 --apply
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ABLATION_DIR = PROJECT_ROOT / "configs" / "ablation"

# 匹配 "iterations: 数字" 行（支持行内注释）
_RE_ITER = re.compile(r"^(\s*iterations\s*:\s*)(\d+)(.*)", re.MULTILINE)
_RE_SEED = re.compile(r"^(\s*seed\s*:\s*)(\d+)(.*)", re.MULTILINE)


def _replace_field(content: str, pattern: re.Pattern, new_val: int) -> tuple[str, int, int]:
    """返回 (新内容, 旧值, 替换次数)。若未匹配到 → 返回原内容和 count=0。"""
    old_val = -1
    count = 0

    def _sub(m: re.Match) -> str:
        nonlocal old_val, count
        old_val = int(m.group(2))
        count += 1
        return f"{m.group(1)}{new_val}{m.group(3)}"

    new_content = pattern.sub(_sub, content)
    return new_content, old_val, count


def main():
    parser = argparse.ArgumentParser(description="批量更新消融配置中的训练迭代数")
    parser.add_argument(
        "--new_val", type=int, required=True,
        help="新的 train.iterations 值",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="同时更新 train.seed（可选）",
    )
    parser.add_argument(
        "--dir", default=str(ABLATION_DIR),
        help="配置目录（默认: configs/ablation）",
    )
    parser.add_argument(
        "--filter", default="",
        help="文件名前缀过滤（例: ablation_3 只更新 3x 系列）",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="实际写入；不加此参数则只预览（dry-run）",
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="写入前备份原始文件（追加 .bak.YYYYMMDD-HHMMSS 后缀）",
    )
    args = parser.parse_args()

    config_dir = Path(args.dir)
    if not config_dir.exists():
        print(f"❌ 目录不存在: {config_dir}", file=sys.stderr)
        sys.exit(1)

    yaml_files = sorted(config_dir.glob("*.yaml"))
    if args.filter:
        yaml_files = [f for f in yaml_files if f.name.startswith(args.filter)]

    if not yaml_files:
        print("未找到匹配的 YAML 文件。")
        sys.exit(0)

    mode_label = "📝 apply" if args.apply else "🔍 dry-run"
    print(f"{mode_label} | 新 iterations={args.new_val}"
          + (f", seed={args.seed}" if args.seed else "")
          + f" | {len(yaml_files)} 个文件\n")

    changed = 0
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    for yf in yaml_files:
        original = yf.read_text(encoding="utf-8")
        content = original

        # 更新 iterations
        content, old_iter, iter_count = _replace_field(content, _RE_ITER, args.new_val)

        # 更新 seed（可选）
        old_seed = -1
        if args.seed is not None:
            content, old_seed, seed_count = _replace_field(content, _RE_SEED, args.seed)
        else:
            seed_count = 0

        rel = yf.relative_to(PROJECT_ROOT)

        if iter_count == 0:
            print(f"  ⚠️  {rel}  — 未找到 iterations 字段")
            continue

        iter_changed = (old_iter != args.new_val)
        seed_changed = (args.seed is not None and old_seed != args.seed and seed_count > 0)

        if not iter_changed and not seed_changed:
            print(f"  ✅ {rel}  — 已是最新值，跳过")
            continue

        changed += 1
        delta_iter = f"  iterations: {old_iter} → {args.new_val}"
        delta_seed = (f"  seed: {old_seed} → {args.seed}" if seed_changed else "")
        print(f"  {'✏️ ' if args.apply else '→ '}{rel}")
        print(f"    {delta_iter}" + (f"\n    {delta_seed}" if delta_seed else ""))

        if args.apply:
            if args.backup:
                backup_path = yf.with_suffix(f".bak.{ts}")
                shutil.copy2(str(yf), str(backup_path))
            yf.write_text(content, encoding="utf-8")

    print(f"\n{'=' * 60}")
    action = "已更新" if args.apply else "需更新（重新运行时加 --apply）"
    print(f"共 {action} {changed} / {len(yaml_files)} 个配置文件。")


if __name__ == "__main__":
    main()
