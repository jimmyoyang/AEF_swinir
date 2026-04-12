#!/usr/bin/env python3
"""
scripts/validate_ablation_configs.py
======================================
验证所有消融实验配置文件的一致性。

检查项：
    - 关键超参数一致性（训练迭代数、随机种子）
    - 必要字段存在（model/train/data/features/trainer）
    - 当前真实路径合法性（如 train.batch、model.params.use_cross_attention、features.mask_band.enabled）
    - 关键字段取值有效性（batch 维度、cross_num_heads）

用法：
  python scripts/validate_ablation_configs.py
  python scripts/validate_ablation_configs.py --strict      # 任何 warn 也视为失败
    python scripts/validate_ablation_configs.py --ci-summary  # 输出单行 CI 摘要
  python scripts/validate_ablation_configs.py --dir configs/ablation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
try:
    from omegaconf import OmegaConf
    _HAS_OC = True
except ImportError:
    _HAS_OC = False

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ABLATION_DIR = PROJECT_ROOT / "configs" / "ablation"

# 参考值（应在所有实验间保持相同）
REFERENCE = {
    "train.iterations": 5000,
    "train.seed": 42,
}

# 必要字段路径（点分隔）
REQUIRED_FIELDS = [
    "model",
    "model.target",
    "model.params",
    "trainer",
    "trainer.target",
    "train",
    "train.iterations",
    "train.seed",
    "train.lr",
    "train.batch",
    "data",
    "data.train",
    "data.train.target",
    "features",
    "features.time_band.enabled",
    "features.mask_band.enabled",
]

# 可选但推荐字段
RECOMMENDED_FIELDS = [
    "model.params.use_cross_attention",
    "model.params.use_pos_emb",
    "model.params.cross_num_heads",
    "features.mask_band.use_advanced_processor",
]


class ConfigChecker:
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def _err(self, msg: str):
        self.errors.append(msg)

    def _warn(self, msg: str):
        self.warnings.append(msg)

    def check_file(self, yaml_path: Path) -> bool:
        """检查单个配置文件。返回是否通过（无错误）。"""
        self.errors = []
        self.warnings = []

        if not _HAS_OC:
            print("❌ omegaconf 未安装，无法解析 YAML。", file=sys.stderr)
            return False

        try:
            cfg = OmegaConf.load(str(yaml_path))
        except Exception as e:
            self._err(f"YAML 解析失败: {e}")
            return False

        # 1. 必要字段
        for field in REQUIRED_FIELDS:
            val = OmegaConf.select(cfg, field)
            if val is None:
                self._err(f"缺少必要字段: {field}")

        # 2. 推荐字段
        for field in RECOMMENDED_FIELDS:
            val = OmegaConf.select(cfg, field)
            if val is None:
                self._warn(f"缺少推荐字段: {field}")

        # 3. 参考值一致性
        for field, ref_val in REFERENCE.items():
            actual = OmegaConf.select(cfg, field)
            if actual is not None and actual != ref_val:
                self._warn(
                    f"字段 '{field}' 值为 {actual!r}，参考值为 {ref_val!r}（可能是故意修改）"
                )

        # 4. train.batch 合法性检查：应为 [train_bs, val_bs]
        batch = OmegaConf.select(cfg, "train.batch")
        if batch is not None:
            if not isinstance(batch, (list, tuple)):
                self._err(f"train.batch={batch!r} 无效，应为长度>=1的列表")
            elif len(batch) < 1:
                self._err("train.batch 不能为空")
            else:
                for i, b in enumerate(batch):
                    if not isinstance(b, int) or b < 1:
                        self._err(f"train.batch[{i}]={b!r} 无效，应为正整数")

        # 5. 若声明了 use_cross_attention=True 但无 cross_num_heads
        use_ca = OmegaConf.select(cfg, "model.params.use_cross_attention")
        cross_heads = OmegaConf.select(cfg, "model.params.cross_num_heads")
        if use_ca is True and cross_heads is None:
            self._warn("model.params.use_cross_attention=True 但未设置 model.params.cross_num_heads")
        if cross_heads is not None and (not isinstance(cross_heads, int) or cross_heads < 1):
            self._err(f"model.params.cross_num_heads={cross_heads!r} 无效，应为正整数")

        # 6. 若声明了 use_pos_emb=True 但同时 use_cross_attention=False
        use_pos = OmegaConf.select(cfg, "model.params.use_pos_emb")
        if use_pos is True and use_ca is False:
            self._warn("model.params.use_pos_emb=True 但 model.params.use_cross_attention=False（位置编码可能无效）")

        # 7. mask 类型检查（若声明）
        mask_type = OmegaConf.select(cfg, "features.mask_band.processor.mask_type")
        if mask_type is not None and str(mask_type).lower() not in {"hard", "soft"}:
            self._err(f"features.mask_band.processor.mask_type={mask_type!r} 无效，应为 hard/soft")

        passed = len(self.errors) == 0 and (not self.strict or len(self.warnings) == 0)
        return passed


def main():
    parser = argparse.ArgumentParser(description="消融实验配置验证")
    parser.add_argument(
        "--dir",
        default=str(ABLATION_DIR),
        help=f"配置目录（默认: configs/ablation）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：warning 也视为失败",
    )
    parser.add_argument(
        "--pattern",
        default="*.yaml",
        help="文件模式过滤（默认: *.yaml）",
    )
    parser.add_argument(
        "--ci-summary",
        action="store_true",
        help="额外输出单行摘要，便于 CI 日志检索",
    )
    args = parser.parse_args()

    config_dir = Path(args.dir)
    if not config_dir.exists():
        print(f"❌ 目录不存在: {config_dir}", file=sys.stderr)
        sys.exit(1)

    yaml_files = sorted(config_dir.glob(args.pattern))
    if not yaml_files:
        print(f"⚠️  目录中未找到 YAML 文件: {config_dir}")
        sys.exit(0)

    checker = ConfigChecker(strict=args.strict)
    total = len(yaml_files)
    passed = 0
    failed = 0
    warned = 0

    print(f"🔍 检查 {total} 个配置文件 ({'严格模式' if args.strict else '普通模式'})\n")

    for yf in yaml_files:
        ok = checker.check_file(yf)
        rel = yf.relative_to(PROJECT_ROOT)
        if checker.errors:
            failed += 1
            status = "❌ FAIL"
        elif checker.warnings:
            warned += 1
            status = "⚠️  WARN" if not args.strict else "❌ FAIL"
            if not ok:
                failed += 1
        else:
            passed += 1
            status = "✅ OK  "

        print(f"  {status}  {rel}")
        for err in checker.errors:
            print(f"         Error: {err}")
        for w in checker.warnings:
            print(f"         Warn:  {w}")

    print(f"\n{'=' * 60}")
    print(f"结果: {passed} pass / {warned} warn / {failed} fail  (共 {total} 个)")

    if args.ci_summary:
        mode = "strict" if args.strict else "normal"
        print(f"CI_SUMMARY mode={mode} total={total} pass={passed} warn={warned} fail={failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
