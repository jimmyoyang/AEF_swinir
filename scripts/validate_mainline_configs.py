#!/usr/bin/env python3
"""
Validate generated mainline ablation configs.

Checks:
- YAML parse success
- mainline model target is SwinIR
- position embedding flags are disabled
- trainer matches base (4c vs 4d)
- essential train keys are present
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def expected_trainer(base: str) -> str:
    if base == "4c":
        return "trainer_mask_ablation.TrainerAlphaSRMaskAblation"
    if base == "4d":
        return "trainer_mask_ablation.TrainerAlphaSRMaskTemporalLossAblation"
    raise ValueError(base)


def expected_model_target(path: Path) -> str:
    name = path.name
    if "_v2_" in name:
        return "models.network_swinir_post_v2.SwinIRPostV2"
    if "_v3_" in name:
        return "models.network_swinir_post_v3.SwinIRPostV3"
    return "models.network_swinir.SwinIR"


def validate_file(path: Path, base: str) -> list[str]:
    errs: list[str] = []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"YAML parse error: {e}"]

    model = data.get("model", {})
    params = model.get("params", {})
    train = data.get("train", {})
    trainer = (data.get("trainer") or {}).get("target")

    exp_model_target = expected_model_target(path)
    if model.get("target") != exp_model_target:
        errs.append(
            f"model.target mismatch: expected {exp_model_target}, got {model.get('target')}"
        )

    if params.get("use_pos_emb") is not False:
        errs.append(f"use_pos_emb should be false, got {params.get('use_pos_emb')}")
    if params.get("use_learnable_pos_emb") is not False:
        errs.append(
            f"use_learnable_pos_emb should be false, got {params.get('use_learnable_pos_emb')}"
        )

    exp_t = expected_trainer(base)
    if trainer != exp_t:
        errs.append(f"trainer.target mismatch: expected {exp_t}, got {trainer}")

    if int(train.get("iterations", -1)) != 5000:
        errs.append(f"train.iterations should be 5000, got {train.get('iterations')}")

    if base == "4d" and "temporal_loss_reduce" not in train:
        errs.append("missing train.temporal_loss_reduce for 4d base")

    for key in ["save_dir", "log_freq", "save_freq", "val_freq"]:
        if key not in train:
            errs.append(f"missing train.{key}")

    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated mainline YAML configs")
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=[
            "configs/ablation_mainline_4c_v2",
            "configs/ablation_mainline_4c_v3",
            "configs/ablation_mainline_4d_v2",
            "configs/ablation_mainline_4d_v3",
        ],
    )
    args = parser.parse_args()

    total = 0
    failed = 0

    for d in args.dirs:
        dp = ROOT / d
        if not dp.exists():
            print(f"[WARN] missing dir: {d}")
            continue

        base = "4d" if "mainline_4d" in d else "4c"
        for p in sorted(dp.glob("*.yaml")):
            total += 1
            errs = validate_file(p, base=base)
            if errs:
                failed += 1
                print(f"[FAIL] {p.relative_to(ROOT)}")
                for e in errs:
                    print(f"       - {e}")
            else:
                print(f"[OK]   {p.relative_to(ROOT)}")

    print(f"\nSUMMARY total={total} failed={failed} passed={total - failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
