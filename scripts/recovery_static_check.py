#!/usr/bin/env python3
"""
Static recovery check for core skeleton files/symbols/config contracts.

Usage:
  python scripts/recovery_static_check.py
  python scripts/recovery_static_check.py --out tmp/recovery_validation_logs/static_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "tmp" / "recovery_validation_logs" / "static_report.json"


def check_symbol(file_path: Path, tokens: list[str]) -> tuple[bool, str]:
    if not file_path.exists():
        return False, f"missing file: {file_path}"
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    for t in tokens:
        if t not in text:
            return False, f"missing token '{t}' in {file_path}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Static recovery validator")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output JSON report path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checks: list[dict] = []

    # A. Entry and routing
    checks.append(_mk("A1-main-train-entry", *check_symbol(
        ROOT / "main.py",
        ["def run_training(", "trainer_cls = get_obj_from_str(configs.trainer.target)"]
    )))
    checks.append(_mk("A2-main-test-entry", *check_symbol(
        ROOT / "main.py",
        ["def run_testing(", "predictor = Predictor(args.cfg_path, args.ckpt_path)"]
    )))

    # B. Dataset
    checks.append(_mk("B1-dataset-class", *check_symbol(
        ROOT / "datapipe" / "datasets.py",
        ["class AnytimeTemporalDataset", "def __getitem__", "'lr_sequence'", "'timestamps'", "'gt'", "'mask'"]
    )))
    checks.append(_mk("B2-create-dataset", *check_symbol(
        ROOT / "datapipe" / "datasets.py",
        ["def create_dataset(configs, parent_configs=None)", "params['parent_configs'] = parent_configs"]
    )))

    # C. Cloud mask processor
    checks.append(_mk("C1-cloud-processor-main", *check_symbol(
        ROOT / "utils" / "cloud_mask_processor.py",
        ["class CloudMaskProcessor", "def process_pixel_mask("]
    )))
    checks.append(_mk("C2-cloud-processor-compat", *check_symbol(
        ROOT / "utils" / "cloud_mask_processor.py",
        ["def _process_hard_mask(", "def _process_soft_mask("]
    )))

    # D. Network
    checks.append(_mk("D1-swinir", *check_symbol(
        ROOT / "models" / "network_swinir.py",
        ["class SwinIR", "class CloudCrossAttention", "TEMPORAL_FUSION_REGISTRY", "def temporal_fusion_mean("]
    )))

    # E. Trainer and indicator
    checks.append(_mk("E1-trainer-core", *check_symbol(
        ROOT / "trainer.py",
        ["class TrainerAlphaSR", "def training_step(", "def validation("]
    )))
    checks.append(_mk("E2-trainer-indicating-mask", *check_symbol(
        ROOT / "trainer.py",
        ["use_indicating_mask_in_training", "indicating_mask"]
    )))
    checks.append(_mk("E3-mask-ablation-trainer", *check_symbol(
        ROOT / "trainer_mask_ablation.py",
        ["class TrainerAlphaSRMaskAblation", "indicating_mask_reduce", "prob_or"]
    )))

    # F. Predictor compatibility
    checks.append(_mk("F1-inference-predictor", *check_symbol(
        ROOT / "inference.py",
        ["class Predictor", "def run_on_batch("]
    )))
    checks.append(_mk("F2-legacy-predictor", *check_symbol(
        ROOT / "predictor.py",
        ["class Predictor", "def run_on_batch(", "def run_inference("]
    )))

    # G. Config matrix contract
    if OmegaConf is None:
        checks.append(_mk("G0-omegaconf", False, "omegaconf unavailable, skip config matrix checks"))
    else:
        checks.extend(check_config_matrix())

    passed = sum(1 for c in checks if c["ok"]) 
    total = len(checks)
    failed = total - passed

    report = {
        "summary": {"total": total, "passed": passed, "failed": failed},
        "checks": checks,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[STATIC] total={total} passed={passed} failed={failed}")
    print(f"[STATIC] report={out_path}")

    return 0 if failed == 0 else 1


def _mk(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def check_config_matrix() -> list[dict]:
    rows: list[dict] = []
    cfg_dir = ROOT / "configs" / "ablation"
    required_soft = [
        "ablation_4a_soft_mask_test.yaml",
        "ablation_4a_with_cross_attention_softmask.yaml",
        "ablation_4b_advanced_processor_soft_simplified.yaml",
        "ablation_4c_mask_reduce_prob_or.yaml",
        "ablation_4d_mask_temporal_loss.yaml",
    ]
    required_hard = [
        "ablation_2b_with_cross_attention_no_posenc.yaml",
        "ablation_3b_with_cross_attention_posenc_learnable.yaml",
        "ablation_1d_baseline_plus_timeband_maskband.yaml",
    ]

    for fn in required_soft:
        p = cfg_dir / fn
        if not p.exists():
            rows.append(_mk(f"G-soft-exists-{fn}", False, "missing config"))
            continue
        cfg = OmegaConf.load(p)
        v = bool(OmegaConf.select(cfg, "features.mask_band.use_advanced_processor"))
        rows.append(_mk(f"G-soft-advanced-{fn}", v is True, f"use_advanced_processor={v}"))

    for fn in required_hard:
        p = cfg_dir / fn
        if not p.exists():
            rows.append(_mk(f"G-hard-exists-{fn}", False, "missing config"))
            continue
        cfg = OmegaConf.load(p)
        v = bool(OmegaConf.select(cfg, "features.mask_band.use_advanced_processor"))
        rows.append(_mk(f"G-hard-advanced-{fn}", v is False, f"use_advanced_processor={v}"))

    p4c = cfg_dir / "ablation_4c_mask_reduce_prob_or.yaml"
    if p4c.exists():
        cfg = OmegaConf.load(p4c)
        tgt = str(OmegaConf.select(cfg, "trainer.target"))
        ok = tgt == "trainer_mask_ablation.TrainerAlphaSRMaskAblation"
        rows.append(_mk("G-4c-trainer-target", ok, f"trainer.target={tgt}"))
    else:
        rows.append(_mk("G-4c-trainer-target", False, "missing ablation_4c config"))

    p4d = cfg_dir / "ablation_4d_mask_temporal_loss.yaml"
    if p4d.exists():
        cfg = OmegaConf.load(p4d)
        tgt = str(OmegaConf.select(cfg, "trainer.target"))
        ok = tgt == "trainer_mask_ablation.TrainerAlphaSRMaskTemporalLossAblation"
        rows.append(_mk("G-4d-trainer-target", ok, f"trainer.target={tgt}"))
    else:
        rows.append(_mk("G-4d-trainer-target", False, "missing ablation_4d config"))

    return rows


if __name__ == "__main__":
    raise SystemExit(main())
