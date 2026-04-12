#!/usr/bin/env python3
"""
Minimal end-to-end recovery probe:
1) Run 1-iter train through main.py
2) Use produced ckpt to run main.py test once
3) Emit JSON report for auditing

Usage:
  python scripts/recovery_e2e_check.py \
      --cfg configs/ablation/config_true_baseline.yaml \
      --input-dir data/Cloud_test/processed_data_SR_10m/test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def latest_ckpt(save_root: Path) -> Path | None:
    if not save_root.exists():
        return None
    runs = sorted([d for d in save_root.iterdir() if d.is_dir()])
    if not runs:
        return None
    run_dir = runs[-1]
    c = run_dir / "ckpts" / "model_1.pth"
    return c if c.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end recovery probe")
    parser.add_argument("--cfg", default="configs/ablation/config_true_baseline.yaml")
    parser.add_argument("--input-dir", default="data/Cloud_test/processed_data_SR_10m/test")
    parser.add_argument("--out-dir", default="tmp/recovery_validation_logs")
    args = parser.parse_args()

    cfg_path = ROOT / args.cfg
    out_root = ROOT / args.out_dir
    e2e_dir = out_root / "e2e"
    train_log = e2e_dir / "train.log"
    test_log = e2e_dir / "test.log"
    report_json = e2e_dir / "e2e_report.json"

    cfg = OmegaConf.load(cfg_path)

    # build a temporary fast config
    fast_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    fast_save = ROOT / "tmp" / "recovery_e2e" / "train_run"
    OmegaConf.update(fast_cfg, "train.save_dir", str(fast_save), merge=True)
    OmegaConf.update(fast_cfg, "train.iterations", 1, merge=True)
    OmegaConf.update(fast_cfg, "train.save_freq", 1, merge=True)
    OmegaConf.update(fast_cfg, "train.val_freq", 999999, merge=True)
    OmegaConf.update(fast_cfg, "train.log_freq", [1, 999999, 999999], merge=True)
    OmegaConf.update(fast_cfg, "train.local_logging", False, merge=True)
    OmegaConf.update(fast_cfg, "train.num_workers", 0, merge=True)
    OmegaConf.update(fast_cfg, "train.export_best_ckpt", False, merge=True)
    if OmegaConf.select(fast_cfg, "data.train.params.sample_num") is not None:
        OmegaConf.update(fast_cfg, "data.train.params.sample_num", 1, merge=True)
    if OmegaConf.select(fast_cfg, "data.val.params.sample_num") is not None:
        OmegaConf.update(fast_cfg, "data.val.params.sample_num", 1, merge=True)

    fast_cfg_path = e2e_dir / "fast_config.yaml"
    fast_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(fast_cfg, fast_cfg_path)

    py = sys.executable
    train_cmd = [py, "main.py", "--cfg_path", str(fast_cfg_path), "--mode", "train"]
    train_rc = run(train_cmd, train_log)

    ckpt = latest_ckpt(fast_save)
    test_rc = 999
    pred_dir = ROOT / "tmp" / "recovery_e2e" / "preds"

    if train_rc == 0 and ckpt is not None:
        test_cmd = [
            py,
            "main.py",
            "--cfg_path",
            str(fast_cfg_path),
            "--mode",
            "test",
            "--ckpt_path",
            str(ckpt),
            "--input_dir",
            str(ROOT / args.input_dir),
            "--output_dir",
            str(pred_dir),
        ]
        test_rc = run(test_cmd, test_log)
    else:
        test_log.write_text("skip test: train failed or ckpt missing\n", encoding="utf-8")

    report = {
        "train_return_code": train_rc,
        "test_return_code": test_rc,
        "train_log": str(train_log.relative_to(ROOT)),
        "test_log": str(test_log.relative_to(ROOT)),
        "fast_cfg": str(fast_cfg_path.relative_to(ROOT)),
        "ckpt": str(ckpt.relative_to(ROOT)) if ckpt else None,
        "pred_dir": str(pred_dir.relative_to(ROOT)),
        "passed": (train_rc == 0 and test_rc == 0),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[E2E] train_rc={train_rc} test_rc={test_rc}")
    print(f"[E2E] report={report_json}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
