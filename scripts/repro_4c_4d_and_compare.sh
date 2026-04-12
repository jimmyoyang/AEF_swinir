#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/lm_data_afs/wangzining/charles/AEF_swinir"
cd "$ROOT"

ENV_NAME="${1:-alphaearth}"
TARGET_ITERS="${2:-7000}"

# historical best runs confirmed by log scan
REC4C_LOG="training_logs/experiments/ablation_4c_mask_reduce_prob_or/2026-03-16_16-20-28/training.log"
REC4C_CKPT="training_logs/experiments/ablation_4c_mask_reduce_prob_or/2026-03-16_16-20-28/ckpts/model_best.pth"
REC4D_LOG="training_logs/experiments/ablation_4d_mask_temporal_loss/2026-03-16_16-08-35/training.log"
REC4D_CKPT="training_logs/experiments/ablation_4d_mask_temporal_loss/2026-03-16_16-08-35/ckpts/model_best.pth"

CFG4C_BASE="configs/ablation/ablation_4c_mask_reduce_prob_or.yaml"
CFG4D_BASE="configs/ablation/ablation_4d_mask_temporal_loss.yaml"
CFG4C_NEW="tmp/ablation_4c_repro_resume_${TARGET_ITERS}.yaml"
CFG4D_NEW="tmp/ablation_4d_repro_resume_${TARGET_ITERS}.yaml"

SAVE4C="training_logs/experiments/ablation_4c_repro_resume_${TARGET_ITERS}_isolated"
SAVE4D="training_logs/experiments/ablation_4d_repro_resume_${TARGET_ITERS}_isolated"

CK4C_ISO="$SAVE4C/ckpts/model_5000.pth"
CK4D_ISO="$SAVE4D/ckpts/model_5000.pth"

OUT4C="debug_output/compare_4c_recovered_vs_repro_${TARGET_ITERS}.csv"
OUT4D="debug_output/compare_4d_recovered_vs_repro_${TARGET_ITERS}.csv"
OUT_SUM="debug_output/compare_4c_4d_repro_summary_${TARGET_ITERS}.csv"

mkdir -p tmp "$SAVE4C/ckpts" "$SAVE4D/ckpts" debug_output

# prepare cfgs with extended iterations
cp "$CFG4C_BASE" "$CFG4C_NEW"
cp "$CFG4D_BASE" "$CFG4D_NEW"
sed -i "s/^[[:space:]]*iterations:[[:space:]]*5000[[:space:]]*$/  iterations: ${TARGET_ITERS}/" "$CFG4C_NEW"
sed -i "s/^[[:space:]]*iterations:[[:space:]]*5000[[:space:]]*$/  iterations: ${TARGET_ITERS}/" "$CFG4D_NEW"

# isolate ckpts
cp "$REC4C_CKPT" "$CK4C_ISO"
cp "$REC4D_CKPT" "$CK4D_ISO"

echo "[RUN] 4c resume -> $SAVE4C"
conda run -n "$ENV_NAME" python main.py --mode train --cfg_path "$CFG4C_NEW" --save_dir "$SAVE4C" --ckpt_path "$CK4C_ISO"

CUR4C_LOG=$(find "$SAVE4C" -type f -name training.log | sort | tail -n 1)
[ -f "$CUR4C_LOG" ] || { echo "[ERR] 4c training.log not found"; exit 1; }

conda run -n "$ENV_NAME" python scripts/compare_training_log_metrics.py --recovered-log "$REC4C_LOG" --current-log "$CUR4C_LOG" --output "$OUT4C"

echo "[RUN] 4d resume -> $SAVE4D"
conda run -n "$ENV_NAME" python main.py --mode train --cfg_path "$CFG4D_NEW" --save_dir "$SAVE4D" --ckpt_path "$CK4D_ISO"

CUR4D_LOG=$(find "$SAVE4D" -type f -name training.log | sort | tail -n 1)
[ -f "$CUR4D_LOG" ] || { echo "[ERR] 4d training.log not found"; exit 1; }

conda run -n "$ENV_NAME" python scripts/compare_training_log_metrics.py --recovered-log "$REC4D_LOG" --current-log "$CUR4D_LOG" --output "$OUT4D"

TARGET_ITERS="$TARGET_ITERS" python - <<'PY'
import csv
import os
from pathlib import Path

iters = os.environ.get("TARGET_ITERS", "7000")
out = Path(f"debug_output/compare_4c_4d_repro_summary_{iters}.csv")
rows = []
for tag, fp in [
    ("4c", Path(f"debug_output/compare_4c_recovered_vs_repro_{iters}.csv")),
    ("4d", Path(f"debug_output/compare_4d_recovered_vs_repro_{iters}.csv")),
]:
    with fp.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f))
    rec = next(r for r in data if r["run"] == "recovered")
    cur = next(r for r in data if r["run"] == "current")
    rows.append({
        "exp": tag,
        "recovered_best_psnr": rec["best_psnr"],
        "repro_best_psnr": cur["best_psnr"],
        "delta_best_psnr_rec_minus_repro": f"{float(rec['best_psnr']) - float(cur['best_psnr']):+.4f}",
        "recovered_last_psnr": rec["last_psnr"],
        "repro_last_psnr": cur["last_psnr"],
        "delta_last_psnr_rec_minus_repro": f"{float(rec['last_psnr']) - float(cur['last_psnr']):+.4f}",
        "repro_log": cur["log_path"],
    })

with out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"[DONE] summary -> {out}")
PY

echo "[DONE] 4c compare: $OUT4C"
echo "[DONE] 4d compare: $OUT4D"
echo "[DONE] summary   : $OUT_SUM"
