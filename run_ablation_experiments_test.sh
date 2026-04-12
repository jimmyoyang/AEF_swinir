#!/usr/bin/env bash
set -euo pipefail
GPU_ID=${1:-0}
ITER_OVERRIDE=${2:-}

CFG_LIST=(
  configs/config_srcnn.yaml
#   configs/ablation/config_true_baseline.yaml # SWINIR baseline
)

echo "[INFO] GPU=$GPU_ID"
for cfg in "${CFG_LIST[@]}"; do
  echo "[RUN] $cfg"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py --cfg_path "$cfg" --mode train
  echo "[DONE] $cfg"
done
