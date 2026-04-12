#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:-0}

CFG_LIST=(
  configs/ablation/config_true_baseline.yaml
  configs/ablation/config_srcnn_fair.yaml
)

echo "[INFO] Fair comparison mode | GPU=$GPU_ID"
for cfg in "${CFG_LIST[@]}"; do
  echo "[RUN] $cfg"
  if [[ "$cfg" == "configs/ablation/config_srcnn_fair.yaml" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" python main_srcnn.py --cfg_path "$cfg" --mode train
  else
    CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py --cfg_path "$cfg" --mode train
  fi
  echo "[DONE] $cfg"
done

echo "[OK] Fair comparison training finished."
