#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:-0}

CFG_LIST=(
  configs/ablation_v3/ablation_v3_5a_nopost_bestpre_quick.yaml
  configs/ablation_v3/ablation_v3_5e_scheme1_plus6_quick.yaml
  configs/ablation_v3/ablation_v3_5f_scheme3_plus6_quick.yaml
)

echo "[INFO] v3 quick run on GPU=${GPU_ID}"
for cfg in "${CFG_LIST[@]}"; do
  echo "[RUN] ${cfg}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python main.py --cfg_path "${cfg}" --mode train
  echo "[DONE] ${cfg}"
done

echo "[INFO] v3 quick ablation done"
