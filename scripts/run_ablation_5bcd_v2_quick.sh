#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:-0}

CFG_LIST=(
  configs/ablation_v2/ablation_v2_5a_nopost_bestpre_quick.yaml
  configs/ablation_v2/ablation_v2_5b_scheme1_gumbel_quick.yaml
  configs/ablation_v2/ablation_v2_5c_scheme3_dual_branch_quick.yaml
  configs/ablation_v2/ablation_v2_5d_scheme6_matrix_quick.yaml
)

echo "[INFO] quick run on GPU=${GPU_ID}"
for cfg in "${CFG_LIST[@]}"; do
  echo "[RUN] ${cfg}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python main.py --cfg_path "${cfg}" --mode train
  echo "[DONE] ${cfg}"
done

echo "[INFO] quick ablation_v2 done"
