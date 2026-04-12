#!/usr/bin/env bash
set -euo pipefail
GPU_ID=${1:-0}
ITER_OVERRIDE=${2:-}

CFG_LIST=(
  configs/config_srcnn.yaml
  configs/ablation/config_true_baseline.yaml # SWINIR baseline
  configs/ablation/ablation_3b_with_cross_attention_posenc_learnable.yaml
  configs/ablation/ablation_4c_mask_reduce_prob_or.yaml
  configs/ablation/ablation_4f_mask_prob_or_learnable_pos_no_cross.yaml
  configs/ablation/ablation_4e_mask_prob_or_learnable_pos.yaml
  configs/ablation/ablation_4d_mask_temporal_loss.yaml
)

echo "[INFO] GPU=$GPU_ID"
for cfg in "${CFG_LIST[@]}"; do
  echo "[RUN] $cfg"
  if [[ "$cfg" == "configs/config_srcnn.yaml" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" python main_srcnn.py --cfg_path "$cfg" --mode train
  else
    CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py --cfg_path "$cfg" --mode train
  fi
  echo "[DONE] $cfg"
done
