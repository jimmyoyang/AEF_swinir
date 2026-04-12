#!/usr/bin/env bash
set -euo pipefail
GPU_ID=${1:-0}
CFG_PATH="configs/ablation/ablation_4c_mask_reduce_prob_or.yaml"

echo "[INFO] Running ablation_4c on GPU ${GPU_ID}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python main.py --cfg_path "${CFG_PATH}" --mode train
