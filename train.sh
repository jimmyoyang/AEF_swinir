#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CFG_PATH="${2:-configs/config_swinir.yaml}"
MODE="${3:-train}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[train.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python -c "import torch; print('[train.sh] cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count(), 'current=', torch.cuda.current_device() if torch.cuda.is_available() else None)"

echo "[train.sh] Run: python main.py --cfg_path ${CFG_PATH} --mode ${MODE}"
python main.py --cfg_path "${CFG_PATH}" --mode "${MODE}"