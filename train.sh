#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CFG_PATH="${2:-configs/ablation/aef_time_aligned_cosine_large_dequant.yaml}"
MODE="${3:-train}"
MODEL_SIZE="${4:-}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/alphaearth/bin/python}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[train.sh] PYTHON_BIN=${PYTHON_BIN} is not executable; falling back to python"
  PYTHON_BIN="python"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[train.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if [[ -n "${MODEL_SIZE}" ]]; then
  echo "[train.sh] MODEL_SIZE=${MODEL_SIZE}"
fi
"${PYTHON_BIN}" -c "import torch; print('[train.sh] python=', '${PYTHON_BIN}', 'cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count(), 'current=', torch.cuda.current_device() if torch.cuda.is_available() else None)"

if [[ "${MODE}" == "train" && "${RUN_SPLIT_DIAG:-0}" == "1" ]]; then
  echo "[train.sh] CPU split diagnostics: python scripts/diagnose_dataset_splits.py --cfg_path ${CFG_PATH}"
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" scripts/diagnose_dataset_splits.py --cfg_path "${CFG_PATH}" --max_samples "${SPLIT_DIAG_MAX_SAMPLES:-8}" || true
elif [[ "${MODE}" == "train" ]]; then
  echo "[train.sh] CPU split diagnostics skipped (set RUN_SPLIT_DIAG=1 to enable)"
fi

CMD=("${PYTHON_BIN}" main.py --cfg_path "${CFG_PATH}" --mode "${MODE}")
if [[ -n "${MODEL_SIZE}" ]]; then
  CMD+=(--model_size "${MODEL_SIZE}")
fi

echo "[train.sh] Run: ${CMD[*]}"
"${CMD[@]}"
