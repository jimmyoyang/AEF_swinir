#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline:
# 1) Train 4c
# 2) Train 4d
# 3) Auto analyze and print 4c vs 4d conclusion
#
# Usage:
#   bash scripts/run_ablation_4c_4d_and_analyze.sh [GPU_ID]
# Example:
#   bash scripts/run_ablation_4c_4d_and_analyze.sh 0

GPU_ID=${1:-0}
TS=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="tmp/recovery_validation_logs/relay_4c_4d_${TS}"
mkdir -p "${LOG_DIR}"

echo "[INFO] Relay run starts at ${TS} on GPU ${GPU_ID}"
echo "[INFO] Logs: ${LOG_DIR}"

echo "[STEP 1/4] Running 4c: ablation_4c_mask_reduce_prob_or"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python main.py \
  --cfg_path configs/ablation/ablation_4c_mask_reduce_prob_or.yaml \
  --mode train | tee "${LOG_DIR}/train_4c.log"

echo "[STEP 2/4] Running 4d: ablation_4d_mask_temporal_loss"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python main.py \
  --cfg_path configs/ablation/ablation_4d_mask_temporal_loss.yaml \
  --mode train | tee "${LOG_DIR}/train_4d.log"

echo "[STEP 3/4] Running ablation result analysis"
python scripts/analyze_ablation_results.py --plot | tee "${LOG_DIR}/analyze_results.log"

echo "[STEP 4/4] Running final report analysis"
python scripts/analyze_ablation_final.py --plot --curves | tee "${LOG_DIR}/analyze_final.log"

echo "[DONE] Relay finished."
echo "[DONE] Key outputs:"
echo "  - ablation_summary_latest.csv"
echo "  - ablation_best_results/ablation_analysis_report.md"
echo "  - ablation_best_results/psnr_bar_chart.png"
echo "  - ablation_best_results/learning_curves.png"
echo "  - ${LOG_DIR}/"
