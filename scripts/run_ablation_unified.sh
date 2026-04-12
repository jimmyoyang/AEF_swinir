#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for legacy and new mainline suites.
#
# Usage examples:
#   bash scripts/run_ablation_unified.sh --list
#   bash scripts/run_ablation_unified.sh --suite legacy_v2_full --gpu 0
#   bash scripts/run_ablation_unified.sh --suite mainline_4d_all_full --gpu 0

GPU_ID=0
SUITE=""
INPUT_DIR=""
OUTPUT_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="$2"; shift 2 ;;
    --gpu)
      GPU_ID="$2"; shift 2 ;;
    --input)
      INPUT_DIR="$2"; shift 2 ;;
    --output)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --list)
      SUITE="__list__"; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

# 智能推断默认输入输出目录（如未指定）
if [[ -z "$INPUT_DIR" ]]; then
  case "$SUITE" in
    mainline_4d_all_full|mainline_4d_v2_full|mainline_4d_v3_full)
      INPUT_DIR="data/test/4d_full/LRHR" # 你实际的推理输入根目录
      ;;
    mainline_4c_all_full|mainline_4c_v2_full|mainline_4c_v3_full)
      INPUT_DIR="data/test/4c_full/LRHR"
      ;;
    *)
      INPUT_DIR="data/test/default/LRHR"
      ;;
  esac
  echo "[INFO] Auto INPUT_DIR: $INPUT_DIR"
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
  case "$SUITE" in
    mainline_4d_all_full|mainline_4d_v2_full|mainline_4d_v3_full)
      OUTPUT_ROOT="debug_output/infer_4d_full"
      ;;
    mainline_4c_all_full|mainline_4c_v2_full|mainline_4c_v3_full)
      OUTPUT_ROOT="debug_output/infer_4c_full"
      ;;
    *)
      OUTPUT_ROOT="debug_output/infer_default"
      ;;
  esac
  echo "[INFO] Auto OUTPUT_ROOT: $OUTPUT_ROOT"
fi

list_suites() {
  cat <<EOF
Available suites:
  legacy_v2_full
  legacy_v3_core_full
  legacy_v3_all_full
  mainline_4c_v2_full
  mainline_4d_v2_full
  mainline_4c_v3_full
  mainline_4d_v3_full
  mainline_4c_all_full
  mainline_4d_all_full
EOF
}

if [[ "${SUITE}" == "__list__" || -z "${SUITE}" ]]; then
  list_suites
  exit 0
fi

declare -a CFG_LIST

case "${SUITE}" in
  legacy_v2_full)
    CFG_LIST=(
      configs/ablation_v2/ablation_v2_5a_nopost_bestpre_full.yaml
      configs/ablation_v2/ablation_v2_5b_scheme1_gumbel_full.yaml
      configs/ablation_v2/ablation_v2_5c_scheme3_dual_branch_full.yaml
      configs/ablation_v2/ablation_v2_5d_scheme6_matrix_full.yaml
    )
    ;;
  legacy_v3_core_full)
    CFG_LIST=(
      configs/ablation_v3/ablation_v3_5a_nopost_bestpre_full.yaml
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_full.yaml
      configs/ablation_v3/ablation_v3_5f_scheme3_plus6_full.yaml
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_stage_full.yaml
      configs/ablation_v3/ablation_v3_5f_scheme3_plus6_stage_full.yaml
    )
    ;;
  legacy_v3_all_full)
    CFG_LIST=(
      configs/ablation_v3/ablation_v3_5a_nopost_bestpre_full.yaml
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_full.yaml
      configs/ablation_v3/ablation_v3_5f_scheme3_plus6_full.yaml
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_stage_full.yaml
      configs/ablation_v3/ablation_v3_5f_scheme3_plus6_stage_full.yaml
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_gated_lightweight_full.yaml
      configs/ablation_v3/ablation_v3_5f_scheme3_plus6_gated_lightweight_full.yaml
    )
    ;;
  mainline_4c_v2_full)
    CFG_LIST=(
      configs/ablation_mainline_4c_v2/ablation_4c_v2_5a_nopost_bestpre_full.yaml
      configs/ablation_mainline_4c_v2/ablation_4c_v2_5b_scheme1_gumbel_full.yaml
      configs/ablation_mainline_4c_v2/ablation_4c_v2_5c_scheme3_dual_branch_full.yaml
      configs/ablation_mainline_4c_v2/ablation_4c_v2_5d_scheme6_matrix_full.yaml
    )
    ;;
  mainline_4d_v2_full)
    CFG_LIST=(
      configs/ablation_mainline_4d_v2/ablation_4d_v2_5a_nopost_bestpre_full.yaml
      configs/ablation_mainline_4d_v2/ablation_4d_v2_5b_scheme1_gumbel_full.yaml
      configs/ablation_mainline_4d_v2/ablation_4d_v2_5c_scheme3_dual_branch_full.yaml
      configs/ablation_mainline_4d_v2/ablation_4d_v2_5d_scheme6_matrix_full.yaml
    )
    ;;
  mainline_4c_v3_full)
    CFG_LIST=(
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5a_nopost_bestpre_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5e_scheme1_plus6_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5f_scheme3_plus6_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5e_scheme1_plus6_stage_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5f_scheme3_plus6_stage_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5e_scheme1_plus6_gated_lightweight_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5f_scheme3_plus6_gated_lightweight_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5e_scheme1_plus6_stage_gated_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5f_scheme3_plus6_stage_gated_full.yaml
    )
    ;;
  mainline_4d_v3_full)
    CFG_LIST=(
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5a_nopost_bestpre_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5e_scheme1_plus6_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5f_scheme3_plus6_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5e_scheme1_plus6_stage_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5f_scheme3_plus6_stage_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5e_scheme1_plus6_gated_lightweight_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5f_scheme3_plus6_gated_lightweight_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5e_scheme1_plus6_stage_gated_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5f_scheme3_plus6_stage_gated_full.yaml
    )
    ;;
  mainline_4c_all_full)
    CFG_LIST=(
      configs/ablation_mainline_4c_v2/ablation_4c_v2_5a_nopost_bestpre_full.yaml
      configs/ablation_mainline_4c_v2/ablation_4c_v2_5b_scheme1_gumbel_full.yaml
      configs/ablation_mainline_4c_v2/ablation_4c_v2_5c_scheme3_dual_branch_full.yaml
      configs/ablation_mainline_4c_v2/ablation_4c_v2_5d_scheme6_matrix_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5a_nopost_bestpre_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5e_scheme1_plus6_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5f_scheme3_plus6_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5e_scheme1_plus6_stage_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5f_scheme3_plus6_stage_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5e_scheme1_plus6_gated_lightweight_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5f_scheme3_plus6_gated_lightweight_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5e_scheme1_plus6_stage_gated_full.yaml
      configs/ablation_mainline_4c_v3/ablation_4c_v3_5f_scheme3_plus6_stage_gated_full.yaml
    )
    ;;
  mainline_4d_all_full)
    CFG_LIST=(
      configs/ablation_mainline_4d_v2/ablation_4d_v2_5a_nopost_bestpre_full.yaml
      configs/ablation_mainline_4d_v2/ablation_4d_v2_5b_scheme1_gumbel_full.yaml
      configs/ablation_mainline_4d_v2/ablation_4d_v2_5c_scheme3_dual_branch_full.yaml
      configs/ablation_mainline_4d_v2/ablation_4d_v2_5d_scheme6_matrix_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5a_nopost_bestpre_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5e_scheme1_plus6_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5f_scheme3_plus6_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5e_scheme1_plus6_stage_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5f_scheme3_plus6_stage_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5e_scheme1_plus6_gated_lightweight_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5f_scheme3_plus6_gated_lightweight_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5e_scheme1_plus6_stage_gated_full.yaml
      configs/ablation_mainline_4d_v3/ablation_4d_v3_5f_scheme3_plus6_stage_gated_full.yaml
    )
    ;;
  *)
    echo "Unsupported suite: ${SUITE}" >&2
    list_suites
    exit 2
    ;;
esac

echo "[INFO] suite=${SUITE} gpu=${GPU_ID} items=${#CFG_LIST[@]}"
for cfg in "${CFG_LIST[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "[ERR] Missing config: ${cfg}" >&2
    exit 1
  fi
done

for cfg in "${CFG_LIST[@]}"; do
  EXP_NAME=$(basename "${cfg}" .yaml)
  CKPT="best_ckpts/${EXP_NAME}/model.pth"
  if [[ ! -f "$CKPT" ]]; then
    echo "[SKIP] No best_ckpt for ${EXP_NAME}"
    continue
  fi
  OUTDIR="${OUTPUT_ROOT}/${EXP_NAME}"
  mkdir -p "$OUTDIR"
  echo "[INFER] ${cfg} | ckpt: $CKPT"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python main.py --cfg_path "${cfg}" --mode test --ckpt_path "$CKPT" --input_dir "$INPUT_DIR" --output_dir "$OUTDIR"
done

echo "[OK] inference suite finished: ${SUITE}"