#!/usr/bin/env bash
set -euo pipefail

# Quick-only launcher (separate from unified).
#
# Usage examples:
#   bash scripts/run_ablation_quick.sh --list
#   bash scripts/run_ablation_quick.sh --suite v2_quick --gpu 0
#   bash scripts/run_ablation_quick.sh --suite v3_all_quick --gpu 0

GPU_ID=0
SUITE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="$2"; shift 2 ;;
    --gpu)
      GPU_ID="$2"; shift 2 ;;
    --list)
      SUITE="__list__"; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

list_suites() {
  cat <<EOF
Available quick suites (with legacy mapping):
  v2_quick
    - 对应旧脚本: scripts/run_ablation_5bcd_v2_quick.sh
    - 对应对比: V2 quick 5a/5b/5c/5d 对比

  v3_core_quick
    - 对应旧脚本: scripts/run_ablation_v3_quick.sh
    - 对应对比: V3 quick 5a/5e/5f core 对比

  v3_stage_quick
    - 对应旧脚本: scripts/run_ablation_v3_stage_quick.sh
    - 对应对比: V3 quick stage(5e_stage/5f_stage) 对比

  v3_stage_gated_quick
    - 对应旧脚本: scripts/run_ablation_v3_stage_gated_quick.sh
    - 对应对比: V3 quick stage_gated(5e/5f) 对比

  v3_all_quick
    - 对应旧脚本: 无单一完全等价（覆盖上面 3 组 + gated_lightweight_quick）
    - 对应对比: V3 quick 全量对比(5a/5e/5f + stage + stage_gated + gated_lightweight)
EOF
}

if [[ "${SUITE}" == "__list__" || -z "${SUITE}" ]]; then
  list_suites
  exit 0
fi

declare -a CFG_LIST

case "${SUITE}" in
  v2_quick)
    CFG_LIST=(
      configs/ablation_v2/ablation_v2_5a_nopost_bestpre_quick.yaml
      configs/ablation_v2/ablation_v2_5b_scheme1_gumbel_quick.yaml
      configs/ablation_v2/ablation_v2_5c_scheme3_dual_branch_quick.yaml
      configs/ablation_v2/ablation_v2_5d_scheme6_matrix_quick.yaml
    )
    ;;
  v3_core_quick)
    CFG_LIST=(
      configs/ablation_v3/ablation_v3_5a_nopost_bestpre_quick.yaml
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_quick.yaml
      configs/ablation_v3/ablation_v3_5f_scheme3_plus6_quick.yaml
    )
    ;;
  v3_stage_quick)
    CFG_LIST=(
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_stage_quick.yaml
      configs/ablation_v3/ablation_v3_5f_scheme3_plus6_stage_quick.yaml
    )
    ;;
  v3_stage_gated_quick)
    CFG_LIST=(
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_stage_gated_quick.yaml
      configs/ablation_v3/ablation_v3_5f_scheme3_plus6_stage_gated_quick.yaml
    )
    ;;
  v3_all_quick)
    CFG_LIST=(
      configs/ablation_v3/ablation_v3_5a_nopost_bestpre_quick.yaml
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_quick.yaml
      configs/ablation_v3/ablation_v3_5f_scheme3_plus6_quick.yaml
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_stage_quick.yaml
      configs/ablation_v3/ablation_v3_5f_scheme3_plus6_stage_quick.yaml
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_stage_gated_quick.yaml
      configs/ablation_v3/ablation_v3_5f_scheme3_plus6_stage_gated_quick.yaml
      configs/ablation_v3/ablation_v3_5e_scheme1_plus6_gated_lightweight_quick.yaml
    )
    ;;
  *)
    echo "Unsupported quick suite: ${SUITE}" >&2
    list_suites
    exit 2
    ;;
esac

echo "[INFO] quick suite=${SUITE} gpu=${GPU_ID} items=${#CFG_LIST[@]}"
for cfg in "${CFG_LIST[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "[ERR] Missing config: ${cfg}" >&2
    exit 1
  fi
done

for cfg in "${CFG_LIST[@]}"; do
  echo "[RUN] ${cfg}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python main.py --cfg_path "${cfg}" --mode train
  echo "[DONE] ${cfg}"
done

echo "[OK] quick suite finished: ${SUITE}"
