# Codex Change Notes

## 2026-06-12: AlphaEarth visualization channels

- Changed `configs/ablation/aef_time_aligned_cosine_large_dequant.yaml`:
  - `train.rgb_chn` from `[2, 1, 0]` to `[17, 31, 22]`.
- Purpose:
  - Validation visualization uses only three channels, not all 64 AlphaEarth embedding bands.
  - The old `[2, 1, 0]` pseudo-RGB made HR/Prediction panels look too similar.
  - A quick train-set sample audit found `[17, 31, 22]` had higher robust dynamic range and low pairwise correlation, so it should show clearer AlphaEarth embedding differences.
- Revert:
  - Set `train.rgb_chn` back to `[2, 1, 0]` in `configs/ablation/aef_time_aligned_cosine_large_dequant.yaml`.
- Note:
  - This changes visualization only. It does not change model inputs, targets, loss, metrics, cache, or training data.
