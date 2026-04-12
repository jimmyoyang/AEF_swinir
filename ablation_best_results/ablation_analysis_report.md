# 消融实验分析报告

**生成时间**: 2026-03-16 17:58:44  
**实验总数**: 25  
**基准实验**: `config_true_baseline`  (PSNR=12.2797 dB)

---

## 排行榜（按 Best PSNR）

| Rank | 实验名称 | Group | PSNR (dB) | SSIM | Best Iter | vs Baseline |
|------|---------|-------|-----------|------|-----------|-------------|
| 🥇 | `ablation_4d_mask_temporal_loss` | G4-软掩膜 | **14.7357** | 0.3380 | 600 | +2.4560 dB |
| 🥈 | `ablation_4c_mask_reduce_prob_or` | G4-软掩膜 | **14.7357** | 0.3288 | 1000 | +2.4560 dB |
| 🥉 | `ablation_3b_with_cross_attention_posenc_learnable` | G3-位置编码 | **14.6694** | 0.3556 | 100 | +2.3897 dB |
| 4 | `exp_cross` | 其他 | **14.6533** | 0.3564 | 100 | +2.3736 dB |
| 5 | `ablation_2c_cross_downsample_rate_2` | G2-交叉注意力 | **14.5803** | 0.3840 | 600 | +2.3006 dB |
| 6 | `ablation_3c_with_cross_attention_posenc_concat` | G3-位置编码 | **14.5684** | 0.3710 | 800 | +2.2887 dB |
| 7 | `ablation_4b_advanced_processor_soft_simplified` | G4-软掩膜 | **14.5181** | 0.3538 | 100 | +2.2384 dB |
| 8 | `ablation_4a_soft_mask_test` | G4-软掩膜 | **14.4990** | 0.3585 | 500 | +2.2193 dB |
| 9 | `exp_cross_learnable` | 其他 | **14.4857** | 0.3554 | 100 | +2.2060 dB |
| 10 | `ablation_2b_with_cross_attention_no_posenc` | G2-交叉注意力 | **14.4730** | 0.3586 | 500 | +2.1933 dB |
| 11 | `ablation_1d_baseline_plus_timeband_maskband` | G1-基线 | **14.2657** | 0.3852 | 300 | +1.9860 dB |
| 12 | `exp_baseline` | 其他 | **14.2278** | 0.3861 | 300 | +1.9481 dB |
| 13 | `exp_baseline_20260303_051931` | 其他 | **14.2031** | 0.3566 | 100 | +1.9234 dB |
| 14 | `exp_cross_concat_20260303_051931` | 其他 | **14.1848** | 0.3568 | 100 | +1.9051 dB |
| 15 | `exp_cross_concat` | 其他 | **14.1805** | 0.3568 | 100 | +1.9008 dB |
| 16 | `ablation_1b_baseline_plus_timeband` | G1-基线 | **14.1225** | 0.3563 | 600 | +1.8428 dB |
| 17 | `exp_cross_learnable_20260303_051931` | 其他 | **13.8090** | 0.2309 | 0 | +1.5293 dB |
| 18 | `exp_cross_20260303_051931` | 其他 | **13.8084** | 0.2266 | 0 | +1.5287 dB |
| 19 | `exp_cross_sincos` | 其他 | **13.5588** | 0.3526 | 100 | +1.2791 dB |
| 20 | `ablation_3a_with_cross_attention_posenc_sincos` | G3-位置编码 | **13.5138** | 0.3523 | 100 | +1.2341 dB |
| 21 | `ablation_3d_posenc_without_cross_attention` | G3-位置编码 | **13.3887** | 0.3540 | 100 | +1.1090 dB |
| 22 | `config_true_baseline` | G1-基线 | **12.2797** | 0.2647 | 400 | +0.0000 dB |
| 23 | `ablation_1c_baseline_plus_maskband` | G1-基线 | **12.1931** | 0.2217 | 0 | -0.0866 dB |
| 24 | `ablation_4a_with_cross_attention_softmask` | G4-软掩膜 | **N/A** | N/A | 0 | - |
| 25 | `exp_cross_sincos_20260303_051931` | 其他 | **N/A** | N/A | 0 | - |

---

## 分组详细结果

### G1-基线

| 实验名称 | PSNR | SSIM | ERGAS | SAM | Best Iter | vs Baseline |
|---------|------|------|-------|-----|-----------|-------------|
| `ablation_1d_baseline_plus_timeband_maskband` | 14.2657 | 0.3852 | 11.73 | 0.3551 | 300 | +1.9860 |
| `ablation_1b_baseline_plus_timeband` | 14.1225 | 0.3563 | 11.91 | 0.3560 | 600 | +1.8428 |
| `config_true_baseline` | 12.2797 | 0.2647 | 14.80 | 0.4265 | 400 | +0.0000 |
| `ablation_1c_baseline_plus_maskband` | 12.1931 | 0.2217 | 15.12 | 0.4340 | 0 | -0.0866 |

### G2-交叉注意力

| 实验名称 | PSNR | SSIM | ERGAS | SAM | Best Iter | vs Baseline |
|---------|------|------|-------|-----|-----------|-------------|
| `ablation_2c_cross_downsample_rate_2` | 14.5803 | 0.3840 | 11.11 | 0.3481 | 600 | +2.3006 |
| `ablation_2b_with_cross_attention_no_posenc` | 14.4730 | 0.3586 | 11.26 | 0.3540 | 500 | +2.1933 |

### G3-位置编码

| 实验名称 | PSNR | SSIM | ERGAS | SAM | Best Iter | vs Baseline |
|---------|------|------|-------|-----|-----------|-------------|
| `ablation_3b_with_cross_attention_posenc_learnable` | 14.6694 | 0.3556 | 10.89 | 0.3501 | 100 | +2.3897 |
| `ablation_3c_with_cross_attention_posenc_concat` | 14.5684 | 0.3710 | 11.00 | 0.3488 | 800 | +2.2887 |
| `ablation_3a_with_cross_attention_posenc_sincos` | 13.5138 | 0.3523 | 12.28 | 0.3938 | 100 | +1.2341 |
| `ablation_3d_posenc_without_cross_attention` | 13.3887 | 0.3540 | 12.67 | 0.3905 | 100 | +1.1090 |

### G4-软掩膜

| 实验名称 | PSNR | SSIM | ERGAS | SAM | Best Iter | vs Baseline |
|---------|------|------|-------|-----|-----------|-------------|
| `ablation_4d_mask_temporal_loss` | 14.7357 | 0.3380 | 10.76 | 0.3428 | 600 | +2.4560 |
| `ablation_4c_mask_reduce_prob_or` | 14.7357 | 0.3288 | 10.77 | 0.3439 | 1000 | +2.4560 |
| `ablation_4b_advanced_processor_soft_simplified` | 14.5181 | 0.3538 | 11.05 | 0.3529 | 100 | +2.2384 |
| `ablation_4a_soft_mask_test` | 14.4990 | 0.3585 | 11.20 | 0.3544 | 500 | +2.2193 |
| `ablation_4a_with_cross_attention_softmask` | -1.0000 | -1.0000 | N/A | N/A | 0 | - |

### 其他

| 实验名称 | PSNR | SSIM | ERGAS | SAM | Best Iter | vs Baseline |
|---------|------|------|-------|-----|-----------|-------------|
| `exp_cross` | 14.6533 | 0.3564 | 10.92 | 0.3503 | 100 | +2.3736 |
| `exp_cross_learnable` | 14.4857 | 0.3554 | 11.14 | 0.3526 | 100 | +2.2060 |
| `exp_baseline` | 14.2278 | 0.3861 | 11.69 | 0.3553 | 300 | +1.9481 |
| `exp_baseline_20260303_051931` | 14.2031 | 0.3566 | 11.58 | 0.3540 | 100 | +1.9234 |
| `exp_cross_concat_20260303_051931` | 14.1848 | 0.3568 | 11.72 | 0.3528 | 100 | +1.9051 |
| `exp_cross_concat` | 14.1805 | 0.3568 | 11.73 | 0.3530 | 100 | +1.9008 |
| `exp_cross_learnable_20260303_051931` | 13.8090 | 0.2309 | 12.07 | 0.3737 | 0 | +1.5293 |
| `exp_cross_20260303_051931` | 13.8084 | 0.2266 | 12.06 | 0.3736 | 0 | +1.5287 |
| `exp_cross_sincos` | 13.5588 | 0.3526 | 12.17 | 0.3936 | 100 | +1.2791 |
| `exp_cross_sincos_20260303_051931` | -1.0000 | -1.0000 | N/A | N/A | 0 | - |

---

## 4c vs 4d 单变量结论

- 4c vs 4d（单变量：mask 聚合策略→时相级损失）: 4d 与 4c 在 PSNR 持平，但 4d 在 SSIM 略优 | ΔPSNR=+0.0000 dB, ΔSSIM=+0.0092 | 判定: 无明显差异

---

## 手工复核结论（2026-03-17）

- 复核依据：直接检查 4c 与 4d 的最新 `training.log`（不依赖汇总脚本）。
- 4c 日志路径：`training_logs/experiments/ablation_4c_mask_reduce_prob_or/2026-03-16_16-20-28/training.log`
- 4d 日志路径：`training_logs/experiments/ablation_4d_mask_temporal_loss/2026-03-16_16-48-21/training.log`
- 手工结论：两者 Best PSNR 同为 14.7357（并列）；4d 的 Best SSIM 更高（0.3380 vs 0.3288），且后期验证平台更稳定。
- 对外结论建议：`PSNR 并列，4d 在 SSIM/稳定性上略优`。
- 说明：若仅按 PSNR 排名会产生并列顺序偏置，当前分析代码已改为 `(PSNR, SSIM)` 排序以避免“4c 单独第一”误读。