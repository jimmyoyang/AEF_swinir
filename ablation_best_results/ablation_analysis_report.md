# 消融实验结果分析报告（最新版）

生成时间：2026-03-09 01:34:41

## 1. 全量结果

| Group | Experiment | Label | Best Iter | PSNR | SSIM |
|---|---|---|---:|---:|---:|
| Group 1 | config_true_baseline | 1a true baseline | 400 | 12.2797 | 0.2647 |
| Group 1 | ablation_1b_baseline_plus_timeband | 1b +time | 600 | 14.1225 | 0.3563 |
| Group 1 | ablation_1c_baseline_plus_maskband | 1c +mask(hard) | 0 | 12.1931 | 0.2217 |
| Group 1 | ablation_1d_baseline_plus_timeband_maskband | 1d +time+mask | 300 | 14.2657 | 0.3852 |
| Group 2 | ablation_2b_with_cross_attention_no_posenc | 2b +cross | 500 | 14.4730 | 0.3586 |
| Group 2 | ablation_2c_cross_downsample_rate_2 | 2c cross_downsample=2 | 600 | 14.5803 | 0.3840 |
| Group 3 | ablation_3a_with_cross_attention_posenc_sincos | 3a +pos(sincos) | 100 | 13.5138 | 0.3523 |
| Group 3 | ablation_3b_with_cross_attention_posenc_learnable | 3b +pos(learnable) | 100 | 14.6694 | 0.3556 |
| Group 3 | ablation_3c_with_cross_attention_posenc_concat | 3c +pos(concat) | 800 | 14.5684 | 0.3710 |
| Group 3 | ablation_3d_posenc_without_cross_attention | 3d +pos(no-cross) | 100 | 13.3887 | 0.3540 |
| Group 4 | ablation_4a_soft_mask_test | 4a soft-mask test | 500 | 14.4990 | 0.3585 |
| Group 4 | ablation_4b_advanced_processor_soft_simplified | 4b soft + advanced off | 100 | 14.5181 | 0.3538 |
| Unknown | ablation_4c_mask_reduce_prob_or | ablation_4c_mask_reduce_prob_or | 600 | 14.5117 | 0.3553 |

## 2. 关键增益

- Δtime (1b - baseline): +1.8428 dB
- Δmask_hard (1c - baseline): -0.0866 dB
- Δcross (2b - 1d): +0.2073 dB
- Δsoft (4a_test - 2b): +0.0260 dB

## 3. 最优配置

- Name: ablation_3b_with_cross_attention_posenc_learnable
- Group: Group 3
- Label: 3b +pos(learnable)
- PSNR: 14.6694
- SSIM: 0.3556
- Best Iter: 100
