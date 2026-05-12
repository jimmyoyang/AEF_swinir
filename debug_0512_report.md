# 2026-05-12 Debug 日志汇报

## 1. 背景与问题

本次 debug 针对 `SPLIT_DIAG_MAX_SAMPLES=8 bash train.sh` 及后续重启训练过程中暴露出的两个核心问题：

1. 验证阶段很慢，GPU 利用率偏低，cache 看起来没有充分发挥作用。
2. 验证集 / 测试集 PSNR 在早期实验中存在下降或不稳定现象，需要确认是实现错误、数据量问题，还是正常训练震荡。

最终复跑日志位于：

- `debug_0512.log`
- 实验目录：`training_logs/experiments/ablation_4f_mask_mean_learnable_pos_no_cross/2026-05-12_04-35-27`

本轮训练已正常结束，日志末尾为：

```text
Training Finished Successfully!
```

未发现 `Traceback`、`ERROR`、`NaN` 或 `non-finite`。

## 2. 定位到的主要问题

### 2.1 LR 输入通道包含非反射率 band

处理后的 LR tile 中存在一个尾部 QA / cloud-mask-like band。旧逻辑会把它当作物理反射率一起输入模型，并且在 hard valid mask 计算时也可能参与 `all(bands > 0)` 判断。

这会带来两个影响：

1. 模型输入通道数偏大，实际为 `10 reflectance + time + mask = 12`。
2. QA / mask 类 band 被重复注入到模型输入，且可能把本应有效的像素 / 时相错误判为无效。

本次修正后，明确只使用前 9 个反射率 band：

```text
Reflectance Input Bands: first 9
Dynamically detected in_chans: 11
```

即当前输入为：

```text
9 reflectance + 1 time band + 1 mask band = 11 channels
```

相关修改：

- `datapipe/datasets.py`
  - `_infer_valid_mask_band_count`
  - `_compute_hard_valid_mask`
  - `_select_reflectance_bands`
- `configs/ablation/ablation_4f_mask_prob_or_learnable_pos_no_cross.yaml`
  - `valid_mask_band_count: 9`
  - `reflectance_band_count: 9`

### 2.2 验证阶段 CPU 指标计算过重

旧验证流程会在每次 val 中计算 SSIM / ERGAS / SAM，并做大量 GPU tensor 到 CPU numpy 的转换。对于 `64 bands x 64 val samples` 的设置，这部分开销远大于模型 forward，导致验证阶段 GPU 利用率低、整体训练被 val 拖慢。

本次修正后：

1. 训练过程中的调度和 best checkpoint 只依赖 PSNR。
2. PSNR / Masked-PSNR 改为 GPU batch 快路径计算。
3. SSIM / ERGAS / SAM 默认跳过，仅在需要离线评估时再打开。
4. 增加 validation timing 日志，拆分 `data_wait / to_device / forward / metrics / visualize`。

日志确认：

```text
val_compute_expensive_metrics: false
Validation Metrics | PSNR: ... | SSIM/ERGAS/SAM: skipped | Masked-PSNR: ...
```

相关修改：

- `trainer.py`
  - `_compute_fast_validation_metrics`
  - validation 中使用 GPU 快速 PSNR / Masked-PSNR
  - `val_compute_expensive_metrics`
  - `val_timing_log`

### 2.3 cache 只覆盖 tile，不覆盖全部重复处理

原先虽然启用了 tile cache，但验证 / 测试阶段仍存在重复处理：

1. hard mask 可能需要重复从原始 LR tif 读取和计算。
2. soft mask / advanced mask processor 结果可能重复生成。
3. HR normalize 结果可能重复生成。
4. val/test 每次验证会重复组装同一批 processed sample。

本次增加了多层 cache：

1. Tile cache：继续使用已有 `cache_dir`。
2. Hard mask sidecar cache：缓存修正后的 hard valid mask。
3. Mask sidecar cache：缓存 soft / advanced mask 结果。
4. HR cache：缓存 HR 归一化结果。
5. Processed sample LRU cache：val/test 使用 `sample_cache_mem_size: 64`，直接缓存完整 processed sample。

日志确认：

```text
Tile Cache: Active
Hard Mask Sidecar Cache: Enabled
Mask Sidecar Cache: Enabled
HR Cache: Enabled
Processed Sample Cache: Enabled (LRU size=64)
```

相关修改：

- `datapipe/datasets.py`
  - `_load_or_build_hard_mask_cache`
  - `_load_or_build_mask_cache`
  - HR cache
  - processed sample LRU cache
  - cache key 中纳入 mask processor / valid band 配置，避免复用错误 cache
- `configs/ablation/ablation_4f_mask_prob_or_learnable_pos_no_cross.yaml`
  - `mask_cache_dir`
  - `hard_mask_cache_dir`
  - `hr_cache_dir`
  - val/test `sample_cache_mem_size: 64`

### 2.4 小样本实验的泛化判断不可靠

最初配置中训练样本很少，容易出现记忆小样本或验证集不稳定。后续配置改为更合理的数据规模：

```yaml
train:
  pre_filter_sample_num: 4096
  sample_num: 1024
val:
  pre_filter_sample_num: 512
  sample_num: 64
test:
  pre_filter_sample_num: 512
  sample_num: 64
```

并采用 deterministic random subset：

```yaml
pre_filter_seed: 42 / 43 / 44
```

这样避免只取排序靠前的少量 tile，使 PSNR 趋势更能代表真实训练效果。

### 2.5 mask 聚合策略过强

旧策略 `prob_or` 在多时相下容易把空间 mask 推近 1，即几乎所有像素都有效。日志中也能看到 `mask_gt_0.5` 经常接近 `1.0000`。

本次改为：

```yaml
indicating_mask_reduce: "mean"
```

并在 trainer 中统一通过 `_reduce_temporal_mask` 处理训练和验证 mask，避免训练 / 验证 mask 聚合逻辑不一致。

## 3. 关键配置变更

当前主要训练配置为：

```yaml
train:
  save_dir: "./training_logs/experiments/ablation_4f_mask_mean_learnable_pos_no_cross"
  batch: [4, 8]
  iterations: 20000
  lr: 0.00005
  lr_schedule: "ReduceLROnPlateau"
  lr_schedule_params:
    patience: 2
    factor: 0.5
    threshold: 0.01
    threshold_mode: abs
    min_lr: 1.0e-6
  weight_decay: 0.0001
  num_workers: 4
  val_num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  save_freq: 500
  val_freq: 500
  indicating_mask_reduce: "mean"
  val_compute_expensive_metrics: false
  val_timing_log: true
```

数据侧关键配置：

```yaml
valid_mask_band_count: 9
reflectance_band_count: 9
use_tile_cache: true
sample_cache_mem_size: 64  # val/test
```

## 4. 验证阶段性能结果

本次训练共执行 40 次验证，每 500 iter 一次。

Validation timing 统计：

```text
count = 40
min   = 0.81s
max   = 1.11s
mean  = 0.884s
```

最终一次验证：

```text
Validation timing | data_wait=0.23s to_device=0.02s forward=0.31s metrics=0.01s visualize=0.36s total=0.93s
```

可以看到：

1. `metrics` 只占约 `0.01s`，CPU 慢指标瓶颈已消除。
2. `forward` 稳定约 `0.31s`。
3. `visualize` 约 `0.35-0.37s`，已经成为 val 中相对明显的一项，但总耗时仍低于 1s 左右。
4. `data_wait` 通常在 `0.11-0.25s`，val/test cache 和多 worker 生效。

结论：验证阶段低 GPU 利用率主要是因为验证本身已非常短，而不是卡在慢数据读取或慢指标计算上。相比修复前约分钟级验证耗时，本轮验证瓶颈已基本解决。

## 5. 训练效果

PSNR 从首个验证点持续提升，并在训练后半程进入稳定微涨区间。

整体统计：

```text
val_count       = 40
first_psnr      = 13.4372 @ iter 500
best_psnr       = 14.2786 @ iter 18000
final_psnr      = 14.2695 @ iter 20000
first_masked    = 13.4374
best_masked     = 14.2787
final_masked    = 14.2696
best_gain       = +0.8414 dB
final_gain      = +0.8323 dB
```

主要 PSNR 轨迹：

```text
500    13.4372
1000   13.5965
2500   13.7687
5000   14.0515
8500   14.1628
11000  14.2377
13000  14.2502
14000  14.2662
16500  14.2753
18000  14.2786  <-- best
20000  14.2695
```

训练结束时：

```text
Iter 20000 | Train Loss: 0.912960 | LR: 1.563e-06
Validation Metrics | PSNR: 14.2695 | SSIM/ERGAS/SAM: skipped | Masked-PSNR: 14.2696
Training Finished Successfully!
```

LR 调度表现正常：

1. 初始 LR 为 `5.000e-05`。
2. plateau 后逐步降至 `2.500e-05`、`1.250e-05`、`6.250e-06`、`3.125e-06`。
3. 最终 LR 为 `1.563e-06`。
4. 降 LR 后仍多次刷新 best，说明 ReduceLROnPlateau 调度对后期收敛有帮助。

## 6. 产物位置

Best checkpoint：

```text
best_ckpts/ablation_4f_mask_mean_learnable_pos_no_cross/model.pth
```

Run 内 best checkpoint：

```text
training_logs/experiments/ablation_4f_mask_mean_learnable_pos_no_cross/2026-05-12_04-35-27/ckpts/model_best.pth
```

最终 checkpoint：

```text
training_logs/experiments/ablation_4f_mask_mean_learnable_pos_no_cross/2026-05-12_04-35-27/ckpts/model_20000.pth
```

训练曲线：

```text
training_logs/experiments/ablation_4f_mask_mean_learnable_pos_no_cross/2026-05-12_04-35-27/training_curves.png
```

checkpoint 目录大小约：

```text
7.2G  training_logs/.../ckpts
179M  best_ckpts/ablation_4f_mask_mean_learnable_pos_no_cross
```

## 7. 最终结论

本次 debug 后没有发现新的训练稳定性问题。主要问题已经定位并修正：

1. 修正 LR 输入 band：只使用前 9 个反射率 band，避免 QA / cloud-like band 被当作反射率输入。
2. 修正 hard valid mask band 选择，避免尾部 mask-like band 影响有效像素判断。
3. 增加 sidecar cache 和 processed sample cache，val/test cache 命中后验证数据准备显著加速。
4. 将验证指标改为 GPU batch PSNR / Masked-PSNR 快路径，跳过训练中不必要的 SSIM / ERGAS / SAM。
5. 增加 validation timing 和 sample-level debug，后续能快速区分 data wait、forward、metrics、visualize 的耗时。
6. 扩大训练 / 验证数据量并使用 deterministic random subset，使 PSNR 趋势更可靠。
7. 将 mask 聚合从 `prob_or` 调整为 `mean`，减少多时相 mask 过饱和的问题。

实验效果上，验证 PSNR 从 `13.4372` 提升到 best `14.2786`，最终 `14.2695`，整体提升约 `+0.84 dB`。验证阶段平均耗时约 `0.884s`，没有再出现明显卡顿。训练最终干净结束，无 NaN、无报错。

## 8. 后续建议

1. 如果需要正式论文 / 表格指标，可在 best checkpoint 上单独跑一次离线评估，打开 SSIM / ERGAS / SAM，而不是在训练中反复计算。
2. 当前 val 中 `visualize` 约占 `0.35s`，若未来 val 数据量更大，可以降低可视化频率或只在 best checkpoint 时可视化。
3. checkpoint 每 500 iter 保存一次，本轮 ckpt 目录约 `7.2G`；长期跑多组实验时建议增加自动清理策略，只保留 best、last 和少量里程碑 checkpoint。
4. 当前 best 出现在 `18000`，后续如果继续训练，可考虑从 best checkpoint 恢复，使用更小 LR 做短程 fine-tune。
