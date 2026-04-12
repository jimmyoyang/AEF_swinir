# Recovery Validation Checklist (Skeleton First)

目标：先确认主链路代码可恢复、可运行、接口不丢失；再考虑优化。

## A. Entry and Routing

- [x] `main.py` 训练入口可达（mode=train）
  - 代码锚点: `run_training()` + `trainer_cls = get_obj_from_str(configs.trainer.target)`
  - 通过标准: 能按配置实例化 `trainer.TrainerAlphaSR` 或 `trainer_mask_ablation.TrainerAlphaSRMaskAblation`

- [x] `main.py` 推理入口可达（mode=test）
  - 代码锚点: `predictor = Predictor(args.cfg_path, args.ckpt_path)`
  - 通过标准: 能创建 test dataset + dataloader + 调用 `predictor.run_inference(...)`

## B. Dataset Skeleton

- [x] `datapipe/datasets.py` 的 `AnytimeTemporalDataset` 存在且返回关键字段
  - 必备字段: `lr_sequence`, `timestamps`, `gt`, `mask`
  - 条件字段（开启 mask_band 时）: `mask_prob`, `indicating_mask`

- [x] `create_dataset(configs, parent_configs)` 存在并传递 parent configs
  - 通过标准: 特征开关 `features.*` 在 dataset 内可见

## C. Cloud Mask Processor (Conditional Use)

- [x] `utils/cloud_mask_processor.py` 存在并有主入口 `process_pixel_mask`
- [x] 兼容旧方法名 `_process_hard_mask`, `_process_soft_mask` 可调用
- [x] 仅在 `features.mask_band.use_advanced_processor=true` 时启用
  - 通过标准: 4a/4b/4c 配置触发；其余配置不触发（符合预期）

## D. Network Skeleton

- [x] `models/network_swinir.py` 的 `SwinIR` 存在
- [x] `CloudCrossAttention` 存在
- [x] `TEMPORAL_FUSION_REGISTRY` + `temporal_fusion_mean` 存在并被 forward 调用
- [x] `mask_prob` 通路可达（cross-attn 启用时）

## E. Trainer and Indicator

- [x] `trainer.py` 的 `TrainerAlphaSR` 存在，`training_step` 和 `validation` 可达
- [x] `indicating_mask` 在 trainer 主线中为可选加权开关
  - 开关: `train.use_indicating_mask_in_training`
- [x] `trainer_mask_ablation.py` 的 `TrainerAlphaSRMaskAblation` 存在并支持
  - `indicating_mask_reduce`: `mean|max|prob_or`

## F. Predictor Compatibility

- [x] `inference.py` 的 `Predictor` 为主实现
- [x] `predictor.py` 保留兼容类（老接口不丢）
  - 构造: `Predictor(config_path, ckpt_path)`
  - 方法: `run_on_batch`, `save_prediction`, `run_inference`

## G. Config-Driven Activation Matrix (Must Match)

- [x] 2b/3x/1x：`use_advanced_processor=false`（cloud processor 不启用）
- [x] 4a/4b/4c：`use_advanced_processor=true`（cloud processor 启用）
- [x] 4c：`trainer.target=trainer_mask_ablation.TrainerAlphaSRMaskAblation`

## G.1 Validation Evidence (2026-03-16)

- static: `tmp/recovery_validation_logs/20260316_095545/static_report.json` (20/20 pass)
- smoke full: `tmp/recovery_validation_logs/20260316_095545/smoke.log` (17/18 pass) + 4c patch后单独复跑通过
- e2e recheck: `tmp/recovery_validation_logs/recheck/e2e/e2e_report.json` (`train_rc=0`, `test_rc=0`, `passed=true`)

## H. Validation Execution Order (Recommended)

1. 先做静态核对（符号存在、配置路径一致）
2. 再做轻量 smoke（单配置、单样本）
3. 最后做端到端（train/test 各一次）

## I. Freeze Rule (Current Stage)

- 本阶段只允许“恢复/兼容/可运行”类修改。
- 不新增算法模块、不改实验结论口径。
- 所有新增改动必须能映射到上述 A-F 某一条验收项。

## J. Linux 执行命令（你直接运行）

### 0) 准备

```bash
cd /mnt/lm_data_afs/wangzining/charles/AEF_swinir
chmod +x scripts/run_recovery_validation.sh
```

### 1) 一键跑完整验证（静态 + smoke + 端到端）

```bash
bash scripts/run_recovery_validation.sh cloud-ai-lab
```

### 2) 若只想分步跑

```bash
# 静态检查
conda run -n cloud-ai-lab python scripts/recovery_static_check.py \
  --out tmp/recovery_validation_logs/static_report.json

# smoke（全量 18 个配置）
conda run -n cloud-ai-lab python smoke_test_ablations.py

# 端到端最小验证（1 iter train + test）
conda run -n cloud-ai-lab python scripts/recovery_e2e_check.py \
  --cfg configs/ablation/config_true_baseline.yaml \
  --input-dir data/Cloud_test/processed_data_SR_10m/test \
  --out-dir tmp/recovery_validation_logs/manual
```

### 3) 跑完后把这些结果路径给我

- `tmp/recovery_validation_logs/<timestamp>/manifest.txt`
- `tmp/recovery_validation_logs/<timestamp>/static_report.json`
- `tmp/recovery_validation_logs/<timestamp>/smoke.log`
- `tmp/recovery_validation_logs/<timestamp>/e2e/e2e_report.json`
- `tmp/recovery_validation_logs/<timestamp>/e2e.log`

我会基于这些日志逐项判断并回填 A-G 的勾选状态。
