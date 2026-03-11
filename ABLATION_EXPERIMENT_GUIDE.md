# AEF_SwinIR 消融实验统一指南（2026-03-04 更新）

> 本文档为唯一推荐实验说明。原 `ABLATION_*.md` 细分文档内容已合并到本文件。

## 1. 实验目标与设计原则

### 1.1 实验目标
- 建立可复现实验链路，分离以下模块的独立贡献：
  - `time_band`
  - `mask_band`（hard/soft）
  - `cross_attention`
  - `position encoding`（sincos/learnable/concat）

### 1.2 单变量对照规则
- 每组只改一个变量，其余保持一致。
- 统一训练策略：`batch=[1,1]`，`lr=1e-4`，`seed=42`。
- 同组比较时，必须固定数据集划分和日志解析口径。

## 2. 数据与维度说明（必须统一）

### 2.1 输入/输出张量维度
- 单时相原始 LR：`(C_lr, H, W)`
- 训练输入序列：`lr_sequence: (T, C_in, H, W)`
- batch 后：`(B, T, C_in, H, W)`
- 输出：`prediction: (B, C_out, 3H, 3W)`

### 2.2 `C_in` 计算逻辑
- 基础反射率通道：`9`
- 若 `time_band.enabled=true`：`+1`
- 若 `mask_band.enabled=true`：`+1`
- 因此常见输入通道：
  - baseline：`9`
  - +time：`10`
  - +mask：`10`
  - +time+mask：`11`
- 你当前日志里的 `detected input channels: 12` 是合理的（取决于具体数据/拼接策略，代码已动态检测）。

### 2.3 掩膜相关维度
- `mask_band`（输入特征）：`(1, H, W)`，值域最终映射到 `[-1,1]`
- `pixel_mask`（内部计算）：`(H, W)`，值域 `[0,1]`
- 软掩膜（`mask_type=soft`）通过距离变换得到连续值，不是二值硬切分。

## 3. 最新消融矩阵（含 soft mask）

### 3.1 主实验链（推荐）

| Group | 配置 | 关键变量 | 对照基准 |
|---|---|---|---|
| G1 | `config_true_baseline` | 真基线 | - |
| G1 | `ablation_1b_baseline_plus_timeband` | `+time_band` | vs baseline |
| G1 | `ablation_1c_baseline_plus_maskband` | `+mask_band(hard)` | vs baseline |
| G1 | `ablation_1d_baseline_plus_timeband_maskband` | `+time+mask(hard)` | vs baseline |
| G2 | `ablation_2b_with_cross_attention_no_posenc` | `+cross_attention` | vs 1d |
| G3 | `ablation_3a_with_cross_attention_posenc_sincos` | `+pos(sincos)` | vs 2b |
| G3 | `ablation_3b_with_cross_attention_posenc_learnable` | `+pos(learnable)` | vs 2b |
| G3 | `ablation_3c_with_cross_attention_posenc_concat` | `+pos(concat)` | vs 2b |

### 3.2 扩展实验链（本周新增）

| Group | 配置 | 关键变量 | 对照基准 |
|---|---|---|---|
| G3-Ext | `ablation_3d_posenc_without_cross_attention` | 无cross时加pos | vs 1d |
| G4 | `ablation_4a_soft_mask_test` | `hard -> soft`（不启用高级处理器） | vs 2b |
| G4 | `ablation_4a_with_cross_attention_softmask` | 软掩膜版本（历史命名） | vs 2b |
| G4 | `ablation_4b_advanced_processor_soft_simplified` | `use_advanced_processor` 影响 | vs 4a |
| G4 | `ablation_4c_mask_reduce_prob_or` | `indicating_mask` 聚合策略优化 | vs 4a |

## 4. 软掩膜口径（最新）

### 4.1 当前生效条件
- `features.mask_band.enabled=true`
- `features.mask_band.processor.mask_type=soft`
- `soft_mask_sigma` 控制平滑强度（默认 `2.0`）

### 4.3 indicating_mask 聚合策略（ablation_4c 新增）

**背景**：当前训练器在损失计算时，会对 `indicating_mask (B,T,H,W)` 沿时间维做 `mean(dim=1)` 聚合为 `(B,1,H,W)`，这会抹平各时相的质量差异。

**改进方案**（新增 `trainer_mask_ablation.py`）：
- **prob_or（默认）**：`1 - Π(1-m_t)` - 任一时相可见即提高监督权重
- **max**：取各时相最大值 - 保留最佳观测
- **mean**：简单平均（原方案）

**对应文件**：
- Trainer：`trainer_mask_ablation.py`
- 配置：`configs/ablation/ablation_4c_mask_reduce_prob_or.yaml`
- 运行脚本：`scripts/run_ablation_4c_mask_reduce_prob_or.sh`

**配置开关**：
```yaml
train:
  indicating_mask_reduce: "prob_or"  # 可选: mean / max / prob_or
```

**适用场景**：
- 单帧输出模型（当前 SwinIR）
- 云质量时相差异大
- 期望更精细的损失权重

**运行命令**：
```bash
bash scripts/run_ablation_4c_mask_reduce_prob_or.sh 0
```

**推荐对比链**：`2b → 4a → 4c`
- `2b → 4a`：验证软掩膜本身收益
- `4a → 4c`：验证聚合策略本身收益

**预期收益（经验口径）**：`+0.1 ~ +0.3 dB`（以同训练预算比较为准）

**回滚方式**（若效果不达预期）：
```bash
rm trainer_mask_ablation.py
rm configs/ablation/ablation_4c_mask_reduce_prob_or.yaml
rm scripts/run_ablation_4c_mask_reduce_prob_or.sh
```

### 4.2 日志判断标准
训练启动时应看到：
- `Mask Type: soft`
- `Soft Mask Sigma: 2.0`

若只看到 `Cloud Mask: Simple inference ...` 不代表软掩膜失效，它表示 `use_advanced_processor=false`（即不读取外部QA文件，仍可做 soft）。

## 5. 训练执行规范

### 5.1 单实验运行
```bash
python main.py --cfg_path configs/ablation/ablation_4a_soft_mask_test.yaml --mode train
```

### 5.2 对比运行
```bash
bash run_soft_mask_comparison.sh
```

### 5.3 统一约束建议
- 同批次对比保持 `iterations` 一致。
- 若历史实验是 `1500`，新实验是 `5000`，结论要注明“预算不同”。

## 6. 指标口径与结论口径

### 6.1 指标来源
- 从 `training.log` 中解析：
  - `Validation Metrics | PSNR: ... | SSIM: ...`
- 最佳定义：`PSNR` 最大值对应的迭代。

### 6.2 报告推荐结论模板
- 绝对指标：`best_psnr / best_ssim / best_iter`
- 对照增益：
  - `Δtime = PSNR(1b)-PSNR(1a)`
  - `Δmask_hard = PSNR(1c)-PSNR(1a)`
  - `Δcross = PSNR(2b)-PSNR(1d)`
  - `Δsoft = PSNR(4a)-PSNR(2b)`

## 7. 快速排障

### Q1: 为什么 dataset 日志显示 simple inference？
表示未启用高级处理器，不等于软掩膜无效。看 `Mask Type` 才是关键。

### Q2: 如何确认 soft 真在跑？
检查配置与日志：
- 配置里必须是 `mask_type: soft`
- 启动日志必须打印 `Mask Type: soft`

### Q3: 结果分析脚本如何跑？
```bash
python scripts/analyze_ablation_results.py
python scripts/analyze_ablation_final.py
```

### Q4: ablation_4c 和其他 4 系的区别？
- **4a**: 使用软掩膜（`mask_type=soft`），但 indicating_mask 仍用简单 mean
- **4b**: 启用高级处理器（`use_advanced_processor=true`）
- **4c**: 改进 indicating_mask 的时间聚合策略（`prob_or` 而非 `mean`），不改其他部分

**推荐对比链**：`2b → 4a → 4c`，验证软掩膜 + 聚合策略的叠加收益。

输出：
- `ablation_summary_latest.csv`
- `ablation_best_results/ablation_analysis_report.md`
- `ablation_best_results/best_results.json`

## 8. 最小执行清单

- [ ] 完成 baseline / 2b / 4a_soft 三组训练
- [ ] 运行分析脚本导出最新报告
- [ ] 在周报中记录：配置、训练预算、最佳指标、相对增益
- [ ] 结论必须附“是否同预算比较”标记

## 9. 常用命令（从旧文档合并）

### 9.1 运行单个实验
```bash
python main.py --cfg_path configs/ablation/config_true_baseline.yaml --mode train
python main.py --cfg_path configs/ablation/ablation_2b_with_cross_attention_no_posenc.yaml --mode train
python main.py --cfg_path configs/ablation/ablation_4a_soft_mask_test.yaml --mode train

# 新增：indicating_mask 聚合策略优化试验
bash scripts/run_ablation_4c_mask_reduce_prob_or.sh 0
```

### 9.2 查看训练日志
```bash
tail -f training_logs/experiments/ablation_4a_soft_mask_test/*/training.log
```

### 9.3 运行结果分析
```bash
python scripts/analyze_ablation_results.py
python scripts/analyze_ablation_final.py
```

## 10. 运行前检查（从 checklist 合并）

- [ ] 数据目录存在：`data/Cloud_test/processed_data_SR_10m/{train,val,test}`
- [ ] 配置目录存在：`configs/ablation/*.yaml`
- [ ] 训练预算在同组比较中一致（`iterations`）
- [ ] soft 对照实验配置包含：`mask_type: soft`
- [ ] 启动日志能看到：`Mask Type` 与 `Soft Mask Sigma`

## 11. 增益显著性口径（从历史说明合并）

- `< 0.3 dB`：可忽略或不稳定
- `0.3 ~ 0.5 dB`：轻微改进
- `0.5 ~ 1.0 dB`：中等改进
- `> 1.0 dB`：显著改进

> 以上阈值需结合同预算比较前提；跨预算（例如 1500 vs 5000 iterations）只可作为趋势参考。

## 12. 参考文献

- SwinIR: Image Restoration via Swin Transformer (2021)
- Rethinking and Improving Relative Position Encoding (2022)
- An Empirical Study of Training Data for Attention-Based Models (2021)

