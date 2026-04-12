# 快速开始指南

## 文件结构说明

```
AEF_swinir/
├── main.py                           # 主训练入口
├── trainer.py                        # Trainer类（包含best_ckpt_manager集成）
├── run_ablation_experiments_extended.sh  # ⭐ 主要脚本：批量运行消融配置
├── train.sh                          # 快速脚本：单配置训练
├── scripts/
│   ├── best_ckpt_manager.py          # 最佳权重管理器
│   ├── validate_ablation_configs.py  # 配置一致性检查器
│   ├── run_ablation_4c_mask_reduce_prob_or.sh
│   ├── run_ablation_4d_mask_temporal_loss.sh
│   ├── run_ablation_4c_4d_and_analyze.sh
│   └── update_iterations.py          # 更新 iterations 的工具
├── configs/
│   ├── ablation/config_true_baseline.yaml
│   ├── ablation/ablation_2b_with_cross_attention_no_posenc.yaml
│   ├── ablation/ablation_3a_with_cross_attention_posenc_sincos.yaml
│   ├── ablation/ablation_3b_with_cross_attention_posenc_learnable.yaml
│   └── ablation/ablation_3c_with_cross_attention_posenc_concat.yaml
└── best_ckpts/                       # 最佳权重自动保存位置
    ├── exp_baseline/model.pth
    ├── exp_cross/model.pth
    └── ...
```

## 使用方式（简化版）

脚本支持参数：`GPU_ID` 和可选 `ITER_OVERRIDE`。

### 快速测试（100迭代）
```bash
bash run_ablation_experiments_extended.sh 0 100
```

### 正式训练（5000迭代）
```bash
bash run_ablation_experiments_extended.sh 0 5000
```

### 大规模训练（10000迭代）
```bash
bash run_ablation_experiments_extended.sh 0 10000
```

**参数说明：**
- 第一个参数：GPU ID（默认0）
- 第二个参数：要设置的 iterations（默认100）

脚本会按预设顺序运行 `configs/ablation/*.yaml` 中的消融配置。

### 4c/4d 接力 + 自动分析
```bash
bash scripts/run_ablation_4c_4d_and_analyze.sh 0
```

## 工作流示例

### 快速验证阶段
```bash
# 设置 100 迭代，快速验证系统是否正常
bash run_ablation_experiments_extended.sh 0 100
```

### 正式训练阶段
```bash
# 设置 5000 迭代，进行正式训练
bash run_ablation_experiments_extended.sh 0 5000

# 训练完成后查看最佳权重
python scripts/best_ckpt_manager.py list
```

## 主要特性

- **集中管理**：脚本统一控制多个消融配置和 iterations
- **简单参数**：只需指定 GPU 和 iterations 两个参数
- **批量执行**：脚本按预设顺序执行 `configs/ablation/*.yaml`
- **最佳权重管理**：
  - 自动保存每个实验的最佳权重到 `./best_ckpts/{exp_name}/model.pth`
  - 自动记录指标（PSNR、SSIM、迭代数）
- **支持Resume**：后续可基于最佳权重继续训练
  ```bash
  # 在配置文件中添加
  resume: ./best_ckpts/exp_baseline/model.pth
  ```

## 查看和修改 iterations

### 批量查看当前 iterations
```bash
grep -R "^[[:space:]]*iterations:" configs/ablation/*.yaml
```

### 3种方式修改 iterations

**方式1：通过脚本（推荐）**
```bash
bash run_ablation_experiments_extended.sh 0 100
```

**方式2：仅修改不训练（批量替换 YAML）**
```bash
find configs/ablation -name '*.yaml' -type f -exec sed -i 's/^\([[:space:]]*\)iterations: .*/\1iterations: 100/' {} +
```

**方式3：手动编辑配置文件**
```bash
vim configs/ablation/config_true_baseline.yaml
# 找到 train: 下的 iterations 行，改为需要的值
```

## 单个配置训练

如果只想训练单个模型：

```bash
# 先修改 iterations（例如改为 100）
find configs/ablation -name '*.yaml' -type f -exec sed -i 's/^\([[:space:]]*\)iterations: .*/\1iterations: 100/' {} +

# 然后训练单个配置
python main.py --cfg_path configs/ablation/config_true_baseline.yaml

# 新增：indicating_mask 聚合策略优化试验（独立分支）
bash scripts/run_ablation_4c_mask_reduce_prob_or.sh 0

# 新增：时相级 loss 试验（独立分支）
bash scripts/run_ablation_4d_mask_temporal_loss.sh 0
```

### 新增试验说明：ablation_4c / ablation_4d

**目的**：在单帧输出模型下，优化 `indicating_mask` 的时间维聚合策略。

**特点**：
- 完全独立：使用新 trainer（`trainer_mask_ablation.py`），不修改原有代码
- 4c 可配置：支持 `prob_or` / `max` / `mean` 三种策略
- 4d 更细粒度：每时相单独计算 masked loss，再做时序聚合
- 可回滚：试验结果不满意可直接删除，不影响其他实验

**运行命令**：
```bash
# GPU 0 运行（默认 5000 iterations）
bash scripts/run_ablation_4c_mask_reduce_prob_or.sh 0
bash scripts/run_ablation_4d_mask_temporal_loss.sh 0

# 查看日志
tail -f training_logs/experiments/ablation_4d_mask_temporal_loss/*/training.log
```

**对比链路**：建议 `2b → 4a → 4c → 4d`，验证软掩膜 + 聚合策略 + 时相级 loss 的叠加收益。

### 一键对比建议（4c/4d 相关）

```bash
# 1) 接力运行 4c + 4d + 分析
bash scripts/run_ablation_4c_4d_and_analyze.sh 0

# 2) 分析全量结果
python scripts/analyze_ablation_results.py
python scripts/analyze_ablation_final.py

# 3) 重点查看 2b / 4a / 4c / 4d 四组对照
grep -E "ablation_2b|ablation_4a|ablation_4c|ablation_4d" ablation_summary_latest.csv
```

> 建议在同训练预算（同 iterations）下对比，4c 预期为稳定性小幅提升（约 +0.1~+0.3 dB）。

## 快速清理

```bash
# 清理训练日志和最佳权重（保留源代码）
rm -rf training_logs best_ckpts

# 恢复配置文件到原始 iterations（以下是手动方式）
# 用 git 恢复
git checkout configs/ablation/*.yaml
```

## 故障排查

### 问题：iterations 没有正确更新
```bash
# 强制更新（批量替换）
find configs/ablation -name '*.yaml' -type f -exec sed -i 's/^\([[:space:]]*\)iterations: .*/\1iterations: 100/' {} +

# 复查
grep -R "^[[:space:]]*iterations:" configs/ablation/*.yaml
```

### 问题：想改回原始 iterations（5000）
```bash
bash run_ablation_experiments_extended.sh 0 5000
# 或批量恢复为 5000
find configs/ablation -name '*.yaml' -type f -exec sed -i 's/^\([[:space:]]*\)iterations: .*/\1iterations: 5000/' {} +
```

### 问题：某个模型训练失败
脚本会继续训练其他模型，最后显示哪些失败。检查对应的日志文件。

