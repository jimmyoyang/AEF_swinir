# 2026-03-04 更新日志：indicating_mask 聚合策略优化试验

## 更新概览

新增独立试验分支 `ablation_4c_mask_reduce_prob_or`，用于验证在单帧输出模型下，优化 `indicating_mask` 时间维聚合策略的效果。

## 新增文件

### 1. 独立试验 Trainer
- **文件**：`trainer_mask_ablation.py`
- **类名**：`TrainerAlphaSRMaskAblation`
- **改动**：只修改 `indicating_mask` 的时间聚合方式，不改其他训练逻辑
- **支持策略**：
  - `prob_or`（默认）：`1 - Π(1-m_t)` - 任一时相可见即提高监督权重
  - `max`：取各时相最大值 - 保留最佳观测
  - `mean`：简单平均（原方案，作为对照）

### 2. 独立试验配置
- **文件**：`configs/ablation/ablation_4c_mask_reduce_prob_or.yaml`
- **关键配置**：
  ```yaml
  trainer:
    target: trainer_mask_ablation.TrainerAlphaSRMaskAblation
  
  train:
    indicating_mask_reduce: "prob_or"  # 可切换策略
    save_dir: "./training_logs/experiments/ablation_4c_mask_reduce_prob_or"
  ```

### 3. 运行脚本
- **文件**：`scripts/run_ablation_4c_mask_reduce_prob_or.sh`
- **用法**：`bash scripts/run_ablation_4c_mask_reduce_prob_or.sh [GPU_ID]`

## 更新的文档

### 1. ABLATION_EXPERIMENT_GUIDE.md
- 在 `3.2 扩展实验链` 中添加 `ablation_4c` 条目
- 新增 `4.3 indicating_mask 聚合策略` 章节
- 更新 `9.1 运行单个实验` 添加新运行命令
- 新增 `Q4` 常见问题：解释 4a/4b/4c 的区别

### 2. WEEKLY_REPORT_2026-03-03.md
- 在 `2026-03-04 更新` 部分新增 `5) indicating_mask 聚合策略优化试验`
- 说明问题背景、改进方案、预期收益

### 3. QUICKSTART.md
- 在 `单个配置训练` 章节添加新试验运行方式
- 新增 `ablation_4c` 说明和对比链路建议

## 技术背景

### 问题识别
当前 `trainer.py` 在损失计算时：
```python
if indicating_mask.ndim == 4:  # (B, T, H, W)
    indicating_mask = indicating_mask.mean(dim=1, keepdim=True)  # → (B, 1, H, W)
```

这会**抹平各时相的质量差异**：
- 时相1：云覆盖 25%，mask=0.75
- 时相2：云覆盖  5%，mask=0.95
- 时相3：云覆盖 40%，mask=0.60

经过 mean 后，三个时相都用同样的 mask=0.767，导致：
- 时相2（最清晰）被向下加权
- 时相3（最模糊）被相对强化
- 梯度方向偏差

### 改进原理
使用 `prob_or = 1 - Π(1-m_t)` 策略：
- 物理意义：任一时相可见，该像素就应该有较高监督权重
- 数学特性：对于 N 个独立观测，至少一个有效的概率
- 适用场景：单帧输出 + 多时相输入的云掩膜场景

### 预期收益
- 在单帧输出模型下：+0.1~0.3 dB（小幅稳定性提升）
- 梯度更精准：每个像素用正确的时相权重计算损失
- 不改变模型结构：纯训练策略优化，易于回滚

## 对比链路
建议实验对比顺序：
1. `ablation_2b` - cross-attention baseline（硬掩膜 + 简单 mean）
2. `ablation_4a` - 软掩膜改进（soft mask + 简单 mean）
3. `ablation_4c` - 聚合策略改进（soft mask + prob_or）

这样可以分离验证：
- 2b → 4a：软掩膜的独立贡献
- 4a → 4c：聚合策略的独立贡献

## 使用示例

```bash
# 1. 运行试验（GPU 0）
bash scripts/run_ablation_4c_mask_reduce_prob_or.sh 0

# 2. 查看日志
tail -f training_logs/experiments/ablation_4c_mask_reduce_prob_or/*/training.log

# 3. 对比结果
python scripts/analyze_ablation_results.py
```

## 注意事项

1. **完全隔离**：不修改原有 `trainer.py`，所有改动在新文件中
2. **可独立验证**：试验效果不满意可直接删除新文件
3. **配置灵活**：可在配置中切换 `indicating_mask_reduce` 策略
4. **语法验证**：已通过 Pylance 语法检查，无错误

## 验证状态

- ✅ 代码创建完成
- ✅ 配置文件创建完成
- ✅ 运行脚本创建完成
- ✅ 文档更新完成
- ✅ 语法检查通过
- ⏳ 训练中（用户当前正在运行）

## 回滚方案

如果试验效果不理想，直接删除以下文件即可：
```bash
rm trainer_mask_ablation.py
rm configs/ablation/ablation_4c_mask_reduce_prob_or.yaml
rm scripts/run_ablation_4c_mask_reduce_prob_or.sh
```

原有训练流程完全不受影响。
