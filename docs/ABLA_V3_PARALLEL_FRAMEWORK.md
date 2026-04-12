# Ablation V3 并行框架说明（不改现有 v2）

本文档对应 v3 组合后处理实验：
- 方案 1 + 方案 6（先 1 后 6）
- 方案 3 + 方案 6（先 3 后 6）

目标：
- 不修改现有 v2 文件。
- 维持与 v2 相同训练逻辑和验证流程。
- 使用与 3b/4c 一致的最佳预处理设置（learnable pos + advanced mask + prob_or）。
- 结果可与 v2 的单独方案 1/3/6 直接同表对比。

## 1. 新增文件

### 1.1 模型与后处理
- models/post_v3/composed.py
- models/post_v3/registry.py
- models/post_v3/__init__.py
- models/network_swinir_post_v3.py

### 1.2 训练器
- trainer_post_v3.py
- trainer_post_v3_mask_ablation.py

### 1.3 配置
- configs/ablation_v3/ablation_v3_5a_nopost_bestpre_quick.yaml
- configs/ablation_v3/ablation_v3_5a_nopost_bestpre_full.yaml
- configs/ablation_v3/ablation_v3_5e_scheme1_plus6_quick.yaml
- configs/ablation_v3/ablation_v3_5e_scheme1_plus6_full.yaml
- configs/ablation_v3/ablation_v3_5f_scheme3_plus6_quick.yaml
- configs/ablation_v3/ablation_v3_5f_scheme3_plus6_full.yaml
- configs/ablation_v3/ablation_v3_5e_scheme1_plus6_stage_quick.yaml
- configs/ablation_v3/ablation_v3_5e_scheme1_plus6_stage_full.yaml
- configs/ablation_v3/ablation_v3_5f_scheme3_plus6_stage_quick.yaml
- configs/ablation_v3/ablation_v3_5f_scheme3_plus6_stage_full.yaml
- configs/ablation_v3/ablation_v3_5e_scheme1_plus6_stage_gated_quick.yaml
- configs/ablation_v3/ablation_v3_5f_scheme3_plus6_stage_gated_quick.yaml

### 1.4 脚本
- scripts/run_ablation_v3_quick.sh
- scripts/run_ablation_v3_full.sh
- scripts/run_ablation_v3_stage_quick.sh
- scripts/run_ablation_v3_stage_full.sh
- scripts/run_ablation_v3_stage_gated_quick.sh
- scripts/compare_ablation_v3_vs_v2.py

## 2. 数据流与维度

与 v2 保持一致：
1) 输入字典（lr_sequence/timestamps/gt/mask/mask_prob/indicating_mask）。
2) SwinIR 主干输出 `(B,64,H*scale,W*scale)`。
3) v3 组合后处理采用并行残差汇合：
   - `r1 = first(x) - x`
   - `r2 = second(x) - x`
   - 默认输出：`x + r1 + r2`
4) 若启用可学习门控：
   - `output = x + alpha * r1 + beta * r2`
   - `alpha,beta` 由 sigmoid(logit) 得到，初始 0.5/0.5。
5) 若启用 staged 训练：
   - 前期：`second` 分支缩放为 0（只训练 first 主干分支）
   - 中期：second 缩放线性ramp
   - 后期：双分支共同训练
6) 输出维度保持 `(B,64,H*scale,W*scale)`，再进入 loss。

## 3. 控制变量与科学性

### 3.1 关键原则

- 仅监控 `alpha/beta` 和分支残差范数不会影响 loss，也不会破坏公平对比。
- 只要不把这些监控项加入 loss，就仍是同一优化目标。
- `staged training` 属于新训练策略，必须单独命名、单独配置、单独对比。

### 3.2 从保守到激进的实验阶梯

1) **保守**：原 v3（无staged、无learnable gates）
2) **中等**：v3-stage（有staged、固定 `alpha=beta=1`）
3) **激进**：v3-stage-gated（有staged、learnable `alpha/beta`）

这三步每次只改变一个主要变量，避免“不充分对比”。

## 4. 运行步骤（和 v2 同逻辑，新增stage分支）

### 4.1 先做语法检查

```bash
bash -n scripts/run_ablation_v3_quick.sh \
&& bash -n scripts/run_ablation_v3_full.sh \
&& bash -n scripts/run_ablation_v3_stage_quick.sh \
&& bash -n scripts/run_ablation_v3_stage_full.sh \
&& bash -n scripts/run_ablation_v3_stage_gated_quick.sh \
&& python -m py_compile scripts/compare_ablation_v3_vs_v2.py \
&& echo "[OK] v3 syntax check passed"
```

### 4.2 跑 baseline v3 quick

```bash
bash scripts/run_ablation_v3_quick.sh 0
```

### 4.3 跑 stage quick（新训练策略）

```bash
bash scripts/run_ablation_v3_stage_quick.sh 0
```

### 4.4 跑 stage+gated quick（可学习门控）

```bash
bash scripts/run_ablation_v3_stage_gated_quick.sh 0
```

### 4.5 汇总 quick（含v2、v3、v3-stage、v3-stage-gated）

```bash
python scripts/compare_ablation_v3_vs_v2.py --v2-root training_logs/experiments_v2 --v3-root training_logs/experiments_v3 --output ablation_v3_quick_vs_v2.csv
cat ablation_v3_quick_vs_v2.csv
```

### 4.6 跑 baseline/stage full

```bash
bash scripts/run_ablation_v3_full.sh 0
bash scripts/run_ablation_v3_stage_full.sh 0
```

### 4.7 汇总 full 并与 v2 对比

```bash
python scripts/compare_ablation_v3_vs_v2.py --v2-root training_logs/experiments_v2 --v3-root training_logs/experiments_v3 --output ablation_v3_full_vs_v2.csv
cat ablation_v3_full_vs_v2.csv
```

## 5. 通过性检查建议

quick 阶段建议满足：
- v3 三组都出现在结果里：v3_5a、v3_5e、v3_5f。
- 每组有 best_psnr/psnr/ssim/sam。
- 1+6 方案 `active_groups` 不长期接近 1（避免路由塌陷）。
- 若跑了 staged/gated，还应检查：
  - `gate_alpha/gate_beta` 不出现长期极端值（长期接近 0 代表分支失活）
  - `residual_first_l2/residual_second_l2` 量级不出现长期单分支独占
  - `stage_phase` 随迭代从 0(warmup) -> 1(ramp) -> 2(joint)

full 阶段建议：
- 以 full 结果作为最终结论。
- 对比列中同时观察：
  - v2 单独 1/3/6
  - v3 baseline 组合 1+6 / 3+6
  - v3-stage 组合 1+6 / 3+6
  - v3-stage-gated quick（探索性）
  - no-post 基线

## 6. 代码与数据接口变化

### 6.1 关键代码改动

- `models/post_v3/composed.py`
  - 新增可选 learnable gates（`use_learnable_gates`）
  - 新增 staged 调度（`staged_training`）
  - 新增统计字段：`gate_alpha/gate_beta/stage_* / residual_*`

- `models/post_v3/registry.py`
  - 新增 `compose` 参数透传，保证模块可开关

- `trainer_post_v3_mask_ablation.py`
  - 保持训练逻辑不变
  - 扩展 `post_v2_stats.csv` 动态字段写入，支持新统计字段自动落盘

- `scripts/compare_ablation_v3_vs_v2.py`
  - 新增 stage 与 stage_gated 目录抓取
  - 动态解析 `post_v2_stats.csv` 尾行的所有数值字段

### 6.2 数据输入输出变化

- 数据输入接口不变：仍为原训练管线字典输入。
- 模型输出张量形状不变：`(B,64,H*scale,W*scale)`。
- 训练 loss 输入不变：仍是 prediction 与 gt。
- 新增的仅是监控指标输出，不改变 loss 计算路径。

## 7. 风险与方法

### 7.1 风险

- `alpha/beta` 可能塌陷到单分支主导。
- staged warmup 过长可能造成 second 分支学习不足。
- quick 迭代数过少时，组合结构可能表现出“未收敛假退化”。

### 7.2 缓解方案

- 先跑 stage（固定门控）再跑 stage+gated（学习门控），分步验证。
- quick 建议提高到 300-500 再判断趋势。
- full 前检查 `gate_alpha/gate_beta` 和两分支残差范数轨迹。

### 7.3 相关思路（无需额外代码库）

- Residual scaling（超分辨率模型常见）
- Learnable weighted feature fusion（检测/分割多分支融合常见）
- Curriculum / warmup 式 staged training

---

## 附录：v3 Quick 结果诊断（历史与当前）

### A.1 初始不理想结果（历史）

```
- v3_5e_scheme1_plus6: best_psnr=14.0833（偏低）
- v3_5f_scheme3_plus6: best_psnr=14.5912（略低）
```

### A.2 修复后最近一次 quick（你当前截图）

```
- v2_5b_scheme1_gumbel: best_psnr=14.4919
- v2_5c_scheme3_dual_branch: best_psnr=14.7254
- v2_5d_scheme6_matrix: best_psnr=14.4848
- v3_5e_scheme1_plus6: best_psnr=14.1054（仍偏低）
- v3_5f_scheme3_plus6: best_psnr=14.6105（可继续优化）
```

结论：
- 1+6 仍需策略层改进（staged 或 staged+gated）。
- 3+6 可继续推进到 stage / full 验证。

### A.3 staged 与 gated 验证命令

```bash
# 新训练策略：stage
bash scripts/run_ablation_v3_stage_quick.sh 0

# 进一步：stage + learnable gates(alpha,beta)
bash scripts/run_ablation_v3_stage_gated_quick.sh 0

# 对比抓取（统一）
python scripts/compare_ablation_v3_vs_v2.py --v2-root training_logs/experiments_v2 --v3-root training_logs/experiments_v3 --output ablation_v3_progressive_quick_vs_v2.csv
cat ablation_v3_progressive_quick_vs_v2.csv
```

---

## 附录：v3 Quick (100 iter) 结果诊断与梯度流修复

### A.1 Quick 测试结果（初始版本，存在问题）

```
Summary:
- v2_5a_nopost_bestpre: best_psnr=14.5568, last_psnr=14.4730, last_ssim=0.3462
- v2_5b_scheme1_gumbel: best_psnr=13.7330, last_psnr=13.7197, last_ssim=0.2711
- v2_5c_scheme3_dual_branch: best_psnr=14.7254, last_psnr=14.7254, last_ssim=0.3414
- v2_5d_scheme6_matrix: best_psnr=14.4848, last_psnr=14.4848, last_ssim=0.3331
- v3_5a_nopost_bestpre: best_psnr=14.6921, last_psnr=14.6921, last_ssim=0.3262 ✓ (合理)
- v3_5e_scheme1_plus6: best_psnr=14.0833, last_psnr=14.0833, last_ssim=0.2723 ⚠️ (偏低) 
- v3_5f_scheme3_plus6: best_psnr=14.5912, last_psnr=14.4474, last_ssim=0.3316 ◐ (略低)
```

**问题分析**：
- **v3_5e (1+6) PSNR=14.0833 偏低**：
  - v2_5b (scheme1单独) = 13.7330
  - v2_5d (scheme6单独) = 14.4848
  - 预期组合应接近 min(14.48) 或略高，但实际是 14.0833（在两者之间或略低）
  - **根本原因**：梯度流设计错误导致学习效率下降

- **v3_5f (3+6) PSNR=14.5912 相对合理**：
  - v2_5c (scheme3单独) = 14.7254
  - 组合后 -0.134 在可接受范围内

### A.2 梯度流问题诊断

#### A.2.1 错误的 Sequential 组合方式

**问题代码**（[models/post_v3/composed.py](models/post_v3/composed.py) 修改前）：

```python
def _forward_impl(self, x, context):
    y = self.first(x)      # y = x + first_residual (e.g., Gumbel output)
    z = self.second(y)     # z = y + second_residual = x + first_res + second_res
    
    # 问题：second 接收到的是 first 的输出，而不是原始输入 x
    # Gumbel 后处理已经改造了特征，导致 Matrix 无法有效地进行二次处理
    
    return z
```

**问题根源**：

每个后处理器都设计为残差连接：
```python
# Gumbel (scheme1): return x + out
# Matrix (scheme6): return x + x_out
```

当 Gumbel 输出作为 Matrix 的输入时：
```
x_original --[Gumbel: x + g(x)]→ y
y (已改造) --[Matrix: y + m_op(y)]→ z = (x + g(x)) + m_op(x + g(x))
```

这导致：
1. **梯度路径复杂化**：∂z/∂x 包含多层嵌套的非线性，early layers 梯度衰减
2. **特征污染**：Matrix 在接收 Gumbel 改造后的特征时，无法有效进行独立路由
3. **学习目标冲突**：Gumbel 试图优化通道路由；Matrix 试图优化空间融合；两者在中间特征上相互干扰

#### A.2.2 预期的 Parallel 组合方式

**修改原因**：

两个后处理应该**独立并行地处理原始输入 x**，然后汇合其贡献：

```
                   ┌→ scheme1(x) = x + g(x) ┐
x_original ────→ ┤                             ┼→ 汇合 → output
                   └→ scheme6(x) = x + m(x) ┘

output = first(x) + second(x) - x
       = (x + g(x)) + (x + m(x)) - x
       = x + g(x) + m(x)
```

这样：
- ✓ 两个后处理器都在原始特征上工作
- ✓ 各自的特征学习不被另一方污染
- ✓ 梯度分支清晰：∂loss/∂x 分别来自两个后处理
- ✓ 残差相加：最终的改进 = g(x) + m(x)

#### A.2.3 修改后的代码

**新的实现**（[models/post_v3/composed.py](models/post_v3/composed.py) 修改后）：

```python
def _forward_impl(self, x, context):
    # 两个后处理器独立处理原始输入 x
    y = self.first(x)   # y = x + first_residual
    z = self.second(x)  # z = x + second_residual (注意：也是处理 x，不是处理 y)
    
    # 合并：y + z - x = (x + first_res) + (x + second_res) - x = x + first_res + second_res
    output = y + z - x
    
    # 统计信息继续来自 first 为主
    first_stats = self.first.get_last_stats() if hasattr(self.first, "get_last_stats") else {}
    second_stats = self.second.get_last_stats() if hasattr(self.second, "get_last_stats") else {}
    
    self._last_stats = {
        "post_tau": float(first_stats.get("post_tau", 0.0)),
        "routing_entropy": float(first_stats.get("routing_entropy", 0.0)),
        "active_groups": float(first_stats.get("active_groups", 0.0)),
        "second_active_groups": float(second_stats.get("active_groups", 0.0)),
    }
    
    extra = {}
    if hasattr(self.first, "get_last_extra_losses"):
        extra.update(self.first.get_last_extra_losses() or {})
    if hasattr(self.second, "get_last_extra_losses"):
        extra.update(self.second.get_last_extra_losses() or {})
    self._last_extra_losses = extra
    
    return output
```

**关键改动**：
- 行 6：`z = self.second(x)` ✓ 而不是 `self.second(y)`
- 行 9：`output = y + z - x` ✓ 并行汇合公式

### A.3 预期改进与下一步

**修改后的预期结果**（待验证）：

运行以下命令重新测试：
```bash
bash scripts/run_ablation_v3_quick.sh 0
python scripts/compare_ablation_v3_vs_v2.py --v2-root training_logs/experiments_v2 --v3-root training_logs/experiments_v3 --output ablation_v3_quick_fixed_vs_v2.csv
cat ablation_v3_quick_fixed_vs_v2.csv
```

预期改进：
- **v3_5e (1+6) PSNR** 应从 14.0833 → 接近或超过 14.48（至少不低于两个单独方案的均值）
- **v3_5f (3+6) PSNR** 应从 14.5912 → 更接近 14.7254

**验证清单**：
- [ ] v3 quick 完成，新结果表
- [ ] v3_5e (scheme1+6) 梯度流确认正常（routing_entropy 健康）
- [ ] v3_5f (scheme3+6) 与单独方案不再有长期下降趋势
- [ ] 通过检验后运行 v3 full

---

## 附录：V3 Full vs V2 对比结论（2026-03-20）

数据来源：
- `ablation_v3_full_vs_v2.csv`
- 汇总命令：`python scripts/compare_ablation_v3_vs_v2.py --v2-root training_logs/experiments_v2 --v3-root training_logs/experiments_v3 --output ablation_v3_full_vs_v2.csv`

说明：
- 本表中 `v3_5e_scheme1_plus6_stage_gated` 与 `v3_5f_scheme3_plus6_stage_gated` 当前来自 quick 运行目录（仅探索性参考），不作为 full 最终结论依据。

### B.1 关键指标（full 主结论使用集合）

| scheme | best_psnr | last_psnr | last_ssim | sam | ergas |
|---|---:|---:|---:|---:|---:|
| v2_5d_scheme6_matrix | 14.5960 | 14.4787 | 0.3492 | 0.3526 | 11.0604 |
| v2_5a_nopost_bestpre | 14.5568 | 14.4733 | 0.3461 | 0.3528 | 11.0840 |
| v3_5a_nopost_bestpre | 14.5621 | 14.4809 | 0.3479 | 0.3524 | 11.0721 |
| v3_5e_scheme1_plus6 | 14.3818 | 14.2464 | 0.3309 | 0.3608 | 11.3389 |
| v3_5e_scheme1_plus6_stage | 14.4062 | 14.2783 | 0.3225 | 0.3613 | 11.3157 |
| v3_5f_scheme3_plus6 | 14.3810 | 14.2933 | 0.3399 | 0.3599 | 11.3604 |
| v3_5f_scheme3_plus6_stage | 14.0616 | 13.7085 | 0.2889 | 0.3826 | 12.5540 |

### B.2 主要发现

1. `v3_5a_nopost_bestpre` 是 v3 full 下最稳的结果，且与 v2 最优基本同级。
  - 相对 `v2_5a_nopost_bestpre`：best_psnr +0.0053，last_psnr +0.0076，ssim +0.0018。
  - 说明“最佳预处理 + 无复杂后处理”仍是强基线。

2. 组合后处理（1+6 / 3+6）在 full 下没有超过 v2 的最优（5d matrix）。
  - 相对 `v2_5d_scheme6_matrix`，v3 组合路线 best_psnr 下降约 0.19 到 0.53。
  - `v3_5f_scheme3_plus6_stage` 退化最明显（PSNR/SSIM/SAM/ERGAS 全面变差）。

3. staged 对 1+6 有小幅改善，但不足以逆转结论。
  - `v3_5e_stage` 比 `v3_5e` 略好（best_psnr +0.0244），但仍低于 no-post 与 v2_5d。

4. 统计信号显示分支贡献不平衡风险仍在。
  - `v3_5e_stage` 的 `residual_second_l2` 仍显著小于 `residual_first_l2`，第二分支贡献偏弱。
  - `v3_5f_stage` 末期指标明显恶化，需优先排查 staged 调度与分支学习率耦合问题。

### B.3 结论（可直接用于报告）

- 当前 full 结果不支持“v3 组合后处理优于 v2 最优单模块”这一假设。
- 在同等对比下，推荐主线为：`v3_5a_nopost_bestpre`（稳定）或直接沿用 `v2_5d_scheme6_matrix`（当前全表 best_psnr 最优）。
- `v3_5e/5f` 及其 stage 版本暂不建议作为主推方案；仅保留为探索分支。

### B.4 下一步建议（最小增量）

1. 若继续验证 gated 路线，需补齐 full 配置与 full 运行后再与本节主结论并表。
2. 对 `v3_5f_stage` 优先做稳定性修复（缩短 warmup、提高 second 分支有效学习窗口）后再复验。
3. 论文/周报层面先采用“full 主结论优先，quick 与 gated 仅作探索性证据”的叙述。