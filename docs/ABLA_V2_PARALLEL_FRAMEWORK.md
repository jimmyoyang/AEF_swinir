# Ablation V2 并行框架说明（不改旧文件）

本文档说明为什么要新增并行框架、每个新增文件的作用、数据如何在各文件之间流动、关键张量维度如何变化、接口输入输出是什么，以及如何运行 quick/full 对比实验。

核心原则：
- 不修改现有主线文件。
- 方案1/3/6在独立命名空间中实现，方便反复实验和回滚。
- 与现有数据集契约保持一致，减少迁移成本。

## 1）为什么要新建这么多文件

你提到“文件太多”是对的。这里的拆分不是为了复杂化，而是为了把“会高频变化的实验代码”和“稳定主线代码”隔离。

拆分收益：
- 风险隔离：v2实验失败不会影响旧训练流程。
- 迭代效率：后处理模块可以独立替换，不动主干。
- 可观测性：训练器可以单独加统计（例如路由熵、温度）而不污染旧日志。
- 可维护性：每个文件只做一件事，定位问题更快。

如果只保留最小必要集合，必须文件其实只有三类：
- 独立网络入口。
- 独立后处理模块包。
- 独立训练器。

其余配置和脚本是“运行便利层”，不是算法必需层。

## 2）新增文件清单与职责

### 2.1 算法核心层

- models/post_v2/base.py
  - 定义统一后处理基类接口。
  - 约束所有方案都有相同调用方式，避免 trainer/network 写分支判断。

- models/post_v2/context.py
  - 运行时上下文桥接层。
  - 把训练状态（iter、stage等）传入后处理模块。

- models/post_v2/registry.py
  - 工厂/注册器。
  - 通过字符串 type 实例化方案1/3/6，配置可切换。

- models/post_v2/gumbel.py
  - 方案1（Gumbel动态路由）实现。
  - 负责路由mask、分组处理、统计项输出。

- models/post_v2/asymmetric_dual_branch.py
  - 方案3（异构双分支）实现。
  - 负责高频分支与低频分支处理及融合。

- models/post_v2/matrix_reshape_soef.py
  - 方案6（矩阵重塑SOEF）实现。
  - 负责通道重塑到伪2D网格并回写。

- models/post_v2/noop.py
  - 空操作模块。
  - 用于开关关闭/对照实验，保持调用链一致。

- models/network_swinir_post_v2.py
  - 并行网络入口。
  - 复用原始SwinIR主体，在 `conv_last` 后接 `post_v2`。

- trainer_post_v2.py
  - 并行训练器基础版。
  - 注入 runtime context，采集后处理统计，写 `post_v2_stats.csv`。

- trainer_post_v2_mask_ablation.py
  - 并行训练器（最佳预处理版）。
  - 支持 `indicating_mask_reduce=prob_or`，用于论文级公平对比。

### 2.2 运行配置层

- configs/ablation_v2/
  - quick/full 配置（5a no-post、5b、5c、5d）。
  - quick用于50-100 iter冒烟；full用于5000 iter正式训练。

### 2.3 执行与分析层

- scripts/run_ablation_5bcd_v2_quick.sh
  - 一键跑 5a(no-post) + 5b/5c/5d 的短程实验。

- scripts/run_ablation_5bcd_v2_full.sh
  - 一键跑 5a(no-post) + 5b/5c/5d 的正式长程实验。

- scripts/compare_ablation_5bcd_v2.py
  - 汇总日志/指标，输出对比结果表。

## 3）端到端数据流（重点）

### 3.1 输入批数据契约（与旧框架一致）

输入字典常见键：
- `lr_sequence`: `(B, T, C, H, W)` 或 `(T, C, H, W)`
- `timestamps`: `(B, T)` 或 `(T,)`
- `gt`: `(B, 64, H*scale, W*scale)`
- `mask`: 可选，时相有效性
- `mask_prob`: 可选，云概率图
- `indicating_mask`: 可选，监督掩膜

输出：
- 预测张量 `pred`: `(B, 64, H*scale, W*scale)`

### 3.2 维度变化主链路

以常见情况 `lr_sequence=(B,T,C,H,W)` 为例：

1. 时相展平
- 输入：`(B,T,C,H,W)`
- 变换：`view -> (B*T,C,H,W)`

2. 主干特征提取与时序聚合（复用原SwinIR）
- 输入：`(B*T,C,H,W)`
- 聚合后：`agg_feat -> (B,D,H,W)`，其中 `D=embed_dim`

3. 上采样与重建头
- `upsample(res)` 后空间放大：`(B,D,H*scale,W*scale)`
- `conv_last` 输出：`(B,64,H*scale,W*scale)`

4. v2后处理
- 输入：`x=(B,64,H*scale,W*scale)`
- 输出：`x_refined=(B,64,H*scale,W*scale)`
- 保持通道与分辨率不变，只做精炼

5. trainer计算loss
- `pred` 与 `gt` 对齐后进入损失函数
- 并可选记录后处理统计项

### 3.3 文件间调用关系

```text
main.py
  -> trainer_post_v2_mask_ablation.TrainerAlphaSRPostV2MaskAblation
      -> models/network_swinir_post_v2.SwinIRPostV2
          -> 原SwinIR主干前向
          -> models/post_v2/registry.py 构建 postprocessor
          -> postprocessor.forward(x)
      -> 训练/验证循环
      -> 保存 training.log + post_v2_stats.csv
```

## 4）统一接口定义（方便维护）

### 4.1 后处理模块接口

所有 `models/post_v2/*.py` 统一实现：

- `set_runtime_context(ctx: dict) -> None`
  - 注入运行状态。

- `forward(x: torch.Tensor) -> torch.Tensor`
  - 输入输出同shape，默认是 `(B,64,H,W)`。

- `get_last_stats() -> dict`
  - 返回最近一次前向统计，如温度、熵、激活组数。

- `get_last_extra_losses() -> dict`
  - 返回额外正则项（如有），trainer可选择加权并入总loss。

### 4.2 runtime context 字段

建议固定字段：
- `stage`: `train` 或 `val`
- `current_iter`: 当前迭代数
- `max_iters`: 总迭代数
- `batch_size`: 当前批大小

### 4.3 统计字段（默认）

- `post_tau`
- `routing_entropy`
- `active_groups`

注：方案3/6如果无上述字段，可返回空值或仅返回自己相关字段。

## 5）运行命令表（对比实验）

### 5.0 运行前验证（建议先做）

1. 语法检查（通过时仅显示最后 OK）

```bash
bash -n scripts/run_ablation_5bcd_v2_quick.sh \
&& bash -n scripts/run_ablation_5bcd_v2_full.sh \
&& python -m py_compile scripts/compare_ablation_5bcd_v2.py \
&& echo "[OK] syntax check passed"
```

2. 核对关键配置是否启用最佳预处理（可选）

```bash
grep -nE "use_pos_emb:|use_advanced_processor:|indicating_mask_reduce:" configs/ablation_v2/*.yaml
```

### 5.1 Quick（50-100 iter）

用于快速验通道、验shape、验日志：

```bash
bash scripts/run_ablation_5bcd_v2_quick.sh 0
```

说明：当前 quick 脚本只接收 1 个参数（GPU_ID），不支持命令行覆盖迭代数。

### 5.2 Full（5000 iter）

用于正式对比：

```bash
bash scripts/run_ablation_5bcd_v2_full.sh 0
```

### 5.3 单配置运行（手动）

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg_path configs/ablation_v2/ablation_v2_5a_nopost_bestpre_quick.yaml --mode train
CUDA_VISIBLE_DEVICES=0 python main.py --cfg_path configs/ablation_v2/ablation_v2_5b_scheme1_gumbel_quick.yaml --mode train
CUDA_VISIBLE_DEVICES=0 python main.py --cfg_path configs/ablation_v2/ablation_v2_5c_scheme3_dual_branch_quick.yaml --mode train
CUDA_VISIBLE_DEVICES=0 python main.py --cfg_path configs/ablation_v2/ablation_v2_5d_scheme6_matrix_quick.yaml --mode train
```

### 5.4 结果对比汇总

```bash
python scripts/compare_ablation_5bcd_v2.py --root training_logs/experiments_v2 --output ablation_v2_quick_comparison.csv
```

输出建议至少包含：
- 最佳/最后 PSNR、SSIM、SAM
- 训练耗时
- post_v2统计均值（如果有）

### 5.5 Full 实验后汇总

```bash
python scripts/compare_ablation_5bcd_v2.py --root training_logs/experiments_v2 --output ablation_v2_full_comparison.csv
```

## 6）验证流程（一步一步照做）

下面流程是“论文级公平对比”的最小执行路径：

1. 先做语法检查

```bash
bash -n scripts/run_ablation_5bcd_v2_quick.sh \
&& bash -n scripts/run_ablation_5bcd_v2_full.sh \
&& python -m py_compile scripts/compare_ablation_5bcd_v2.py
```

2. 跑 quick（含 no-post/1/3/6）

```bash
bash scripts/run_ablation_5bcd_v2_quick.sh 0
```

3. 生成 quick 对比表

```bash
python scripts/compare_ablation_5bcd_v2.py --root training_logs/experiments_v2 --output ablation_v2_quick_comparison.csv
```

4. 检查 quick 是否通过

```bash
cat ablation_v2_quick_comparison.csv
```

通过标准建议：
- 4组实验都出现在 CSV 中（5a/5b/5c/5d）。
- 每组都能解析出 `best_psnr`、`psnr`、`ssim`。
- 5b 路由统计不长期塌陷（`active_groups` 不长期接近 1）。

5. 通过后跑 full（5000 iter）

```bash
bash scripts/run_ablation_5bcd_v2_full.sh 0
```

6. 生成 full 对比表

```bash
python scripts/compare_ablation_5bcd_v2.py --root training_logs/experiments_v2 --output ablation_v2_full_comparison.csv
```

7. 出最终结论时，统一看 full 结果

```bash
cat ablation_v2_full_comparison.csv
```

## 7）日志与产物说明

每次运行目录通常包含：
- `training.log`: 训练主日志
- `post_v2_stats.csv`: 后处理统计

`post_v2_stats.csv` 常见列：
- `iter`
- `stage`
- `post_tau`
- `routing_entropy`
- `active_groups`

## 8）后续扩展规范

新增后处理方案时，按以下步骤：

1. 在 `models/post_v2/` 新建实现文件。
2. 继承基类并实现核心前向逻辑。
3. 在 `models/post_v2/registry.py` 注册别名。
4. 在 `configs/ablation_v2/` 新增 quick/full 配置。
5. 先跑 quick，确认shape/日志无误后再跑 full。

## 9）维护建议（避免再膨胀）

- 算法逻辑只放 `models/post_v2/`，不要把方案分支塞进 trainer。
- trainer 只做调度、日志、优化，不做算法细节。
- 一个配置只对应一个明确实验目的，命名带 `quick/full` 后缀。
- 对比脚本保持“只读日志”，不要包含训练逻辑。

## 10）V2 Full 结果统一结论（2026-03-20）

数据来源：
- `ablation_v2_full_comparison.csv`
- 命令：`python scripts/compare_ablation_5bcd_v2.py --root training_logs/experiments_v2 --output ablation_v2_full_comparison.csv`

### 10.1 关键指标总表（按 best_psnr 排序）

| rank | scheme | best_psnr | last_psnr | last_ssim | sam | ergas | active_groups | post_tau | routing_entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5d_scheme6_matrix | 14.5960 | 14.4787 | 0.3492 | 0.3526 | 11.0604 | 1.0 | 0.0 | 0.0 |
| 2 | 5a_nopost_bestpre | 14.5568 | 14.4733 | 0.3461 | 0.3528 | 11.0840 | - | - | - |
| 3 | 5b_scheme1_gumbel | 14.4919 | 14.2991 | 0.3235 | 0.3603 | 11.3288 | 4.0 | 0.5 | 1.3760 |
| 4 | 5c_scheme3_dual_branch | 14.1110 | 13.8071 | 0.2934 | 0.3814 | 11.9023 | 2.0 | 0.0 | 0.0 |

### 10.2 相对基线（5a）增减

以 `5a_nopost_bestpre` 作为对照，记 $\Delta=\text{scheme}-\text{5a}$：

- 5d（matrix）：
  - $\Delta$best_psnr = +0.0392
  - $\Delta$last_psnr = +0.0054
  - $\Delta$ssim = +0.0031
  - $\Delta$sam = -0.0002（更好）
  - $\Delta$ergas = -0.0236（更好）
- 5b（gumbel）：
  - $\Delta$best_psnr = -0.0649
  - $\Delta$last_psnr = -0.1742
  - $\Delta$ssim = -0.0226
  - $\Delta$sam = +0.0075（更差）
  - $\Delta$ergas = +0.2448（更差）
- 5c（dual_branch）：
  - $\Delta$best_psnr = -0.4458
  - $\Delta$last_psnr = -0.6662
  - $\Delta$ssim = -0.0527
  - $\Delta$sam = +0.0286（更差）
  - $\Delta$ergas = +0.8183（更差）

### 10.3 效果是否好：结论

- 结论1：V2 full 下，最优方案是 `5d_scheme6_matrix`，但对基线增益非常小（best_psnr +0.0392，last_psnr +0.0054）。
- 结论2：`5a_nopost_bestpre` 已经非常强，说明“最佳预处理”贡献显著，后处理模块总体增益空间有限。
- 结论3：`5b` 和 `5c` 均明显退化，尤其 `5c` 退化幅度大，不建议作为当前主推路线。

### 10.4 不好的原因（可能）

- 5b（gumbel）退化原因可能是“路由不够稳定”：
  - `routing_entropy=1.376` 且 `active_groups=4`，说明多组同时激活、分流较散。
  - 在 full 训练末期 `last_psnr` 比 `best_psnr` 低 0.1928，出现明显回落，疑似后期过拟合或温度/熵约束不足。
- 5c（dual_branch）退化原因可能是“分支能力不对称但融合策略偏硬”：
  - `active_groups=2` 但 `post_tau=0`、`routing_entropy=0`，表现像近似确定性路由。
  - 确定性分配可能导致部分频段长期被次优分支处理，误差累积到最终重建。
- 公共问题：
  - 后处理放在 `conv_last` 之后，输入已是 64 通道重建空间，结构先验较弱。
  - 在此位置做复杂路由/分支，收益可能不如在特征域（重建头之前）做轻量校正。

### 10.5 好的原因（可能）

- 5d（matrix）微弱领先原因可能是“低风险结构归纳偏置”：
  - `active_groups=1`、`post_tau=0`、`routing_entropy=0`，行为上接近稳定单路径变换。
  - 相比动态路由，矩阵重塑更像可控的通道-空间重排与局部混合，优化更平滑。
- 5a 强势原因可能是“预处理已解决主要误差来源”：
  - 无后处理也能达到接近最优，说明当前瓶颈更多在主干表达/数据分布，而不是输出端缺少复杂后处理。

### 10.6 后续怎么做（只给 V2 可执行项）

1. 以 `5d_scheme6_matrix` 作为 V2 默认后处理，仅保留“轻量、稳定”方向。
2. 对 5b 做最小改动复验：
   - 增加温度退火（高温到低温）并加熵下限/上限约束，避免长期过散或早塌陷。
   - 记录并对比 `post_v2_stats.csv` 的 epoch 级均值与方差，重点看 `active_groups` 和 `routing_entropy` 是否随训练后期失稳。
3. 对 5c 做融合改造后再评估：
   - 将硬融合改为可学习软融合（标量或逐通道门控），并加入残差保护：`y = x + alpha * f(x)`。
   - 限制新增参数量，先做 quick 再 full，避免把退化归因到过参。
4. 统一使用“最终判定看 full，不看 quick”规则，继续沿用当前文档第 6 节流程。
5. 论文叙述建议：
   - 主结论写“V2 中复杂动态路由并未带来稳定收益，轻量矩阵重排仅带来边际提升”。
   - 强调工程价值：并行框架验证了方案可插拔性，负结果同样缩小了搜索空间。

### 10.7 现阶段推荐结论（可直接引用）

- 推荐主线：`5a_nopost_bestpre`（简单、稳）或 `5d_scheme6_matrix`（指标最优但增益很小）。
- 不推荐主线：`5b_scheme1_gumbel`、`5c_scheme3_dual_branch`（full 指标整体落后，且后期稳定性不足）。
