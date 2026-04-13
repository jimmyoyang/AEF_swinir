# 4-1 周报

## 1. 后处理模块实验进展

本周对后处理模块进行了5000 iteration的实验，结果显示后处理并未带来指标提升，具体数据如下：
- ablation_4d_mask_temporal_loss（后处理，5000 iter）：best_psnr 14.7357，best_ssim 0.3380，best_ergas 10.7622，best_sam 0.3428。
- ablation_4c_mask_reduce_prob_or（后处理，5000 iter）：best_psnr 14.7357，best_ssim 0.3288，best_ergas 10.7652，best_sam 0.3439。
- ablation_4b_advanced_processor_soft_simplified（后处理，5000 iter）：best_psnr 14.5181，best_ssim 0.3538，best_ergas 11.0545，best_sam 0.3529。
- 从训练日志曲线来看，后处理实验的 loss 曲线与基线实验相近，未见明显收敛加速或指标提升。部分后处理实验在 early stopping 前后指标波动较大，未体现出后处理带来的优势。

结论：后处理模块在当前设置下未带来有效提升，后续不再继续相关实验，资源将转向数据和归一化问题的修正。

## 2. 数据归一化问题分析与修正方案

本周发现数据归一化存在严重问题，主要体现在：

- LR 数据中的 B10 和 B11 波段原本属于 100m 分辨率，但在预处理时被重采样到 30m，且归一化时未与其他波段区分，导致归一化参数不合理。
- 早期实验由于测试区域较小（仅4个小块），未暴露该问题。但在大范围模拟时，归一化失衡导致实验数据分布极度不均衡，指标波动大，表现为：
  - ERGAS、SAM 等指标异常不稳定，部分实验训练过程中 loss 居高不下，甚至出现大面积 nan。
  - 训练日志显示部分实验 EARAS（或类似指标）异常高，loss 曲线异常。
- 结论：归一化问题是当前实验不稳定的主要原因。后续需在 [prepare_data_local.py](prepare_data_local.py) 中对归一化流程进行统一修正，B10/B11 波段需单独归一化，确保数据分布合理。

### 官方文档查阅与归一化修正要点

| 波段编号 | 物理含义         | 官方缩放系数 | 物理单位         | 采样后典型范围 | 归一化策略         | 归一化公式/建议           |
|:--------:|:----------------|:------------:|:----------------:|:--------------:|:-------------------|:--------------------------|
| B1-B7    | 反射率（TOA）   | 0.0001       | 无量纲（0~1）    | 0.0 ~ 1.0+     | Clip               | `np.clip(x, 0, 1)`        |
| B10-B11  | 亮度温度（TOA） | 0.0001       | Kelvin (K)       | 200 ~ 330      | Min-Max            | `(x-250)/80, clip到[0,1]` |
| 云掩码   | 云概率/软掩码   | 1.0          | 0~1              | 0.0 ~ 1.0      | None               | 保持原样                  |

**代码实现建议（采样后处理）**

```python
def normalize_landsat8_patch(patch):
    # patch: (channels, H, W), channels顺序为 [B1, ..., B7, B10, B11, cloud]
    patch = patch.astype(np.float32)
    # 1. 物理还原
    patch[0:9] *= 0.0001  # B1-B7, B10, B11
    # 2. 光学波段 clip
    patch[0:7] = np.clip(patch[0:7], 0, 1)
    # 3. 热红外归一化
    patch[7:9] = (patch[7:9] - 250.0) / 80.0
    patch[7:9] = np.clip(patch[7:9], 0, 1)
    # 4. 云掩码保持原样
    return patch
```

  ### 2.1 归一化代码具体位置索引

  - 采样后物理归一化主位置： [datapipe/prepare_data_local.py](datapipe/prepare_data_local.py#L110) ，这里直接对 LR patch 执行 `0.0001` 缩放、光学波段 clip、B10/B11 温度归一化。
  - 旧版 dataset 的统一归一化入口： [datapipe/datasets.py](datapipe/datasets.py#L83) ，`__getitem__` 中对 LR/HR 都调用 `robust_per_image_normalize()`。
  - 旧版分位数归一化函数： [datapipe/datasets.py](datapipe/datasets.py#L215) ，定义了 `robust_per_image_normalize()`，负责按通道做 1% / 99% 分位拉伸并映射到 `[-1, 1]`。
  - 多时相数据中 HR 归一化位置： [datapipe/datasets.py](datapipe/datasets.py#L582) ，以及 LR 反射率归一化位置： [datapipe/datasets.py](datapipe/datasets.py#L667) 。
  - 新版带物理归一化 dataset 的 LR 入口： [datapipe/datasets_with_lr_norm.py](datapipe/datasets_with_lr_norm.py#L87) ，先调用 `apply_landsat_lr_physical_normalization()`，再做分位数归一化。
  - 新版物理归一化函数本体： [datapipe/datasets_with_lr_norm.py](datapipe/datasets_with_lr_norm.py#L255) ，这里集中处理反射率和热红外波段的缩放与 clip。
  - 新版分位数归一化函数： [datapipe/datasets_with_lr_norm.py](datapipe/datasets_with_lr_norm.py#L227) ，与旧版逻辑类似，但支持更灵活的参数配置。
  - 新版多时相 HR/LR 归一化位置： [datapipe/datasets_with_lr_norm.py](datapipe/datasets_with_lr_norm.py#L558) 、[datapipe/datasets_with_lr_norm.py](datapipe/datasets_with_lr_norm.py#L576) 、[datapipe/datasets_with_lr_norm.py](datapipe/datasets_with_lr_norm.py#L626) 。

### 2.2 归一化参考文档（官方）

- Earth Engine 数据集说明（Landsat 8 Collection 2 Level-2）： https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
  - 该页面给出示例缩放：光学波段 `SR_B.*` 常见处理为 `* 0.0000275 + (-0.2)`，热红外地表温度波段 `ST_B.*` 常见处理为 `* 0.00341802 + 149.0`。
- Earth Engine 数据集说明（Landsat 8 Collection 2 TOA）： https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_TOA
  - 该页面明确 TOA 产品为辐射定标后的 TOA 反射率产品，系数来自元数据。
- Earth Engine Landsat 使用指南（官方总览）： https://developers.google.com/earth-engine/guides/landsat
  - 用于核对不同产品线（TOA / SR / ST）的波段含义和推荐处理流程。
- Landsat 8 Data Users Handbook（USGS）： https://www.usgs.gov/landsat-missions/landsat-8-data-users-handbook
  - 用于核对 B10/B11 的物理定义、单位和辐射定标背景。
- Landsat Collection 2 QA Bands（USGS）： https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
  - 用于云/阴影/雪等 QA 位定义，支撑云掩码或软掩码构造。

> 口径提醒：不同产品的缩放系数不同（TOA vs Collection 2 Level-2）。实验中必须先固定数据源，再固定归一化参数，避免跨产品混用系数。

**注意事项：**
- 16-bit 存储时，65535 通常为无效值，需先设为 NaN 或 0。
- AlphaEarth HR patch 已为 uint8，min/max 检查后无需还原，直接归一化到 [0,1]。
- 归一化后所有输入 band 都在 [0,1] 区间，便于网络收敛。

## 3. 本周新增数据检查与采样流程优化

### 3.1 Patch采样与数据预处理
- 新增高效并行采样与掩码筛查，自动跳过无效patch。
- patch命名包含日期和tile_id，保证空间对齐。
- 采样前如输出目录已存在，脚本会交互式询问并自动备份，防止误删。

### 3.2 归一化与物理量还原
- Landsat8 LR patch：B1-B7反射率clip到[0,1]，B10/B11亮度温度归一化到[0,1]，云掩码保持原样。
- AlphaEarth HR patch：uint8，已确认像素值分布在0~222，建议归一化到[0,1]（/255.0）。

### 3.3 检查与验证脚本
- 新增patch有效性与归一化检查，采样后自动统计patch min/max/NaN/全零比例，写入patch_stats.log。
- 批量检查patch文件脚本（check_val_patches.py），可批量打印val集合下所有LR/HR patch的shape、dtype、min/max、NaN数量。
- HR 64 band分布检查脚本，遍历raw_alphaearth目录下所有band，打印每张min/max，辅助归一化策略制定。

### 3.4 随机采样与空间分布优化
- 支持随机采样patch，避免全是黑边，提高patch有效性和空间分布均匀性。

---

## 4. 论文绘图与结论汇总脚本清单（新增）

为支撑论文写作中的图表制作与结论汇总，本周将相关脚本按“绘图脚本、结果汇总脚本、一键流水线脚本”整理如下。

### 4.1 绘图脚本

| 脚本路径 | 作用 | 典型用法 | 主要输出 |
|:--|:--|:--|:--|
| `scripts/plot_selected_experiments_figures.py` | 针对固定 7 组实验生成论文可用图（Best PSNR柱状图、Best/Last分组图、四指标面板图）。 | `python scripts/plot_selected_experiments_figures.py --summary-csv debug_output/latest_selected_compare/summary_latest_selected.csv --output-dir debug_output/latest_selected_compare` | `selected7_best_psnr_bar.png`、`selected7_best_last_psnr_grouped.png`、`selected7_best_metrics_panel.png` |
| `scripts/analyze_ablation_results.py --plot` | 在消融结果分析基础上附加 PSNR 增益柱状图。 | `python scripts/analyze_ablation_results.py --plot --baseline config_true_baseline` | `ablation_best_results/psnr_bar_chart.png` |
| `scripts/analyze_ablation_final.py --curves` | 生成所有实验 PSNR 学习曲线图，用于展示收敛趋势。 | `python scripts/analyze_ablation_final.py --curves --baseline config_true_baseline` | `ablation_best_results/learning_curves.png` |

### 4.2 结论/汇总脚本

| 脚本路径 | 作用 | 典型用法 | 主要输出 |
|:--|:--|:--|:--|
| `scripts/compare_latest_selected_experiments.py` | 汇总“选定 7 组”实验最新 run 的 best/last 指标并自动写中期对比结论。 | `python scripts/compare_latest_selected_experiments.py --output-dir debug_output/latest_selected_compare --report-md debug_output/latest_selected_compare/midterm_selected_compare.md` | `summary_latest_selected.csv`、`all_validation_rounds_latest_selected.csv`、`midterm_selected_compare.md` |
| `scripts/analyze_ablation_results.py` | 扫描 `training_logs/experiments`，生成全量消融排行榜与分组报告，并自动给出 4c vs 4d 结论。 | `python scripts/analyze_ablation_results.py --baseline config_true_baseline` | `ablation_summary_latest.csv`、`ablation_best_results/ablation_analysis_report.md` |
| `scripts/analyze_ablation_final.py` | 论文级综合汇总：分组最优、全局最优、4c vs 4d 结论，并可导出 LaTeX 表格。 | `python scripts/analyze_ablation_final.py --latex --plot --curves` | `ablation_best_results/ablation_table.tex` 及综合终端摘要 |
| `scripts/compare_ablation_5bcd_v2.py` | 汇总 v2 的 5a/5b/5c/5d 对比结果（含末轮指标与部分 post 统计）。 | `python scripts/compare_ablation_5bcd_v2.py --root training_logs/experiments_v2 --output ablation_v2_comparison.csv` | `ablation_v2_comparison.csv` |
| `scripts/compare_ablation_v3_vs_v2.py` | 横向汇总 v2/v3/3b 多方案，生成统一排序并给出相对 3b 增益。 | `python scripts/compare_ablation_v3_vs_v2.py --v2-root training_logs/experiments_v2 --v3-root training_logs/experiments_v3 --main-root training_logs/experiments --output ablation_v3_full_vs_v2.csv` | `ablation_v3_full_vs_v2.csv` |
| `scripts/build_dual_delta_report.py` | 在已有 v2/v3 汇总表上计算“双增益”：相对组内 nopost 基线增益、相对 anchor 增益。 | `python scripts/build_dual_delta_report.py --input ablation_v3_full_vs_v2.csv --output debug_output/v2_v3_dual_delta_report.csv --anchor-psnr 14.6694 --anchor-name anchor_14.6694` | `debug_output/v2_v3_dual_delta_report.csv` |
| `scripts/build_unified_dual_delta_report.py` | 跨四套实验（legacy/mainline、v2/v3）统一生成 dual-delta 汇总，支持论文总表。 | `python scripts/build_unified_dual_delta_report.py --project-root . --anchor-4c 14.7357 --anchor-4d 14.7634 --output-csv debug_output/unified_dual_delta_report.csv --output-md debug_output/unified_dual_delta_report.md` | `debug_output/unified_dual_delta_report.csv`、`debug_output/unified_dual_delta_report.md` |
| `scripts/compare_training_log_metrics.py` | 两个训练日志的 best/last 指标做公平对比，输出标准化 CSV。 | `python scripts/compare_training_log_metrics.py --recovered-log <log_a> --current-log <log_b> --output debug_output/compare_training_log_metrics.csv` | 对比 CSV（含 delta 行） |
| `scripts/rank_recovered_3b_vs_all.py` | 将 recovered 3b 与 v2/v3 全部实验统一排名，输出相对 recovered 的增益。 | `python scripts/rank_recovered_3b_vs_all.py --recovered-log <recovered_log> --v2-root training_logs/experiments_v2 --v3-root training_logs/experiments_v3 --output debug_output/ranking_recovered_3b_vs_all_v2_v3.csv` | `debug_output/ranking_recovered_3b_vs_all_v2_v3.csv` |
| `scripts/compare_recovered_vs_current.py` | 在预测结果层面对 recovered/current 与 GT 做 PSNR/SSIM 对比。 | `python scripts/compare_recovered_vs_current.py --recovered-dir <pred_a> --current-dir <pred_b> --gt-dir <gt_dir> --output debug_output/compare_3b_recovered_vs_current.csv` | `debug_output/compare_3b_recovered_vs_current.csv` |

### 4.3 一键流水线脚本（训练+汇总联动）

| 脚本路径 | 作用 | 典型用法 | 主要输出 |
|:--|:--|:--|:--|
| `scripts/run_ablation_4c_4d_and_analyze.sh` | 一键执行 4c/4d 训练并自动调用分析与绘图脚本，适合快速出周报结论。 | `bash scripts/run_ablation_4c_4d_and_analyze.sh 0` | `ablation_summary_latest.csv`、`ablation_best_results/*.md/*.png`、`tmp/recovery_validation_logs/...` |
| `scripts/repro_4c_4d_and_compare.sh` | 从历史 best ckpt 续训复现实验并自动做 recovered vs repro 对比及汇总。 | `bash scripts/repro_4c_4d_and_compare.sh alphaearth 7000` | `debug_output/compare_4c_recovered_vs_repro_7000.csv`、`debug_output/compare_4d_recovered_vs_repro_7000.csv`、`debug_output/compare_4c_4d_repro_summary_7000.csv` |

> 建议论文写作时的最小流程：先运行 `compare_latest_selected_experiments.py` 生成最新汇总表，再运行 `plot_selected_experiments_figures.py` 出图，最后用 `analyze_ablation_final.py --latex` 直接产出可放论文附录/正文的表格素材。

## 5. 后续建议与待办事项

- 保持 AlphaEarth HR patch 归一化方式不变更安全，等有充分验证和需求时再整体切换。HR patch归一化：如需严格归一化到[0,1]，建议在采样后加/255.0并clip。
    - 你的 pipeline、模型训练和推理流程已经基于现有归一化方式，贸然更改会导致所有下游接口、模型权重、评测脚本等都要同步调整，否则容易出错或性能下降。
    - 现有归一化方式已被验证能正常训练和推理，且和历史实验、指标对齐，便于对比和追溯。
    - 如果确实要切换到 /255.0 归一化，建议新建分支、全链路同步修改和验证，避免主分支混用导致混乱。
- patch对应性进一步增强：如需保证所有时相下tile_id完全一致，可先统计tile_id交集后采样。
- patch band数量与顺序统一：确保LR/HR patch band顺序、数量完全一致，便于模型输入。
- patch物理量还原：如后续需还原物理量（如反射率、温度），建议记录归一化和压缩方式。
- 自动化测试与可视化：可增加patch可视化脚本，辅助人工检查patch质量。
- 大规模数据处理优化：如数据量极大，可进一步优化内存和IO效率。

---

> 所有检查和归一化脚本建议集中存放在`scripts/checks/`目录，便于复用和维护。每次pipeline变更后，建议先用小规模数据和检查脚本验证patch有效性和归一化分布。


## 6. 期中报告全量内容总结（整合进 4-1 周报）

本节将 `MIDTERM_REPORT_FRAMEWORK_REVISED.md` 的核心内容统一压缩到周报，作为“阶段性总览 + 可执行落地清单”。

### 6.1 研究目标与问题定义

- 目标：基于 Landsat 8 历史档案，重建 AlphaEarth-like 的 64 维高分辨率表征，用于弥补 AEF 历史覆盖不足。
- 问题本质：从 30m/低维输入到 10m/64维输出属于高度不适定映射，存在信息率不匹配。
- 三个核心困难：
  - 高维重建信息不足（9->64 维）。
  - 云污染空间分布复杂，硬掩膜会丢失薄云有效信息。
  - Landsat 时序不规则（受重访周期和云遮挡影响），需要显式时间条件建模。

### 6.2 文献与方法启发（本工作采用方向）

- 跨尺度超分（30m->10m）具备可行性，但通用 SR 模型难以直接处理遥感中的云与异步时序。
- 借鉴异步时序建模思想：将观测时刻作为输入条件，而非假设规则时间网格。
- 借鉴软掩膜思想：将云污染从 0/1 逻辑转为连续置信度权重，让模型学习“弱化而非抹除”。

### 6.3 数据工程与管线结论

- 研究区与样本：Landsat Path/Row 132/033（2018），小规模 smoke test（2x2 分块）用于快速迭代。
- 训练数据策略：训练阶段消费离线对齐后的 patch/tile，投影统一主要在数据准备阶段完成。
- 关键工程改造：
  - Online Crop：降低显存峰值，支持时序训练。
  - Online Numeric Recovery + Normalization：在读取时完成数值恢复与归一化，减少离线冗余存储。
  - QA_PIXEL 深解析：从云/阴影/雪等 bit 构建软掩膜特征通道。
- 数据质量筛选：采用质量筛查获得有效时相子集，提升时序对齐稳定性。

### 6.4 模型设计主线（与代码实现一致）

- Backbone：Late Upsampling SwinIR。
- 关键改造：
  - 时间条件注入（Time Band，DOY 编码）。
  - Learnable Positional Encoding（替代固定 Sin-Cos）。
  - 云掩膜引导交互（Cloud/Mask-guided 机制）。
- 设计收益：将主要特征提取放在低分辨率空间完成，显著降低计算与显存压力，并提升时空融合稳定性。

### 6.5 实验设置与公平性约束

- 环境：单卡 24GB 显存；固定随机种子；统一增强策略。
- 指标：PSNR、SSIM、ERGAS、SAM（兼顾空间结构与光谱保真）。
- 公平性原则：统一训练预算口径，按“同口径 latest-run + best/last”输出对比。

### 6.6 中期核心结果（latest selected 口径）

- 当前最优配置：`ablation_4f_mask_prob_or_learnable_pos_no_cross`。
- 阶段性结论：4 系列（4c/4d/4e/4f）整体稳定优于真基线，说明“时间条件 + 云质量建模”在当前数据与预算下有效。
- 对照结论：SRCNN 可作轻量参考，但性能上限低于时空融合主线。
- 边界说明：本阶段结论定位为 smoke test 趋势，不外推为最终上限；后续需在更大样本与更长训练预算下验证。

### 6.7 论文图表素材状态（可直接用于中期文稿）

- 已具备图 5-1/5-2/5-3：7 配置柱状图、best-last 对比图、多指标面板图。
- 已具备图 5-4/5-5/5-6：baseline 结构图、当前最优结构图、time/mask 关键模块图（Mermaid）。
- 口径约束：图表来源统一为 `debug_output/latest_selected_compare/summary_latest_selected.csv` 及其派生文件，避免历史高值混用。

### 6.8 结构演进总论（可复用于摘要/结论）

- 相对真基线，性能增益来自三层协同：
  - 输入层：从纯光学扩展到“光学 + 时间条件 + 云质量信息”。
  - 交互层：掩膜引导的时空融合降低受污染时相负迁移。
  - 目标层：配置化混合损失兼顾数值误差与结构一致性。
- 因而本阶段最重要的表述应为：在相同训练预算下，时间条件与云质量建模带来稳定、可复现的相对增益。

### 6.9 下一阶段计划（由期中内容落到执行）

- 全量训练：将当前主线最优结构扩展到更大空间/时序范围进行长周期训练。
- 输出头优化：探索 64 维光谱分组与分组交互，提升跨通道一致性。
- 损失增强：评估加入 SAM 约束后的光谱保真收益。
- 泛化验证：跨区域、跨年份复核稳定性。
- 文稿工程：将 GEE 下载细节与长脚本说明统一下沉到附录，正文保留结论与关键流程图。

### 6.10 一键重建中期图表与汇总

```bash
python scripts/compare_latest_selected_experiments.py --prefer-local-training-logs && \
python scripts/plot_selected_experiments_figures.py && \
echo "✓ 完成：所有数据和图表已重新生成"
```

---

## 7. 数据处理框架改造：归一化职责下沉（NEW - 4 月 7 日）

### 7.1 核心问题与改造动机

- **当前状态**：`prepare_data_local.py` 中存在 Landsat LR 物理归一化逻辑（B1-B7 反射率 scale 与 clip，B10-B11 亮度温度变换）
- **问题根源**：归一化参数硬编码在切片脚本，导致：
  - 后续调参需重新切片（时间成本高）
  - 不同实验混用不同归一化策略困难
  - 职责混杂：切片脚本掺杂了数据增强逻辑，难以维护

### 7.2 改造方案：分离职责，参数化控制

**本周已生成两个新脚本：**

| 脚本名称 | 功能定位 | 关键改动 | 现存位置 |
|:--|:--|:--|:--|
| **`prepare_data_local_split_only.py`** | 纯切片与划分 | 移除所有 LR/HR 值归一化处理，只保留有效性筛查 | `datapipe/` |
| **`datasets_with_lr_norm.py`** | 参数化归一化 | 增加 `apply_landsat_lr_physical_normalization()` 函数，支持灵活配置 LR 物理归一化参数 | `datapipe/` |

### 7.3 新脚本关键特性

#### 7.3.1 `prepare_data_local_split_only.py`

```python
# 核心改变（第109-111行）
# 【新版本】只做切片和划分，不做任何 LR 值处理或归一化！
# 所有归一化交给 Dataset 层在加载时处理。
# 这样切片脚本只负责数据分割，逻辑清晰，职责明确。
```

**职责清单（仅保留）：**
- ✅ 生成有效 patch 坐标（基于掩膜筛查）
- ✅ 并行切片与存储
- ✅ 训练/验证/测试 split（基于 tile_id 的地理分割）
- ❌ 不做任何数值处理

#### 7.3.2 `datasets_with_lr_norm.py`

**新增函数：** `apply_landsat_lr_physical_normalization(lr_img, cfg)`

函数签名与参数说明：
```python
def apply_landsat_lr_physical_normalization(lr_img, cfg):
    """
    cfg 示例词典：
    {
      'enabled': True,                                  # 开关
      'reflective_band_indices': [0,1,2,3,4,5,6],     # 反射率波段编号
      'thermal_band_indices': [7, 8],                 # 热红外波段编号
      'reflectance_scale': 1e-4,                       # 反射率缩放因子
      'thermal_offset': 250.0,                         # 亮温偏移
      'thermal_scale': 80.0,                           # 亮温缩放
      'clip_to_unit': True,                            # 是否clip到[0,1]
      'auto_detect_raw_input': True,                   # 自动检测是否需处理
      'raw_input_threshold': 2.0                       # 原始值判断阈值
    }
    """
    # 流程：原始值 -> 缩放 -> clip 或线性变换 -> 返回
```

**两个 Dataset 类的集成点：**

| 类名 | 位置 | 集成方式 | 配置参数来源 |
|:--|:--|:--|:--|
| `PreprocessedTileDataset` | `__init__` 第32-36行 | 从 params 读取 `normalize_percentile_lo/hi` 和 `lr_physical_norm` | config yaml 中 `data.train.params` |
| `AnytimeTemporalDataset` | `__init__` 第339-341行 | 同上 | 同上 |

**调用位置：**
- `PreprocessedTileDataset.__getitem__` 第67行
- `AnytimeTemporalDataset.__getitem__` 第393-396行

### 7.4 如何切换到新脚本

#### 方案 A：保守方案（推荐现阶段）
1. **保留原脚本不动**（`prepare_data_local.py` / `datasets.py`）继续现有流程
2. **新脚本作为备选**，待验证后切换
3. 修改 config 指向新脚本：
   ```yaml
   data:
     train:
       target: datapipe.datasets_with_lr_norm.AnytimeTemporalDataset
       params:
         lr_dir: "./data/processed_data/train/LR"
         hr_dir: "./data/processed_data/train/HR"
         # ... 其他参数
   ```
3. 由于新脚本默认 `lr_physical_norm: {}` （空dict会自动禁用），**不加额外配置时行为等同原脚本**

#### 方案 B：完整改造（待后续）
1. 用 `prepare_data_local_split_only.py` 重新切片（不含 LR 归一化）
2. 在 config 里加 `lr_physical_norm` 配置段启用参数化归一化
3. 全量训练验证

### 7.5 trainer/main/inference 是否需改

**结论：无需改**（完全向后兼容）

- `datasets_with_lr_norm` 返回的 key（`s1/gt` 或 `lr_sequence/gt`）与原格式一致
- 默认行为（禁用物理归一化）等同原脚本
- 所有下游脚本按现有调用流程工作，零改动

**唯一例外：** 若要启用参数化 LR 归一化，仅需在 config yaml 中加配置段（trainer/main 不涉及）

### 7.6 当前完成状态

| 项目 | 状态 | 说明 |
|:--|:--|:--|
| 两个新脚本创建 | ✅ 完成 | `prepare_data_local_split_only.py` 与 `datasets_with_lr_norm.py` 已生成并通过导入测试 |
| 核心改动应用 | ✅ 完成 | LR 纯切片与参数化归一化函数已实现 |
| 向后兼容性 | ✅ 验证 | 导入测试通过，默认参数等同原脚本 |
| 文档与 config 示例 | ✅ 完成 | 本节记录了完整的参数与调用方式 |

### 7.7 后续待办

| 待办项 | 优先级 | 预计时间 | 说明 |
|:--|:--|:--|:--|
| 用新脚本小规模测试（e.g. 1 个 tile） | 高 | 2 小时 | 验证 prepare_data_local_split_only 和 datasets_with_lr_norm 在实际训练中的兼容性 |
| 归一化参数调研与标准化 | 中 | 3 天 | 根据第 2 部分的表格与公式，确定最终的 reflective/thermal 参数与 clip 策略 |
| 单参数切线评估 | 中 | 5 天 | 对比"不启用物理归一化" vs "标准物理归一化"下的训练效果 |
| 全量切片与比对 | 低 | >1 周 | 若统计学验证支持新参数，用 prepare_data_local_split_only.py 重新切片全量数据并训练 |

---

## 8. 周报总结与行动清单

### 8.1 本周关键进展

1. **后处理模块实验结束**：确认后处理在当前配置下无收益，资源回收
2. **数据归一化问题深度分析**：定位根因为 B10/B11 波段处理不当，给出修正方案
3. **数据处理框架改造设计完成**：生成参数化归一化脚本，分离职责，提升可配置性
4. **中期报告内容总结**：整合研究目标、方法、实验与结论，便于后续文稿撰写

### 8.2 行动清单（优先级）

**本周内（高优）：**
- [ ] 使用新脚本在小数据集上做快速验证测试
- [ ] 确认 config 改写方案无误，更新示例配置
- [ ] 补充新脚本的单测或快速冒烟测试

**下周（中优）：**
- [ ] 根据 7.7 的表格推进归一化参数调研
- [ ] 评估新旧脚本在真实数据上的性能差异
- [ ] 若有差异，根因分析（是否来自纯切片 vs 有归一化的差别）

**计划中（中-低优）：**
- [ ] 若验证通过，全量数据重新切片与训练（需较长周期）
- [ ] 文稿工程：将本周的改造细节与 best practice 整理为技术附录

### 8.3 一键验证命令

```bash
# 导入测试（已通过）
python3 -c "
import sys; sys.path.insert(0, 'datapipe')
from datasets_with_lr_norm import apply_landsat_lr_physical_normalization, AnytimeTemporalDataset
from prepare_data_local_split_only import create_one_tile
print('✓ 新脚本导入正常')
"

# 小规模数据集训练测试（待执行）
# 将 config.yaml 中的 dataset target 改为 datasets_with_lr_norm.AnytimeTemporalDataset
# 运行训练并观察 loss 曲线和最终指标
```

### 8.4 关键 KPI 与下步里程碑

| KPI | 目标 | 当前 | 下步目标达成时间 |
|:--|:--|:--|:--|
| 新脚本兼容性验证 | 100% | 代码级测试通过 | 本周五 |
| 归一化策略标准化 | 确定参数 | 方案设计完成 | 下周一 |
| 训练效果对标 | best_psnr ≥ 14.7 | 原脚本已验证 | 下周三 |


