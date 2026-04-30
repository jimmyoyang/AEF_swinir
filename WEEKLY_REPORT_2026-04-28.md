# 4-28 周报（指标-观感不一致专项 + 逐样本评估工具落地）

## 1. 一句话总结

本周遇到并开始系统化处理一个关键现象：**推理/验证可视化在变清晰，但训练日志中的 PSNR 却在下降**；初步排查发现现有验证实现可能存在**只统计 batch 内第 1 个样本**的口径风险，已新增独立脚本用于 **batch_size=1 的逐样本指标导出**与**同一样本跨 checkpoint 并排对比**，用于把“观感变好”与“数值变差”拆解到可解释层面。

---

## 2. 现象与影响（必须在周报里明确）

### 2.1 现象描述

- 在某些 run 的训练过程中，保存的可视化结果主观上“更清晰/细节更好”。
- 但同一阶段 `training.log` 中 `📊 Validation Metrics` 的 **PSNR 呈下降或不升反降**。

### 2.2 为什么这件事重要

- 如果真实情况是“模型确实更好但 PSNR 没对齐”，需要调整评估协议/指标（例如更关注 SSIM、SAM、ERGAS、Masked-PSNR 或分层统计）。
- 如果是“验证统计口径有偏差/实现 bug”，则训练曲线与 best ckpt 选择可能被误导，直接影响实验结论。

---

## 3. 初步定位：验证口径风险点（代码级）

在 `trainer.py` 的验证实现中，发现指标计算/落盘可视化处存在 **仅使用 batch 中第 1 个样本 `[0]`** 的行为（当 `val.batch_size > 1` 时会导致：日志记录不是“整套 val 的平均”，而更像“每个 batch 的第一个样本的统计”）。

- 风险：
  - 当 val batch_size=8（常见设置）时，日志曲线可能受“被选中的第一个样本”强烈影响；
  - 数据集较小（例如 val=8）时，这种偏差会被放大，表现为曲线不稳定或与观感不一致。

> 结论：在完成修复/确认之前，**不应把训练日志中的平均 PSNR 趋势当作严格证据**；需要用独立评估脚本重算。

---

## 4. 本周落地：逐样本指标 + 多 checkpoint 并排对比脚本（已完成）

### 4.1 新脚本

- 脚本路径：`scripts/eval_run_per_sample_and_compare.py`
- 功能：
  1. 从 `<run_dir>/training.log` 解析 YAML 配置
  2. 重建 dataset（默认 `val` split），并自动探测输入通道数 `in_chans`
  3. 对指定 checkpoint 列表，使用 **DataLoader(batch_size=1)** 逐样本推理
  4. 输出每个样本的指标 CSV：PSNR/SSIM/ERGAS/SAM（以及可选 Masked-PSNR）
  5. 对指定样本 idx，保存跨 checkpoint 的并排图（LR/HR/Pred/Error map）

### 4.2 预期输出

- 指标 CSV：`<run_dir>/analysis/per_sample_metrics_<tag>.csv`
- 并排对比图：`<run_dir>/analysis/compare_sample_<idx>__<tags>.png`

### 4.3 目标 run（用于复现该现象）

- run 目录：
  - `training_logs/experiments/ablation_4f_mask_prob_or_learnable_pos_no_cross/2026-04-22_06-12-22`
- 对比 checkpoint：iter 500 vs iter 20000（若存在）

### 4.4 本周已完成一次跑通（关键结果可直接对外描述）

本周已在上述 run 上完成一次“全流程跑通”（逐样本评估 + 同一样本跨 checkpoint 并排图），关键终端输出与结论如下：

- Dataset 构建（val split）：
  - LR index：2329 tiles indexed
  - pair-zero filter：matched=160, kept_tiles=8（最终 val 评估样本数 = 8）
  - mask：soft + advanced processor = True；time/mask feature injection = enabled
- Checkpoint 加载：iter_500 / iter_20000 均 `missing=0, shape_mismatch=0`
- batch_size=1 重新评估得到的均值指标（val=8）：
  - iter_500：PSNR=13.1701，SSIM=0.2900，ERGAS=14.0140，SAM=0.4128
  - iter_20000：PSNR=12.8852，SSIM=0.2467，ERGAS=14.4461，SAM=0.4266
  - 结论（当前口径下）：**从 500 → 20000，PSNR/SSIM 均下降，ERGAS/SAM 变差**；与“观感变好”的主观结论存在冲突，需要进入逐样本分解与更严格可视化对照。

- 已生成落盘产物（用于下周分析）：
  - per-sample CSV：
    - `analysis/per_sample_metrics_iter_500.csv`
    - `analysis/per_sample_metrics_iter_20000.csv`
  - 同一样本跨 checkpoint 并排图：
    - `analysis/compare_sample_0__iter_500__iter_20000.png`

### 4.6 逐样本对齐分析（直接读取 CSV 得到的结论）

对 `per_sample_metrics_iter_500.csv` 与 `per_sample_metrics_iter_20000.csv` 做逐行对齐后（val 共 8 个样本），得到：

1. **不是单点 outlier 导致均值下降，而是 8/8 样本整体下降**：
  - 逐样本 PSNR 的变化量（iter_20000 - iter_500）分别为：
    - idx0: -0.2146, idx1: -0.2715, idx2: -0.3692, idx3: -0.2485
    - idx4: -0.2924, idx5: -0.3260, idx6: -0.3882, idx7: -0.1689
  - SSIM 也为 8/8 样本全部下降（与 PSNR 同向）。

2. **最好/最差样本（PSNR 视角）在两个 checkpoint 上一致**：
  - iter_500：best=idx5 (PSNR=13.7895)，worst=idx7 (PSNR=12.0579)
  - iter_20000：best=idx5 (PSNR=13.4635)，worst=idx7 (PSNR=11.8889)

3. **离群点判断（小样本 n=8，结论仅供参考）**：
  - iter_500 的 PSNR 分布按 Tukey IQR 规则，idx7（12.0579）处在轻微低端离群边界附近；
  - iter_20000 时 idx7 不再满足离群条件。
  - 结合结论 1（8/8 全下降），本次“均值下降”无法用单个离群样本解释。

4. **“观感变好但 PSNR 变差”目前仅能严格落到 sample_0（已生成对比图）**：
  - sample_0 的 PSNR 从 12.9731 → 12.7584（-0.2146 dB）。
  - 若对比图 `compare_sample_0__iter_500__iter_20000.png` 的主观观感确实更好，则这是一个明确的“观感↑/PSNR↓”样本。
  - 其余 idx1–idx7 尚未生成并排图，无法在周报中严谨声明其“观感变好/变差”；建议下周补齐 8 个样本的同样对比图后再下结论。

### 4.5 脚本可用性修复（避免复现失败）

首次运行时遇到 `ModuleNotFoundError: No module named 'datapipe'`，原因是用 `python scripts/xxx.py` 运行时默认不会把 repo root 放到 `sys.path`。已在脚本中加入“自动插入项目根目录到 sys.path”的逻辑，确保从任意工作目录执行都能导入项目模块。

注：终端提示 `torch.cuda.amp.autocast` FutureWarning（接口迁移到 `torch.amp.autocast`），不影响本次结果，但后续可顺手清理以减少噪声。

---

## 5. 本周结论（阶段性，不做过度外推）

- 该“观感提升但 PSNR 下降”的现象，**更可能是评估协议/口径问题或样本分布问题**，而不是一句话的“模型变差”。
- 在得到逐样本统计之前，训练日志里的平均指标趋势**不够可信**（尤其是 val batch_size>1）。

---

## 6. 下周计划（可执行、可验证）

1. 用 `scripts/eval_run_per_sample_and_compare.py` 在目标 run 上跑通：
   - 导出 `iter_500` 与 `iter_20000` 的逐样本 CSV
   - 生成 `sample_0` 的跨 checkpoint 并排对比图
2. 从逐样本 CSV 中定位：
   - “PSNR 下降的主贡献样本”是否为少数 outlier
   - PSNR 与 SSIM/SAM/ERGAS 的排序是否一致（判断是否“PSNR 不对齐但其它指标变好”）
3. 若确认 `trainer.py` 验证只统计 batch[0]：
   - 评估修复方案（对整个 batch/整个 val 求平均）
   - 回算若干关键 run 的验证曲线，修正 best checkpoint 选择口径

