# 6-10 组会汇报：AEF 时间对齐 + Cosine 嵌入回归训练链路落地

## 1. 一句话总结

本周在上周「2018–2024 多年份统一训练」基础上，把任务从 **Landsat 反射率超分** 推进到 **AlphaEarth 64 维嵌入向量回归**：新增同年时间对齐（`same_year`）、uint8 反量化归一化、`AEFEmbeddingLoss`（L1 + Cosine）及以 Cosine 为主指标的验证流程；并完成 **Large SwinIR + dequant** 配置的 **20k iter 全流程训练**。最终验证集（抽样 8 个样本）达到 **PSNR 33.50 dB / Cosine 0.9269（夹角约 21.6°）**，训练曲线与可视化均显示模型已能稳定拟合 AEF 嵌入目标。

---

## 2. 本周工程改动

### 2.1 任务定义切换：从反射率 SR → AEF 嵌入回归

| 维度 | 上周（`ablation_4f_multiyear`） | 本周（`aef_time_aligned_cosine_*`） |
|---|---|---|
| 监督目标 | HR 反射率（robust normalize） | AlphaEarth 64-band 嵌入（uint8 反量化 + L2 normalize） |
| 损失函数 | `MixedLoss`（L2 + SSIM） | `AEFEmbeddingLoss`（L1 + Cosine，向量归一化后计算） |
| 模型输出 | 反射率幅值 | `output_l2_normalize: true`，输出单位向量 |
| 主验证指标 | PSNR | **Cosine**（`primary_metric: cosine`），同时记录 PSNR / Angle |

核心动机：AlphaEarth 嵌入位于单位超球面上，**Cosine 比像素域 PSNR 更贴合几何意义**；训练与选 best ckpt 已按 Cosine 驱动。

### 2.2 数据侧：同年时间对齐（`time_alignment: same_year`）

在 `datapipe/datasets.py` 的 `AnytimeTemporalDataset` 中新增：

- **`time_alignment: same_year`**：每个 tile 的 LR 时序只保留与 HR 目标 **同一年** 的时间步，避免多年份混训时跨年份错误配对。
- **`hr_target_strategy: first_in_year`**：HR 监督取该 tile 在目标年份内的首个可用 HR 文件。
- **`time_alignment_strict: true`**：严格模式，对齐失败直接报错而非静默回退。
- **HR 读取索引**：新增 `_tile_to_hr_files` 与 `_resolve_hr_target_path()`，按 tile + 年份显式选择 HR 目标。

仍保留上周能力：`fixed_temporal_len: 23` padding、多年份 `MultiPathAnytimeTemporalDataset` 外层 concat、各年独立时序不混 tile。

### 2.3 HR 归一化：AEF uint8 反量化

新增 `hr_normalization: aef_uint8_dequantize`：

```python
# datapipe/datasets.py
def aef_uint8_dequantize(img, power=2.0, scale=127.5, offset=127.0):
    # uint8 [0,255] → 反量化到嵌入空间，再 L2 normalize
```

配置中同时开启 `hr_l2_normalize: true`、`hr_clip: true`，与模型 `output_l2_normalize` 对齐。

新增审计脚本 `scripts/audit_aef_io_distribution.py`，用于抽样检查 Dataset 实际读入的 LR/GT 分布是否与预期一致。

### 2.4 训练器与验证协议增强（`trainer.py`）

- 新增 **`AEFEmbeddingLoss`** 与 **`compute_embedding_cosine_metrics()`**。
- 验证日志扩展：**Cosine / Angle / Masked-Cosine / per-band PSNR / 样本级 spread**。
- 支持 `primary_metric: cosine` 选 best ckpt，导出至 `best_ckpts/aef_time_aligned_cosine_large_dequant/model.pth`。
- 修复 **device 处理**（commit `427a716`），避免部分张量未正确上 GPU。
- 增强 `scripts/eval_run_per_sample_and_compare.py`、`scripts/eval_previous_train_sh_sample_cosine.py`，便于对旧 run 做 Cosine 对齐评估。

### 2.5 配置与默认入口

| 配置文件 | 说明 |
|---|---|
| `configs/ablation/aef_time_aligned_cosine_default.yaml` | 默认 SwinIR（embed_dim=180, 4 stage），cache 指向各年 `processed_data_*/cache/` |
| `configs/ablation/aef_time_aligned_cosine_large_dequant.yaml` | **本周主力**：Large SwinIR（embed_dim=240, 9 stage），独立 tile cache 路径，`require_tile_cache: true` |

`train.sh` 默认配置已切换为：

```bash
bash train.sh 0 configs/ablation/aef_time_aligned_cosine_large_dequant.yaml train
```

---

## 3. 数据规模（与上周一致）

多年份 6 年（2018/2019/2021/2022/2023/2024，2020 仍注释）：

| split | tile 时序样本数 |
|---|---:|
| train | 28,196 |
| val | 14,081 |
| test | 14,114 |

每个样本：`lr_sequence` shape `(23, 11, 64, 64)`（9 reflectance + time + mask），`gt` shape `(64, 192, 192)`（AEF 64-band HR 嵌入）。

---

## 4. 训练实验与数值结果

### 4.1 主力 Run（已完成 20k）

- **Run 目录**：`training_logs/experiments/aef_time_aligned_cosine_large_dequant/2026-06-10_09-24-56/`
- **配置**：`aef_time_aligned_cosine_large_dequant.yaml`
- **训练**：batch=12，20,000 iter，最终 Train Loss **0.063**（初始约 1.13）
- **验证协议**：每 500 iter 验证一次；`val_max_batches=2`，`val_max_samples=8`（val 全量 14,081，日志为**固定前 8 样本**子集均值，便于纵向对比）

### 4.2 验证指标里程碑（val 子集 n=8）

| Iter | PSNR (dB) | Cosine | Angle (°) | 备注 |
|---:|---:|---:|---:|---|
| 0（随机初始化） | — | — | — | 预测为彩色噪声，Error Map 均匀高误差 |
| 500 | 29.11 | 0.7250 | 42.69 | 已学到全局色调，结构仍模糊 |
| 1,000 | 30.94 | 0.8566 | 30.68 | Cosine 快速上升 |
| 4,500 | 32.74 | 0.9119 | 23.81 | PSNR 突破 32 dB |
| 10,000 | ~33.0 | ~0.918 | ~23 | 进入平台期 |
| 19,500 | 33.49 | 0.9265 | 21.66 | 接近收敛 |
| **20,000（final / best）** | **33.50** | **0.9269** | **21.62** | best ckpt |

**iter 20,000 样本级 spread**（同一固定 8 样本）：

- PSNR：min=32.37 / median=33.26 / max=**35.36**（sample_0，2018 年 tile）
- Cosine：min=0.904 / median=0.922 / max=**0.947**

训练曲线见：`2026-06-10_09-24-56/training_curves.png`

- **Loss**：1.1 → 0.06，前 5k iter 下降最快，15k 后趋于平稳。
- **PSNR**：29 → 33.5 dB，全程单调上升。
- **Cosine**：0.73 → 0.93，与 Loss 同步改善。

### 4.3 与 default 小模型 Run 的对比（未完成）

早期 run `aef_time_aligned_cosine/2026-06-03_09-14-45` 在 iter 18,000 处中断，且 val PSNR 长期停留在 **~12.5 dB**、Cosine **~0.60**，与 large_dequant 差异极大。初步判断与 **cache 路径 / tile cache 未对齐 dequant 流程** 有关；本周 commit `427a716` 已修正 cache 路径后，large 配置复跑结果正常。default 配置需用修正后的 cache 策略 **重跑对照**。

### 4.4 与上周 reflectance 消融的量级差异（供组会说明）

上周 `ablation_4f_multiyear` 在 reflectance 任务上 val PSNR 约 **12–13 dB**（MixedLoss + 不同监督空间）。本周切换到 AEF 嵌入空间后，在 **dequant + L2 normalize + Cosine loss** 设定下 PSNR 升至 **33+ dB**。**两者指标不可直接横向比**，需按任务定义分开汇报；本周主结论应看 **Cosine 0.93 / Angle 22°**。

---

## 5. 可视化效果说明

可视化路径：`2026-06-10_09-24-56/images/val/iter_*_sample_*.png`  
布局统一为 2×2：**Input LR | Prediction | Ground Truth | Absolute Error Map**。

> 说明：GT 与 Prediction 使用 AEF 嵌入的 RGB 投影（`rgb_chn: [2,1,0]`），与 Input LR 的自然色 Landsat 反射率**不在同一色彩空间**，因此 GT 呈紫粉色平滑纹理属正常现象，不宜用 LR 外观直接对比 GT。

### 5.1 iter 0（训练前）

- **Prediction**：彩色随机噪声。
- **Error Map**：误差 0.05–0.09，全图均匀，模型尚未学习。

### 5.2 iter 500

- **Prediction**：已接近 GT 的紫粉色调，但偏模糊，细节未成形。
- **Error Map**：误差降至 0.022–0.036，Input LR 对角结构处误差仍偏高。
- 对应指标：PSNR 29.1 / Cosine 0.73。

### 5.3 iter 10,000

- **Prediction** 与 **GT** 在色调和大尺度纹理上已非常接近。
- **Error Map**：误差收窄至 0.012–0.026，高误差仍沿 Input LR 结构分布。
- 主观观感：从「模糊色块」过渡到「纹理对齐但偏平滑」。

### 5.4 iter 20,000（最终）

- **Prediction** 与 **GT** 肉眼难辨，大尺度紫粉纹理一致。
- **Error Map**：误差进一步降低，高误差带变窄、幅度减弱；sample_0 PSNR 达 **35.36 dB**。
- 主观观感：嵌入空间回归已收敛；若需评估地物细节，建议后续增加 **per-band Cosine 热力图** 或 **embedding 空间 t-SNE** 分析。

---

## 6. 本周结论

1. **任务链路打通**：多年份 Landsat → SwinIR → AlphaEarth 64D 嵌入的完整训练/验证/存 ckpt 流程已跑通。
2. **时间对齐必要**：`same_year` + `first_in_year` 避免跨年份 LR/HR 错配，是本周指标能正常上升的前提。
3. **归一化与 cache 关键**：uint8 反量化 + L2 normalize 必须与 tile cache 路径一致；修正 cache 后 large 模型 20k iter 稳定收敛。
4. **主指标 Cosine 0.927**：固定 val 子集上夹角约 22°，满足「嵌入方向对齐」的阶段性目标。
5. **可视化与指标一致**：iter 0→500→10k→20k 预测从噪声到与 GT 贴合，与 PSNR/Cosine 曲线一致，未出现上周「观感变好但 PSNR 下降」的异常（本周验证已按多样本聚合 + 固定子集 spread 记录）。

---

## 7. 产物索引

| 类型 | 路径 |
|---|---|
| 最佳权重 | `best_ckpts/aef_time_aligned_cosine_large_dequant/model.pth` |
| 完整 20k ckpt | `.../2026-06-10_09-24-56/ckpts/model_20000.pth` |
| 训练日志 | `.../2026-06-10_09-24-56/training.log` |
| 训练曲线 | `.../2026-06-10_09-24-56/training_curves.png` |
| 验证可视化 | `.../2026-06-10_09-24-56/images/val/iter_*_sample_*.png` |
| 主力配置 | `configs/ablation/aef_time_aligned_cosine_large_dequant.yaml` |

---

## 8. 下一步计划

1. **default 小模型对照重跑**：在修正 cache 后复跑 `aef_time_aligned_cosine_default.yaml`，与 large 模型比参数量/速度/精度 trade-off。
2. **全量 val/test 评估**：当前日志仅固定 8 样本；用 `scripts/eval_run_per_sample_and_compare.py` 对 best ckpt 跑完整 val/test（batch_size=1），导出 per-sample CSV。
3. **开启 SSIM/ERGAS/SAM**（可选）：`val_compute_expensive_metrics: true` 做辅助参考，但主指标仍建议 Cosine。
4. **2020 年数据**：filtered root 就绪后取消配置注释，检查 `same_year` 对齐下总样本量与指标是否漂移。
5. **推理与下游**：用 `inference.py` + test split 验证泛化，并抽样对比旧 reflectance 消融与 AEF 嵌入方案的下游可用性。
