# 5-26 组会汇报：2018-2024 多年份数据预处理与统一训练方案

## 1. 一句话总结

本阶段主要完成了 2018-2024 多年份 AlphaEarth/Landsat 数据预处理流程的检查与统一训练配置改造。核心问题有两个：一是 2020 年数据存在 HR 波段尺寸和地理网格不一致的问题；二是不同年份通过 `filter_processed_by_exact_T.py` 筛选后，每年的有效时相数 T 不一致。针对这些问题，我在预处理阶段改成以 LR 网格为基准对齐 HR，在训练阶段用多路径数据集、统一 padding 长度和 temporal mask 解决不同年份 T 不一致的问题。

---

## 2. 数据预处理范围

本次整理的数据覆盖 2018-2024 年。每一年经过处理后都按照 `train/val/test` 拆分，并保留 `LR` 与 `HR` 成对的 GeoTIFF patch。

当前已形成 filtered 数据根目录的年份包括：

| 年份 | filtered 数据状态 | exact T |
|---|---:|---:|
| 2018 | 已生成 | 20 |
| 2019 | 已生成 | 23 |
| 2020 | 预留，当前配置中先注释 | 19 |
| 2021 | 已生成 | 21 |
| 2022 | 已生成 | 19 |
| 2023 | 已生成 | 22 |
| 2024 | 已生成 | 22 |

其中 2019 年的 T 最大，为 23；2022 和 2020 为 19；2021 为 21；2023 和 2024 为 22。这意味着如果直接把不同年份放进同一个 batch，时间维度无法天然对齐，必须在 dataset 或 collate 前统一处理。

---

## 3. 2020 年预处理问题

2020 年是本次预处理中最特殊的一年，主要问题不是简单的文件缺失，而是输入数据本身存在网格不一致。

### 3.1 HR 波段尺寸不一致

在 `datapipe/prepare_data_local.py` 的 STAGE 1 规划阶段，原逻辑会用第一个 AlphaEarth HR 波段的尺寸初始化 `valid_band_count`，然后把后续 HR 波段整幅读入相加：

```python
valid_band_count += (~np.isnan(band_src.read(1))).astype(np.uint8)
```

但 2020 年的 AlphaEarth HR 波段尺寸不完全一致：

- `A00/A01/A02`: `(23936, 23512)`
- `A03` 及之后多个波段: `(23937, 23507)`

因此在 NumPy 相加时触发 shape broadcast 报错。更重要的是，这些波段不只是 shape 不同，左上角坐标和覆盖范围也存在差异。如果只做裁剪，虽然可能绕过报错，但会造成不同 HR band 在空间上错位。

### 3.2 LR 文件存在两套网格

2020 年 Landsat LR 文件也不是完全统一网格：

- 前 4 个时相属于一套 LR grid；
- 后 15 个时相属于另一套 LR grid。

所以 HR valid mask 不能只按第一个 Landsat 文件缓存一次，否则后续时相会复用错误的 HR 对齐关系。

---

## 4. 2020 问题的修复思路

针对 2020 年的网格问题，预处理逻辑改成“以 LR 网格为基准”的 HR 读取方式，而不是假设所有 HR band 有同一 shape。

主要改动包括：

1. 新增 `lr_aligned_hr_transform()`  
   根据 LR 的 30m transform 推导对应的 10m HR transform，保证 HR patch 与 LR patch 在地理坐标上严格对齐。

2. 新增 `read_alpha_band_on_grid()`  
   读取每个 AlphaEarth band 时，把它采样到 LR 对齐的目标 HR 网格上。如果 CRS 和分辨率一致，则使用快速窗口读取；否则回退到 `rasterio.warp.reproject()`。

3. 新增 `build_hr_valid_mask_lr_res()`  
   在 LR-aligned HR 网格上统计 HR 有效波段数，再按 `3x3` HR block 下采样回 LR 网格，避免不同 HR band shape 直接相加。

4. 新增 `lr_grid_cache_key()`  
   HR valid mask 的缓存键加入 LR 文件的 CRS、尺寸和 transform。这样 2020 的两套 LR grid 会生成两份独立 mask，不会错误复用。

5. 修改 `create_one_tile()`  
   真正导出 HR patch 时，也使用同一套 LR-aligned HR 读取逻辑，并写入正确的 10m transform。

验证结果：语法检查通过；两套 2020 LR grid 都能成功构建 HR valid mask；真实数据下可以成功生成一对 LR/HR patch，其中 LR 为 `(10, 64, 64)`，HR 为 `(64, 192, 192)`。

---

## 5. 多年份训练中的 T 不一致问题

`scripts/filter_processed_by_exact_T.py` 会按 tile 统计 LR 文件数量，并只保留恰好等于 `target_t` 的 tile。因此每一年内部是 exact T，但跨年份之间 T 不一样。

如果直接合并不同年份目录，会有两个风险：

1. 时间维度不一致，DataLoader 无法直接组成 batch；
2. 不同年份可能有相同 `tile_id`，如果简单把所有文件混进一个目录，原始 `AnytimeTemporalDataset` 会按 `tile_id` 聚合，导致 2019 和 2024 等年份的时序被错误合并成一个更长序列。

因此不能通过简单拷贝或软链接所有年份到同一个目录来解决。

---

## 6. 训练端统一方案：多路径数据集 + mask

训练端新增了 `datapipe.datasets.MultiPathAnytimeTemporalDataset`。它的核心思路是：

- 每个年份仍然作为一个独立的 `AnytimeTemporalDataset`；
- 每个年份内部继续按自己的 exact T 读取时序；
- 最外层只做 dataset concat，让 DataLoader 看到一个统一的大数据集；
- 相同 `tile_id` 在不同年份不会互相混合。

新的默认训练配置为：

```bash
configs/ablation/ablation_4f_multiyear_2018_2024.yaml
```

其中 2020 已经写入配置但先注释掉：

```yaml
# - ".../data/2020-charles/processed_data_SR_10m_filer"
# - "2020_T19"
```

当前启用的年份为 2018、2019、2021、2022、2023、2024。

### 6.1 统一时间长度

配置中设置：

```yaml
fixed_temporal_len: 23
```

原因是 2019 年 T=23，为当前最大长度。其它年份会 padding 到 23：

- 真实时间步保留；
- 不足的位置补零；
- `timestamps`、`lr_sequence`、`mask`、`mask_prob` 等时间相关张量一起补齐；
- temporal mask 中 padding 位置为 0，模型融合时可以忽略这些无效时间步。

这样既不丢掉 2019 的 23 个时相，也不强行重复其它年份的时相。

### 6.2 为什么关闭二次 HR zero filter

新配置中设置：

```yaml
enable_hr_zero_filter: false
```

原因是这些 filtered 目录已经由 `filter_processed_by_exact_T.py` 按 exact T 筛过。如果训练时再启用旧的 pair-zero filter，某些 tile 的个别 timestep 会被二次删除，实际 T 就不再等于目录筛选时的 T，反而破坏这次多年份统一训练的前提。

---

## 7. 当前验证结果

多年份训练配置已经可以成功构建 train/val/test 数据集。

当前启用年份下的数据量为：

| split | 总 tile 序列数 |
|---|---:|
| train | 28196 |
| val | 14081 |
| test | 14114 |

按年份抽样检查后，真实时相数与预期一致，同时模型输入统一为长度 23：

| 年份 | 真实 T | 输出时间长度 |
|---|---:|---:|
| 2018 | 20 | 23 |
| 2019 | 23 | 23 |
| 2021 | 21 | 23 |
| 2022 | 19 | 23 |
| 2023 | 22 | 23 |
| 2024 | 22 | 23 |

抽样得到的 `lr_sequence` shape 为：

```text
(23, 11, 64, 64)
```

其中 11 个通道来自 9 个 reflectance band，加上 time band 和 mask band。

---

## 8. 本周结论

本周完成了从数据预处理到训练配置的一条完整梳理：

- 2020 年的主要问题是 HR band 尺寸、覆盖范围和 LR grid 不一致，不能用简单裁剪解决；
- 预处理端已改为以 LR grid 为基准读取和对齐 HR，避免空间错位；
- 多年份训练端不能简单合并文件目录，否则会错误聚合相同 `tile_id`；
- 新增多路径 dataset 后，每一年保持独立时序，外层统一成大数据集；
- 不同年份 T 不一致的问题通过 `fixed_temporal_len=23` 和 temporal mask 解决；
- 当前 2018、2019、2021、2022、2023、2024 已经可以进入统一训练，2020 等 filtered root 完成后再取消注释加入。

---

## 9. 下一步计划

1. 完成 2020 年 filtered root 的正式生成，确认 T=19 的 train/val/test 结果稳定。
2. 将 2020 从配置注释中恢复，重新检查多年份总数据量和每年真实 T。
3. 启动多年份训练，重点观察：
   - 不同年份样本是否被均匀采样；
   - padding timestep 是否被 mask 正确忽略；
   - 加入 2020 后训练是否出现分布异常。
4. 后续可考虑在日志中增加 year/source 标识，方便分析某一年是否对 loss 或 validation 指标有明显影响。
