# prepare_data_local.py 问题分析与修复说明

## 问题原因

报错发生在 `datapipe/prepare_data_local.py` 的 STAGE 1 任务规划阶段：

```python
valid_band_count += (~np.isnan(band_src.read(1))).astype(np.uint8)
```

原代码先用第一个 AlphaEarth HR 波段的尺寸创建 `valid_band_count`，然后把后续波段整幅读出来直接相加。2020 数据里的 HR 波段并不完全同尺寸：

- `A00/A01/A02`: `(23936, 23512)`
- `A03` 及之后很多波段: `(23937, 23507)`

所以 NumPy 在相加时无法广播，触发：

```text
ValueError: operands could not be broadcast together with shapes ...
```

更深一层的问题是：这些 HR 波段不只是尺寸不同，左上角坐标和覆盖范围也不同。如果只做简单裁剪，虽然可能绕过 shape 报错，但不同波段会对应到不同地理位置，生成的 HR patch 会空间错位。

另外，2020 年的 Landsat LR 文件本身也有两套网格：

- 前 4 个时相是一个 LR 网格
- 后 15 个时相是另一个 LR 网格

因此 HR 有效掩膜也不能只按第一个 Landsat 文件缓存一次，必须按 LR 网格分别生成。

## 修复方法

我把 HR 读取逻辑改成“以 LR 网格为基准”的方式：

1. 新增 `lr_aligned_hr_transform()`  
   根据 LR 的 30m transform 生成对应的 10m HR transform，确保 HR patch 与 LR patch 在地理坐标上严格对齐。

2. 新增 `read_alpha_band_on_grid()`  
   读取每个 AlphaEarth 波段时，不再直接用 `r * 3, c * 3` 去读源文件窗口，而是把波段读到 LR 对齐的目标 HR 网格上。  
   如果 CRS 和分辨率一致，就走快速窗口读取；否则回退到 `rasterio.warp.reproject()`。

3. 新增 `build_hr_valid_mask_lr_res()`  
   STAGE 1 规划时，在统一的 LR-aligned HR 网格上统计每个 HR 像素有多少个有效波段，再按 `3x3` HR block 下采样回 LR 网格。这样不会再受不同 HR band shape 影响。

4. 新增 `alpha_band_valid_on_grid()` 快速路径  
   当前 AlphaEarth 输入是 `uint8` 且没有 nodata，STAGE 1 判断有效性时主要是判断覆盖范围。这个函数避免为了生成有效掩膜而反复整幅读取所有 HR 像素，速度更实际。

5. 新增 `lr_grid_cache_key()`  
   按 LR 文件的 CRS、尺寸和 transform 缓存 HR 掩膜。2020 数据有两套 LR 网格，所以会分别生成两份 HR valid mask，避免复用错误网格。

6. 修改 `create_one_tile()`  
   STAGE 2 真正导出 HR patch 时，也使用同一套 LR-aligned HR 读取逻辑，并把输出 HR GeoTIFF 的 transform 写成正确的 10m 对齐 transform。

## 验证结果

我做了这些检查：

- `python -m py_compile datapipe/prepare_data_local.py` 通过，说明脚本语法正确。
- 对两套 2020 Landsat LR 网格分别构建 64-band HR valid mask，均成功完成，没有再出现 shape broadcast 报错。
- 用真实数据调用 `create_one_tile()`，成功生成：
  - LR: `(10, 64, 64)`
  - HR: `(64, 192, 192)`
  - HR transform 与 LR patch 对应的 10m 网格一致。
- 做了一次非破坏性的端到端验证：把一个 Landsat 时相链接到 `/tmp`，运行 `main_processing(sample_num=1, pair_zero_max=1.0)`，成功完成 STAGE 1、STAGE 2、STAGE 3，并写出一对 LR/HR tif。

没有直接运行原始的无参数命令到结束，是因为它会备份/替换当前 `processed_data_SR_10m_32samples` 目录，并处理全部候选 tile，属于破坏性且耗时较长的完整导出。上述验证已经覆盖了这次崩溃所在的 STAGE 1 路径，以及后续 STAGE 2 写出路径。
