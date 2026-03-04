# NaN 值修复总结

## 一、问题分析

### 1.1 prepare_data_local.py 的处理逻辑

**关键发现**：
1. **第 77 行**：`if np.all(lr_tile == 0) or np.isnan(lr_tile).any(): return None` - 会跳过全0或包含NaN的瓦片
2. **第 166 行**：`lr_valid_mask = ~np.any(np.isnan(lr_data), axis=0) & np.all(lr_data != 0, axis=0)` - 要求所有通道都不为0
3. **第 182 行**：`VALID_PIXEL_RATIO_THRESHOLD = 0.98` - 只要98%的像素有效，瓦片会被保留

**潜在问题**：
- 瓦片中可能包含 **2% 的0值像素**（云区）
- 这些0值在归一化后会被映射到接近-1的值
- 如果使用归一化后的数据推断掩膜，可能无法正确检测云区

### 1.2 robust_per_image_normalize 的处理

**代码逻辑**（datasets.py 第 35-60 行）：
- 全0通道：直接返回全0（不会产生NaN）
- 分母为零：返回全0（不会产生NaN）
- 其他情况：正常归一化（不会产生NaN）

**结论**：`robust_per_image_normalize` 本身不会产生 NaN，但如果输入包含 NaN，NaN 会传播。

## 二、修复内容

### 2.1 在数据加载时添加 NaN 检查

**位置**：`datapipe/datasets.py` 第 159-170 行

**修复**：
```python
# 【安全检查】检查 NaN 和全0情况
if np.isnan(reflectance_data).any():
    print(f"⚠️  Warning: NaN found in {lr_path.name}, filling with 0")
    reflectance_data = np.nan_to_num(reflectance_data, nan=0.0)

# 检查是否全0（可能是全云，应该跳过）
if np.all(reflectance_data == 0):
    print(f"⚠️  Warning: All zeros in {lr_path.name}, skipping this timestep")
    continue  # 跳过该时相
```

### 2.2 在归一化后添加 NaN 检查

**位置**：`datapipe/datasets.py` 第 175-180 行

**修复**：
```python
normalized_reflectance = robust_per_image_normalize(reflectance_data)

# 【安全检查】归一化后再次检查 NaN
if np.isnan(normalized_reflectance).any():
    print(f"⚠️  Warning: NaN after normalization in {lr_path.name}, filling with 0")
    normalized_reflectance = np.nan_to_num(normalized_reflectance, nan=0.0)
```

### 2.3 修复掩膜推断逻辑

**问题**：简单模式下使用了 `normalized_reflectance > -0.9`，这是错误的。

**原因**：
- 数据下载时，云区被赋值为 0
- 归一化后，0值会被映射到接近-1的值
- 无法正确检测云区

**修复**：使用原始 `reflectance_data > 0` 检测云区

**位置**：`datapipe/datasets.py` 第 250-255 行

```python
# 【简单模式】必须使用原始 reflectance_data 而不是 normalized_reflectance
# 因为 normalized_reflectance 中0值会被映射到接近-1的值，无法正确检测云区
pixel_mask = (np.all(reflectance_data > 0, axis=0)).astype(np.float32)

# 【安全检查】确保掩膜中没有 NaN
if np.isnan(pixel_mask).any():
    print(f"⚠️  Warning: NaN in pixel_mask for {lr_path.name}, filling with 0")
    pixel_mask = np.nan_to_num(pixel_mask, nan=0.0)
```

### 2.4 在 HR 数据加载时添加 NaN 检查

**位置**：`datapipe/datasets.py` 第 154-170 行

**修复**：
```python
# 【安全检查】检查 HR 数据
if np.isnan(hr_img).any():
    print(f"⚠️  Warning: NaN found in HR {target_hr_path.name}, filling with 0")
    hr_img = np.nan_to_num(hr_img, nan=0.0)

hr_normalized = robust_per_image_normalize(hr_img)

# 【安全检查】归一化后再次检查
if np.isnan(hr_normalized).any():
    print(f"⚠️  Warning: NaN after HR normalization in {target_hr_path.name}, filling with 0")
    hr_normalized = np.nan_to_num(hr_normalized, nan=0.0)
```

### 2.5 在高级处理器中添加 NaN 检查

**位置**：`datapipe/datasets.py` 第 250-255 行

**修复**：
```python
pixel_mask, _ = self.cloud_mask_processor.process_pixel_mask(...)

# 【安全检查】确保掩膜中没有 NaN
if np.isnan(pixel_mask).any():
    print(f"⚠️  Warning: NaN in processed pixel_mask for {lr_path.name}, filling with 0")
    pixel_mask = np.nan_to_num(pixel_mask, nan=0.0)
```

## 三、修复总结

### 3.1 已修复的问题

1. ✅ **数据加载时的 NaN 检查**：在读取数据后立即检查并填充
2. ✅ **归一化后的 NaN 检查**：在归一化后再次检查
3. ✅ **掩膜推断逻辑修复**：使用原始数据检测云区，而不是归一化后的数据
4. ✅ **HR 数据的 NaN 检查**：确保 HR 数据也没有 NaN
5. ✅ **高级处理器的 NaN 检查**：确保处理后的掩膜没有 NaN

### 3.2 关键修复点

**最重要的修复**：简单模式下的掩膜推断必须使用原始 `reflectance_data`，而不是 `normalized_reflectance`。

**原因**：
- 数据下载时，云区被赋值为 0（`unmask(0)`）
- 归一化后，0值会被映射到接近-1的值
- 如果使用 `normalized_reflectance > -0.9`，可能无法正确检测云区
- 必须使用 `reflectance_data > 0` 来检测原始的0值（云区）

## 四、验证

现在所有数据加载和处理步骤都包含了 NaN 检查，确保：
1. 不会因为 NaN 导致 `default_collate` 错误
2. 不会因为 NaN 导致模型训练失败
3. 掩膜推断逻辑正确（使用原始数据检测云区）

## 五、建议

如果仍然遇到 NaN 问题，可以：
1. 检查原始数据文件是否包含 NaN
2. 检查 `prepare_data_local.py` 是否正确过滤了包含 NaN 的瓦片
3. 查看警告信息，了解哪些文件包含 NaN
