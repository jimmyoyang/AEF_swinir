import rasterio
import numpy as np
import os

# 选择一个原始未分割的大影像文件
file_path = "./data/raw_alphaearth/AlphaEarth_Path132_Row33_reprojected_A00.tif"

# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"File not found: {file_path}")

# with rasterio.open(file_path) as src:
#     img = src.read()  # (bands, H, W)
#     print(f"File: {file_path}")
#     print(f"  dtype: {img.dtype}")
#     print(f"  shape: {img.shape}")
#     print(f"  min: {img.min()}  max: {img.max()}")
#     for c in range(img.shape[0]):
#         print(f"    band {c}: min={img[c].min()}  max={img[c].max()}")
def print_band_nan_info(file_path, tag):
    print(f"\nFile: {file_path}")
    with rasterio.open(file_path) as src:
        arr = src.read()  # (bands, H, W)
        print(f"  dtype: {arr.dtype}")
        print(f"  shape: {arr.shape}  (bands, H, W)")
        for b in range(arr.shape[0]):
            band = arr[b]
            nan_count = np.isnan(band).sum()
            total_count = band.size
            valid_count = total_count - nan_count
            nan_ratio = nan_count / total_count
            min_val = np.nanmin(band)
            max_val = np.nanmax(band)
            print(f"  {tag} band {b}: min={min_val}  max={max_val}")
            print(f"    NaN count: {nan_count} / {total_count} ({nan_ratio:.2%})")
            if nan_count == total_count:
                print("    ⚠️ 全部为NaN！")
            elif nan_count > 0:
                print("    部分为NaN。")
            else:
                print("    无NaN。")

# 修改为你的 LR 文件路径
# lr_file = "./data/raw_landsat/L8_13233_20180218_Masked.tif"
hr_file = "./data/raw_alphaearth/AlphaEarth_Path132_Row33_reprojected_A00.tif"
print_band_nan_info(hr_file, "HR")
# print_band_nan_info(lr_file, "LR")
# import os
# import glob
# import rasterio
# import numpy as np

# def print_band_info(file_path, tag):
#     print(f"\nFile: {file_path}")
#     with rasterio.open(file_path) as src:
#         arr = src.read()  # (bands, H, W)
#         print(f"  dtype: {arr.dtype}")
#         print(f"  shape: {arr.shape}  (bands, H, W)")
#         print(f"  min: {arr.min()}  max: {arr.max()}")
#         for b in range(arr.shape[0]):
#             print(f"    {tag} band {b}: min={arr[b].min()}  max={arr[b].max()}")

# # 修改为你的原始 HR/LR 路径
# hr_files = sorted(glob.glob("./data/raw_alphaearth/AlphaEarth_Path132_Row33_reprojected_A13.tif"))
# lr_files = sorted(glob.glob("./data/raw_landsat/L8_13233_20180218_Masked.tif"))

# if hr_files:
#     print("=== HR 原始影像检查 ===")
#     print_band_info(hr_files[0], "HR")
# else:
#     print("未找到 HR 原始影像")

# if lr_files:
#     print("\n=== LR 原始影像检查 ===")
#     print_band_info(lr_files[0], "LR")
# else:
#     print("未找到 LR 原始影像")