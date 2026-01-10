# 文件路径: debug_mask.py
# (v7.1 - 修复了 'Window' is not defined 的错误)

import rasterio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from rasterio.windows import Window  # 【修复】新增缺失的 Window 导入

# --- 1. 请在此处配置与主脚本相同的参数 ---
BASE_DATA_DIR = Path("./data")
RAW_LANDSAT_DIR = BASE_DATA_DIR / "raw_landsat"
RAW_ALPHA_DIR = BASE_DATA_DIR / "raw_alphaearth"
DEBUG_OUTPUT_DIR = Path("./debug_output")
DEBUG_DATE = '2018-01-01'
REF_PATH, REF_ROW = 132, 33
NUM_ALPHA_BANDS_TO_STACK = 64
ALPHAEARTH_BANDS_TO_USE = [f'A{i:02d}' for i in range(NUM_ALPHA_BANDS_TO_STACK)]
SCALE_FACTOR = 3
HR_BAND_REQUIREMENT_RATIO = 0.9

# --- 2. 调试主逻辑 ---
def run_mask_debug():
    print("===== Starting Mask Debugging Script =====")
    DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)
    date_str = DEBUG_DATE.replace('-', '')
    
    try:
        landsat_path = list(RAW_LANDSAT_DIR.glob(f'L8_{REF_PATH}{REF_ROW}_{date_str}_Masked.tif'))[0]
        alpha_paths_str = [str(p[0]) for p in [list(RAW_ALPHA_DIR.glob(f"AlphaEarth_Path{REF_PATH}_Row{REF_ROW}_reprojected_{b}.tif")) for b in ALPHAEARTH_BANDS_TO_USE]]
        if any(not p for p in alpha_paths_str): raise FileNotFoundError("Some AlphaEarth bands are missing!")
            
        print(f"[INFO] Debugging with Landsat: {landsat_path.name}")

        with rasterio.open(landsat_path) as lr_src, rasterio.open(alpha_paths_str[0]) as first_hr_src:
            # 同主脚本，首先确定最终的处理尺寸
            lr_h_orig, lr_w_orig = lr_src.height, lr_src.width
            hr_h_orig, hr_w_orig = first_hr_src.height, first_hr_src.width
            effective_lr_h = hr_h_orig // SCALE_FACTOR
            effective_lr_w = hr_w_orig // SCALE_FACTOR
            proc_h = min(lr_h_orig, effective_lr_h)
            proc_w = min(lr_w_orig, effective_lr_w)
            print(f"[INFO] Effective processing dimensions (LR): {proc_h} x {proc_w}")

            # 使用严格的处理尺寸读取LR数据
            print("[INFO] 1. Generating LR valid mask...")
            lr_data = lr_src.read(window=Window(0, 0, proc_w, proc_h))
            lr_valid_mask = ~np.any(np.isnan(lr_data), axis=0) & np.all(lr_data != 0, axis=0)
            plt.imsave(DEBUG_OUTPUT_DIR / "1_lr_valid_mask.png", lr_valid_mask, cmap='gray')
            print(f"   - Saved to '1_lr_valid_mask.png'. Shape: {lr_valid_mask.shape}")

            print("[INFO] 2. Generating HR valid mask (memory-efficiently)...")
            valid_band_count = np.zeros((hr_h_orig, hr_w_orig), dtype=np.uint8)
            for p in alpha_paths_str:
                with rasterio.open(p) as band_src:
                    valid_band_count += (~np.isnan(band_src.read(1))).astype(np.uint8)
            
            required_bands = int(NUM_ALPHA_BANDS_TO_STACK * HR_BAND_REQUIREMENT_RATIO)
            hr_valid_mask_full_res = valid_band_count >= required_bands
            plt.imsave(DEBUG_OUTPUT_DIR / "2_hr_valid_mask_full_res.png", hr_valid_mask_full_res, cmap='gray')
            print(f"   - Saved to '2_hr_valid_mask_full_res.png'. Shape: {hr_valid_mask_full_res.shape}")

            # 使用严格的处理尺寸降采样HR掩码
            print("[INFO] 3. Downsampling HR mask to LR resolution...")
            hr_valid_mask_lr_res = hr_valid_mask_full_res[:proc_h*SCALE_FACTOR, :proc_w*SCALE_FACTOR].reshape(proc_h, SCALE_FACTOR, proc_w, SCALE_FACTOR).all(axis=(1,3))
            plt.imsave(DEBUG_OUTPUT_DIR / "3_hr_valid_mask_lr_res.png", hr_valid_mask_lr_res, cmap='gray')
            print(f"   - Saved to '3_hr_valid_mask_lr_res.png'. Shape: {hr_valid_mask_lr_res.shape}")

            print("[INFO] 4. Generating the final combined mask...")
            combined_mask = lr_valid_mask & hr_valid_mask_lr_res
            plt.imsave(DEBUG_OUTPUT_DIR / "4_combined_mask.png", combined_mask, cmap='gray')
            print(f"   - Saved to '4_combined_mask.png'. Shape: {combined_mask.shape}. Total valid pixels: {np.count_nonzero(combined_mask)}")
            
            print("\n✅ Debugging complete. Please check the images in the 'debug_output' folder.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR during debug: {e}")

if __name__ == '__main__':
    run_mask_debug()
