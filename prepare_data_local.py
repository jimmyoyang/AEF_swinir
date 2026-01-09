# 文件路径: prepare_data_local.py
# (最终健壮版 v2 - 增加独立的 Test 集划分)

import os
import rasterio
import numpy as np
import glob
from tqdm import tqdm
from rasterio.windows import Window
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import random
import shutil

# ==============================================================================
# 1. 路径和参数定义 (已修改)
# ==============================================================================
BASE_DATA_DIR = Path("./data")
RAW_LANDSAT_DIR = BASE_DATA_DIR / "raw_landsat"
RAW_ALPHA_DIR = BASE_DATA_DIR / "raw_alphaearth"
PROCESSED_DATA_ROOT = BASE_DATA_DIR / "processed_data"

TARGET_DATES = ['2018-01-01', '2018-01-17', '2018-02-02']
REF_PATH, REF_ROW = 132, 33
NUM_ALPHA_BANDS_TO_STACK = 3
ALPHAEARTH_BANDS_TO_USE = [f'A{i:02d}' for i in range(NUM_ALPHA_BANDS_TO_STACK)]
LR_PATCH_SIZE, SCALE_FACTOR, HR_PATCH_SIZE = 64, 3, 192

# --- 【核心修改】定义三路划分比例 ---
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.15   # 15% for validation
# TEST_RATIO is implicitly 15% (1.0 - 0.7 - 0.15)

# ==============================================================================
# 2. 交互式调试与可视化函数 (保持不变)
# ==============================================================================
def debug_and_visualize():
    """
    在批量处理前，对一个固定位置的瓦片进行内存中的预处理，
    并进行详细的可视化，最后等待用户确认。
    """
    print("\n===== STEP 1: Entering Interactive Debug & Visualization Mode =====")
    DEBUG_TILE_ROW_INDEX, DEBUG_TILE_COL_INDEX = 30, 30
    
    try:
        test_date_str = TARGET_DATES[0].replace('-', '')
        print(f"[INFO] Using test date: {TARGET_DATES[0]}")
        
        landsat_file = list(RAW_LANDSAT_DIR.glob(f'L8_{REF_PATH}{REF_ROW}_{test_date_str}_Masked.tif'))[0]
        alpha_paths = sorted([list(RAW_ALPHA_DIR.glob(f"AlphaEarth_Path{REF_PATH}_Row{REF_ROW}_reprojected_{b}.tif"))[0] for b in ALPHAEARTH_BANDS_TO_USE])
        
        print(f"[INFO] Found LR image: {landsat_file.name}")
        
        with rasterio.open(landsat_file) as lr_src, rasterio.open(alpha_paths[0]) as first_hr_src:
            lr_meta, (lr_h, lr_w) = lr_src.meta, (lr_src.height, lr_src.width)
            hr_meta, (hr_h, hr_w) = first_hr_src.meta, (first_hr_src.height, first_hr_src.width)
            
            r_start, c_start = DEBUG_TILE_ROW_INDEX * LR_PATCH_SIZE, DEBUG_TILE_COL_INDEX * LR_PATCH_SIZE
            if r_start >= lr_h - LR_PATCH_SIZE or c_start >= lr_w - LR_PATCH_SIZE:
                raise ValueError(f"Debug tile index ({DEBUG_TILE_ROW_INDEX}, {DEBUG_TILE_COL_INDEX}) is out of bounds.")
            
            print(f"\n[DEBUG] Cropping from fixed tile index: (Row: {DEBUG_TILE_ROW_INDEX}, Col: {DEBUG_TILE_COL_INDEX})")
            
            stacked_hr_data = np.zeros((len(alpha_paths), hr_h, hr_w), dtype=hr_meta['dtype'])
            for i, band_path in enumerate(alpha_paths):
                with rasterio.open(band_path) as band_src: stacked_hr_data[i,:,:] = band_src.read(1)
            
            lr_win = Window(c_start, r_start, LR_PATCH_SIZE, LR_PATCH_SIZE)
            hr_win = Window(c_start * SCALE_FACTOR, r_start * SCALE_FACTOR, HR_PATCH_SIZE, HR_PATCH_SIZE)
            lr_tile = lr_src.read(window=lr_win)
            hr_tile = stacked_hr_data[:, hr_win.row_off:hr_win.row_off+hr_win.height, hr_win.col_off:hr_win.col_off+hr_win.width]
            
            print(f"[DEBUG] Cropped LR tile shape: {lr_tile.shape}, Cropped HR tile shape: {hr_tile.shape}")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"Debug Visualization for Tile (R:{DEBUG_TILE_ROW_INDEX}, C:{DEBUG_TILE_COL_INDEX})", fontsize=16)
        def normalize(band): return (band - np.nanmin(band)) / (np.nanmax(band) - np.nanmin(band)) if (np.nanmax(band) - np.nanmin(band)) > 0 else band
        
        axes[0,0].imshow(np.stack([normalize(lr_tile[2]),normalize(lr_tile[1]),normalize(lr_tile[0])],axis=-1)); axes[0,0].set_title(f'LR - RGB Composite')
        axes[0,1].imshow(normalize(lr_tile[3]), cmap='viridis'); axes[0,1].set_title(f'LR - NIR Band')
        axes[0,2].imshow(normalize(lr_tile[7]), cmap='magma'); axes[0,2].set_title(f'LR - SWIR2 Band')
        
        if NUM_ALPHA_BANDS_TO_STACK >= 3:
            axes[1,0].imshow(np.stack([normalize(hr_tile[0]),normalize(hr_tile[1]),normalize(hr_tile[2])],axis=-1)); axes[1,0].set_title(f'HR - RGB Composite')
            axes[1,1].imshow(normalize(hr_tile[1]), cmap='viridis'); axes[1,1].set_title(f'HR - Band A01')
            axes[1,2].imshow(normalize(hr_tile[2]), cmap='magma'); axes[1,2].set_title(f'HR - Band A02')
        else:
            axes[1,0].imshow(normalize(hr_tile[0]), cmap='gray'); axes[1,0].set_title('HR - Band A00'); axes[1,1].axis('off'); axes[1,2].axis('off')
        
        for ax in axes.flatten(): ax.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        PROCESSED_DATA_ROOT.mkdir(exist_ok=True)
        save_path = PROCESSED_DATA_ROOT / "debug_visualization.png"
        plt.savefig(save_path, dpi=150)
        print(f"\n[INFO] Debug visualization saved to: {save_path}")
        plt.show()
        
        print("\n" + "="*60)
        proceed = input("[ACTION] Visualization complete. Proceed with batch processing? (y/n): ")
        print("="*60 + "\n")
        return proceed.lower() == 'y'
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR during debug phase: {e}", file=sys.stderr)
        return False

# ==============================================================================
# 3. 核心批量处理函数 (保持不变)
# ==============================================================================
def process_and_tile_pair(landsat_path, alpha_band_paths, save_dir, date_str):
    """
    【最终健壮版】对单对 LR/HR 数据进行堆叠、掩码生成和切片。
    """
    print(f"\n--- Processing Date: {date_str} ---")
    try:
        with rasterio.open(landsat_path) as lr_src:
            lr_meta, (lr_h, lr_w) = lr_src.meta, (lr_src.height, lr_src.width)
            
            with rasterio.open(alpha_band_paths[0]) as first_hr_src:
                hr_meta, (hr_h, hr_w) = first_hr_src.meta, (first_hr_src.height, first_hr_src.width)
            # --- 步骤 1: 加载所有数据到内存 ---
            print("  - Loading all source data into memory...")
            lr_data_full = lr_src.read()
            stacked_hr_data = np.zeros((len(alpha_band_paths), hr_h, hr_w), dtype=hr_meta['dtype'])
            for i, band_path in enumerate(tqdm(alpha_band_paths, desc="    - Stacking HR bands", ncols=100)):
                with rasterio.open(band_path) as band_src: stacked_hr_data[i, :, :] = band_src.read(1)

            if hr_h < lr_h * SCALE_FACTOR or hr_w < lr_w * SCALE_FACTOR:
                print("  - [WARNING] HR image is smaller than expected. Adjusting LR processing area to match.")
                lr_h = hr_h // SCALE_FACTOR
                lr_w = hr_w // SCALE_FACTOR
                lr_data_full = lr_data_full[:, :lr_h, :lr_w]
            target_hr_h = lr_h * SCALE_FACTOR
            target_hr_w = lr_w * SCALE_FACTOR
            
            stacked_hr_data = stacked_hr_data[:, :target_hr_h, :target_hr_w]
            
            hr_h, hr_w = target_hr_h, target_hr_w
            
            print("  - Creating combined valid data mask...")
            lr_valid_mask = np.all(lr_data_full != 0, axis=0) & ~np.any(np.isnan(lr_data_full), axis=0)
            hr_valid_mask_full_res = ~np.any(np.isnan(stacked_hr_data), axis=0)
            hr_valid_mask_lr_res = hr_valid_mask_full_res.reshape(lr_h, SCALE_FACTOR, lr_w, SCALE_FACTOR).mean(axis=(1,3)) > 0.99
            combined_mask = lr_valid_mask & hr_valid_mask_lr_res
            
            tile_count, discarded_count = 0, 0
            print("  - Tiling within combined valid data area...")
            for r in tqdm(range(0, lr_h - LR_PATCH_SIZE + 1, LR_PATCH_SIZE), desc="    - Tiling", ncols=100):
                for c in range(0, lr_w - LR_PATCH_SIZE + 1, LR_PATCH_SIZE):
                    mask_patch = combined_mask[r:r+LR_PATCH_SIZE, c:c+LR_PATCH_SIZE]
                    if np.count_nonzero(mask_patch) / mask_patch.size < 0.98:
                        discarded_count += 1
                        continue
                        
                    lr_tile = lr_data_full[:, r:r+LR_PATCH_SIZE, c:c+LR_PATCH_SIZE]
                    hr_r, hr_c = r * SCALE_FACTOR, c * SCALE_FACTOR
                    hr_tile = stacked_hr_data[:, hr_r:hr_r+HR_PATCH_SIZE, hr_c:hr_c+HR_PATCH_SIZE]
                    
                    if np.isnan(lr_tile).any() or np.isnan(hr_tile).any():
                        discarded_count += 1; continue

                    fname = f"{date_str}_tile_{r//LR_PATCH_SIZE}_{c//LR_PATCH_SIZE}.tif"
                    lr_tile_meta=lr_meta.copy(); lr_tile_meta.update({'height':LR_PATCH_SIZE,'width':LR_PATCH_SIZE,'transform':rasterio.windows.transform(Window(c,r,LR_PATCH_SIZE,LR_PATCH_SIZE),lr_src.transform)})
                    with rasterio.open(save_dir/"LR"/fname,'w',**lr_tile_meta) as dst: dst.write(lr_tile)
                    hr_tile_meta=hr_meta.copy(); hr_tile_meta.update({'count':len(alpha_band_paths),'height':HR_PATCH_SIZE,'width':HR_PATCH_SIZE,'transform':rasterio.windows.transform(Window(hr_c,hr_r,HR_PATCH_SIZE,HR_PATCH_SIZE),first_hr_src.transform)})
                    with rasterio.open(save_dir/"HR"/fname,'w',**hr_tile_meta) as dst: dst.write(hr_tile)
                    tile_count += 1
            
            print(f"✅ Processing complete: {tile_count} strictly paired tiles saved. {discarded_count} tiles were discarded.")
    except Exception as e: print(f"❌ CRITICAL ERROR in 'process_and_tile_pair': {e}", file=sys.stderr)

# ==============================================================================
# 4. 主处理与划分函数 (已修改)
# ==============================================================================
def main_processing():
    """
    采用“先混合，再划分”的科学策略。
    """
    print("===== STEP 2: Starting Batch Data Processing (Scientific Split) =====")
    
    temp_dir = PROCESSED_DATA_ROOT / "all_tiles_temp"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    print(f"[INFO] Creating temporary directory: {temp_dir}"); (temp_dir / "LR").mkdir(parents=True); (temp_dir / "HR").mkdir(parents=True)
    
    for date_str in [d.replace('-', '') for d in TARGET_DATES]:
        landsat_f = list(RAW_LANDSAT_DIR.glob(f'L8_{REF_PATH}{REF_ROW}_{date_str}_Masked.tif'))
        if not landsat_f: print(f"⚠️ Warning: Landsat for {date_str} not found. Skipping."); continue
        alpha_paths = [list(RAW_ALPHA_DIR.glob(f"AlphaEarth_Path{REF_PATH}_Row{REF_ROW}_reprojected_{b}.tif")) for b in ALPHAEARTH_BANDS_TO_USE]
        if any(not p for p in alpha_paths): print(f"⚠️ Warning: AlphaEarth for {date_str} not found. Skipping."); continue
        process_and_tile_pair(landsat_f[0], [p[0] for p in alpha_paths], temp_dir, date_str)

    print("\n" + "="*80 + "\n===== STEP 3: Randomly Shuffling and Splitting Dataset =====")
    
    all_lr_files = sorted(list((temp_dir / "LR").glob("*.tif")))
    if not all_lr_files: 
        print("❌ CRITICAL ERROR: No tiles were generated."); return
        
    print(f"[INFO] Total valid tiles generated: {len(all_lr_files)}"); random.shuffle(all_lr_files)
    
    # --- 【核心修改】计算三路划分的切分点 ---
    num_files = len(all_lr_files)
    train_end_idx = int(num_files * TRAIN_RATIO)
    val_end_idx = train_end_idx + int(num_files * VAL_RATIO)
    
    train_files = all_lr_files[:train_end_idx]
    val_files = all_lr_files[train_end_idx:val_end_idx]
    test_files = all_lr_files[val_end_idx:] # The rest are for testing

    print(f"[INFO] Splitting dataset into:")
    print(f"  - {len(train_files)} files for training ({TRAIN_RATIO:.0%})")
    print(f"  - {len(val_files)} files for validation ({VAL_RATIO:.0%})")
    print(f"  - {len(test_files)} files for testing (~{(1 - TRAIN_RATIO - VAL_RATIO):.0%})")
    
    # --- 【核心修改】创建所有目标目录 ---
    train_dir = PROCESSED_DATA_ROOT / "train"
    val_dir = PROCESSED_DATA_ROOT / "val"
    test_dir = PROCESSED_DATA_ROOT / "test"  # 新增
    
    for d in [train_dir, val_dir, test_dir]:
        if d.exists(): shutil.rmtree(d)
        (d / "LR").mkdir(parents=True)
        (d / "HR").mkdir(parents=True)
        
    print("\n[INFO] Moving files to final train/val/test directories...")
    
    # --- 【核心修改】使用一个通用循环移动所有文件 ---
    datasets_to_move = {
        "train": (train_files, train_dir),
        "val": (val_files, val_dir),
        "test": (test_files, test_dir)
    }

    for name, (file_list, dest_dir) in datasets_to_move.items():
        for file_path in tqdm(file_list, desc=f"Moving {name} files", ncols=100):
            # 确保使用 Path 对象进行操作
            src_lr_path = Path(file_path)
            src_hr_path = temp_dir / "HR" / src_lr_path.name
            
            dest_lr_path = dest_dir / "LR" / src_lr_path.name
            dest_hr_path = dest_dir / "HR" / src_lr_path.name

            # 使用 shutil.move，它对于 Path 对象同样有效
            shutil.move(str(src_lr_path), str(dest_lr_path))
            shutil.move(str(src_hr_path), str(dest_hr_path))
            
    print("\n[INFO] Cleaning up temporary directory..."); shutil.rmtree(temp_dir)
    print("\n✅ All data preprocessing and splitting (train/val/test) is complete.")

# ==============================================================================
# 5. 脚本执行入口 (保持不变)
# ==============================================================================
def run_interactive_mode():
    if debug_and_visualize():
        main_processing()
    else:
        print("[INFO] User aborted. No batch processing will be performed.")

if __name__ == '__main__':
    run_interactive_mode()

