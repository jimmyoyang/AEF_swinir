# 文件路径: prepare_data_local.py
# (最终健壮版 - 修复了TARGET_DATES引用错误)

import re
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
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 1. 路径和参数定义 (保持不变)
# ==============================================================================
BASE_DATA_DIR = Path("./data")
RAW_LANDSAT_DIR = BASE_DATA_DIR / "raw_landsat"
RAW_ALPHA_DIR = BASE_DATA_DIR / "raw_alphaearth"
PROCESSED_DATA_ROOT = BASE_DATA_DIR / "processed_data"

# TARGET_DATES 列表不再需要，已被注释
# TARGET_DATES = ['2018-01-01', '2018-01-17', '2018-02-02']

REF_PATH, REF_ROW = 132, 33
NUM_ALPHA_BANDS_TO_STACK = 64
ALPHAEARTH_BANDS_TO_USE = [f'A{i:02d}' for i in range(NUM_ALPHA_BANDS_TO_STACK)]
LR_PATCH_SIZE, SCALE_FACTOR, HR_PATCH_SIZE = 64, 3, 192
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
RANDOM_SEED = 42
HR_BAND_REQUIREMENT_RATIO = 1.0
NUM_WORKERS = 48

# ==============================================================================
# 2. 交互式调试与可视化函数 (已修复)
# ==============================================================================
def debug_and_visualize():
    """
    在批量处理前，对一个固定位置的瓦片进行内存中的预处理，
    并进行详细的可视化，最后等待用户确认。
    """
    print("\n===== STEP 1: Entering Interactive Debug & Visualization Mode =====")
    DEBUG_TILE_ROW_INDEX, DEBUG_TILE_COL_INDEX = 30, 30
    
    try:
        # --- 【代码修改开始】 ---
        # 动态发现第一个可用日期用于调试，不再依赖 TARGET_DATES
        all_landsat_files = sorted(RAW_LANDSAT_DIR.glob(f'L8_{REF_PATH}{REF_ROW}_*_Masked.tif'))
        if not all_landsat_files:
            print(f"❌ CRITICAL ERROR in debug: 在 {RAW_LANDSAT_DIR} 目录下未找到任何Landsat文件。", file=sys.stderr)
            return False
        
        first_landsat_file = all_landsat_files[0]
        match = re.search(r'_(\d{8})_', first_landsat_file.name)
        if not match:
            print(f"❌ CRITICAL ERROR in debug: 无法从文件名 {first_landsat_file.name} 中提取日期", file=sys.stderr)
            return False
            
        test_date_str = match.group(1)
        test_date_for_print = f"{test_date_str[:4]}-{test_date_str[4:6]}-{test_date_str[6:]}"
        print(f"[INFO] Using first discovered date for debug: {test_date_for_print}")
        # --- 【代码修改结束】 ---
        
        alpha_paths = sorted([list(RAW_ALPHA_DIR.glob(f"AlphaEarth_Path{REF_PATH}_Row{REF_ROW}_reprojected_{b}.tif"))[0] for b in ALPHAEARTH_BANDS_TO_USE])
        
        print(f"[INFO] Found LR image: {first_landsat_file.name}")
        
        with rasterio.open(first_landsat_file) as lr_src, rasterio.open(alpha_paths[0]) as first_hr_src:
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
        
        # ... (可视化的其余部分保持不变) ...
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
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# 3. 并行处理的 "工人" 函数 (保持不变)
# ==============================================================================
def create_one_tile(landsat_path, alpha_band_paths, r, c, save_dir, date_str):
    try:
        with rasterio.open(landsat_path) as lr_src:
            lr_meta = lr_src.meta
            lr_transform = lr_src.transform
            lr_tile = lr_src.read(window=Window(c, r, LR_PATCH_SIZE, LR_PATCH_SIZE))
        with rasterio.open(alpha_band_paths[0]) as first_hr_src:
            hr_meta = first_hr_src.meta
            hr_transform = first_hr_src.transform
        if np.isnan(lr_tile).any() or np.all(lr_tile == 0):
            return False
        hr_tile = np.zeros((len(alpha_band_paths), HR_PATCH_SIZE, HR_PATCH_SIZE), dtype=hr_meta['dtype'])
        hr_r, hr_c = r * SCALE_FACTOR, c * SCALE_FACTOR
        
        for i, band_path in enumerate(alpha_band_paths):
            with rasterio.open(band_path) as band_src:
                hr_tile[i, :, :] = band_src.read(1, window=Window(hr_c, hr_r, HR_PATCH_SIZE, HR_PATCH_SIZE))
        if np.isnan(hr_tile).any():
            return False
        fname = f"{date_str}_tile_{r//LR_PATCH_SIZE}_{c//LR_PATCH_SIZE}.tif"
        lr_save_path = save_dir / "LR" / fname
        hr_save_path = save_dir / "HR" / fname
        lr_tile_meta = lr_meta.copy()
        lr_tile_meta.update({
            'height': LR_PATCH_SIZE, 'width': LR_PATCH_SIZE,
            'transform': rasterio.windows.transform(Window(c, r, LR_PATCH_SIZE, LR_PATCH_SIZE), lr_transform)
        })
        with rasterio.open(lr_save_path, 'w', **lr_tile_meta) as dst:
            dst.write(lr_tile)
        hr_tile_meta = hr_meta.copy()
        hr_tile_meta.update({
            'count': len(alpha_band_paths), 'height': HR_PATCH_SIZE, 'width': HR_PATCH_SIZE,
            'transform': rasterio.windows.transform(Window(hr_c, hr_r, HR_PATCH_SIZE, HR_PATCH_SIZE), hr_transform)
        })
        with rasterio.open(hr_save_path, 'w', **hr_tile_meta) as dst:
            dst.write(hr_tile)
        return True
    except Exception:
        return False

# ==============================================================================
# 4. 主处理与划分函数 (已修复)
# ==============================================================================
def main_processing():
    temp_dir = PROCESSED_DATA_ROOT / "all_tiles_temp"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    (temp_dir / "LR").mkdir(parents=True); (temp_dir / "HR").mkdir(parents=True)
    print(f"[INFO] Created temporary directory: {temp_dir}")

    print("\n===== STEP 2: Generating Task List for Parallel Processing =====")
    tasks = []
    
    # --- 【代码修改部分 1】 ---
    # 从文件名动态发现所有Landsat日期
    all_landsat_files = sorted(RAW_LANDSAT_DIR.glob(f'L8_{REF_PATH}{REF_ROW}_*_Masked.tif'))
    if not all_landsat_files:
        print(f"❌ CRITICAL ERROR: 在 {RAW_LANDSAT_DIR} 目录下未找到 P{REF_PATH}R{REF_ROW} 的Landsat文件。", file=sys.stderr)
        return
        
    target_dates_from_files = []
    for f in all_landsat_files:
        # 从文件名 'L8_132033_20180101_Masked.tif' 中提取 '20180101'
        match = re.search(r'_(\d{8})_', f.name)
        if match:
            target_dates_from_files.append(match.group(1))
    
    print(f"[INFO] 从Landsat文件名中发现 {len(target_dates_from_files)} 个唯一日期进行处理。")
    # --- 【代码修改部分 1 结束】 ---
    
    # --- 【代码修改部分 2】 ---
    # 使用我们动态发现的日期列表(target_dates_from_files)，而不是固定的TARGET_DATES
    for date_str in tqdm(target_dates_from_files, desc="Scanning Dates"):
    # --- 【代码修改部分 2 结束】 ---
        try:
            landsat_f = list(RAW_LANDSAT_DIR.glob(f'L8_{REF_PATH}{REF_ROW}_{date_str}_Masked.tif'))
            if not landsat_f: print(f"⚠️ Warning: Landsat for {date_str} not found. Skipping."); continue
            
            landsat_path = landsat_f[0]
            alpha_paths_str = [str(p[0]) for p in [list(RAW_ALPHA_DIR.glob(f"AlphaEarth_Path{REF_PATH}_Row{REF_ROW}_reprojected_{b}.tif")) for b in ALPHAEARTH_BANDS_TO_USE]]
            if any(not p for p in alpha_paths_str): print(f"⚠️ Warning: Some AlphaEarth bands for {date_str} not found. Skipping."); continue

            with rasterio.open(landsat_path) as lr_src, rasterio.open(alpha_paths_str[0]) as first_hr_src:
                # 【修复】这里是核心：首先确定最终的处理尺寸
                lr_h_orig, lr_w_orig = lr_src.height, lr_src.width
                hr_h_orig, hr_w_orig = first_hr_src.height, first_hr_src.width
                
                # 以 HR 影像为基准，反算出 LR 影像应该有的最大尺寸
                effective_lr_h = hr_h_orig // SCALE_FACTOR
                effective_lr_w = hr_w_orig // SCALE_FACTOR
                # 取 LR 原始尺寸和反算尺寸中较小的一个，作为最终处理尺寸
                proc_h = min(lr_h_orig, effective_lr_h)
                proc_w = min(lr_w_orig, effective_lr_w)
                
                # 【修复】读取 LR 数据时，严格使用确定好的处理尺寸
                lr_data = lr_src.read(window=Window(0, 0, proc_w, proc_h))
                lr_valid_mask = ~np.any(np.isnan(lr_data), axis=0) & np.all(lr_data != 0, axis=0)
                
                valid_band_count = np.zeros((hr_h_orig, hr_w_orig), dtype=np.uint8)
                for p in alpha_paths_str:
                    with rasterio.open(p) as band_src:
                        valid_band_count += (~np.isnan(band_src.read(1))).astype(np.uint8)
                
                required_bands = int(NUM_ALPHA_BANDS_TO_STACK * HR_BAND_REQUIREMENT_RATIO)
                hr_valid_mask_full_res = valid_band_count >= required_bands
                # 【修复】对 HR 掩码也使用严格的处理尺寸进行切片和降采样
                hr_valid_mask_lr_res = hr_valid_mask_full_res[:proc_h*SCALE_FACTOR, :proc_w*SCALE_FACTOR].reshape(proc_h, SCALE_FACTOR, proc_w, SCALE_FACTOR).all(axis=(1,3))
                combined_mask = lr_valid_mask & hr_valid_mask_lr_res
            
            for r in range(0, proc_h - LR_PATCH_SIZE + 1, LR_PATCH_SIZE):
                for c in range(0, proc_w - LR_PATCH_SIZE + 1, LR_PATCH_SIZE):
                    if np.count_nonzero(combined_mask[r:r+LR_PATCH_SIZE, c:c+LR_PATCH_SIZE]) / (LR_PATCH_SIZE**2) >= 0.98:
                        task_args = (str(landsat_path), alpha_paths_str, r, c, temp_dir, date_str)
                        tasks.append(task_args)
        except Exception as e:
            print(f"❌ CRITICAL ERROR during task generation for date {date_str}: {e}", file=sys.stderr)
    
    if not tasks: print("❌ CRITICAL ERROR: No valid tiles found to process. Try lowering HR_BAND_REQUIREMENT_RATIO."); return
    print(f"\n✅ Task generation complete. Found {len(tasks)} potential tiles to create.")

    print("\n" + "="*80 + f"\n===== STEP 3: Starting Parallel Tiling with {NUM_WORKERS} Cores =====")
    
    results = []
    effective_workers = min(NUM_WORKERS, os.cpu_count()) if NUM_WORKERS > 0 else 1
    if effective_workers > 1:
        with Pool(processes=effective_workers) as pool:
            results = list(tqdm(pool.starmap(create_one_tile, tasks), total=len(tasks), desc="Processing Tiles"))
    else:
        results = [create_one_tile(*task) for task in tqdm(tasks, desc="Processing Tiles (Single-threaded)")]

    successful_tiles = sum(1 for r in results if r)
    print(f"✅ Parallel processing complete. Successfully created {successful_tiles} tiles. {len(tasks) - successful_tiles} tiles failed.")

    print("\n" + "="*80 + "\n===== STEP 4: Randomly Shuffling and Splitting Dataset =====")
    
    all_lr_files = sorted(list((temp_dir / "LR").glob("*.tif")))
    if not all_lr_files: print("❌ CRITICAL ERROR: No tiles were actually generated after processing."); return
    
    print(f"[INFO] Total valid tiles to be split: {len(all_lr_files)}")
    random.seed(RANDOM_SEED); random.shuffle(all_lr_files)
    
    num_files = len(all_lr_files)
    train_end_idx = int(num_files * TRAIN_RATIO)
    val_end_idx = train_end_idx + int(num_files * VAL_RATIO)
    
    train_files = all_lr_files[:train_end_idx]; val_files = all_lr_files[train_end_idx:val_end_idx]; test_files = all_lr_files[val_end_idx:]
    print(f"[INFO] Splitting dataset into:\n  - {len(train_files)} for training\n  - {len(val_files)} for validation\n  - {len(test_files)} for testing")
    
    train_dir = PROCESSED_DATA_ROOT / "train"; val_dir = PROCESSED_DATA_ROOT / "val"; test_dir = PROCESSED_DATA_ROOT / "test"
    
    for d in [train_dir, val_dir, test_dir]:
        if d.exists(): shutil.rmtree(d)
        (d / "LR").mkdir(parents=True); (d / "HR").mkdir(parents=True)
        
    print("\n[INFO] Moving files from temporary directory to final train/val/test directories...")
    datasets_to_move = {"train": (train_files, train_dir), "val": (val_files, val_dir), "test": (test_files, test_dir)}
    
    for name, (file_list, dest_dir) in datasets_to_move.items():
        for lr_file_path in tqdm(file_list, desc=f"Moving {name} files", ncols=100):
            hr_file_path = temp_dir / "HR" / lr_file_path.name
            dest_lr_path = dest_dir / "LR" / lr_file_path.name
            dest_hr_path = dest_dir / "HR" / lr_file_path.name
            shutil.move(str(lr_file_path), str(dest_lr_path))
            shutil.move(str(hr_file_path), str(dest_hr_path))

    print("\n[INFO] Cleaning up temporary directory..."); shutil.rmtree(temp_dir)
    print("\n✅ All data preprocessing and splitting (train/val/test) is complete.")

# ==============================================================================
# 5. 主执行函数 (修改以决定是否进入调试模式)
# ==============================================================================
def run_interactive_mode():
    """
    根据用户输入决定是否先进入调试模式。
    """
    # 默认不再强制进入调试模式，直接处理
    # 如果需要调试，可以取消下面三行注释
    if not debug_and_visualize():
        print("🛑 Processing aborted by user.")
        return
    main_processing()

if __name__ == '__main__':
    run_interactive_mode()

