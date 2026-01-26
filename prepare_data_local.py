# ==============================================================================
# 文件路径: prepare_data_local.py
# (最终完全版 - 高性能并行)
#
# 功能:
#   - 从原始数据目录 (raw_landsat, raw_alphaearth) 读取数据。
#   - 生成内存高效的、基于有效像素掩码的瓦片。
#   - 将每个时相的瓦片保存为独立文件 (e.g., 20180101_tile_0_0.tif)。
#   - 使用固定的随机种子，将所有瓦片物理分割到 train/val/test 文件夹。
# ==============================================================================
import os
import re
import glob
import random
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# --- 1. 核心参数配置 (请根据您的环境进行调整) ---

# 路径配置
SCRIPT_DIR = Path(__file__).resolve().parent

# BASE_DATA_DIR = SCRIPT_DIR/ "data"  #Path("./data") 
ABSOLUTE_PATH="/mnt/lm_data_afs/wangzining/charles/AEF_swinir/data"
BASE_DATA_DIR=Path(ABSOLUTE_PATH)
RAW_LANDSAT_DIR = BASE_DATA_DIR / "raw_landsat"
RAW_ALPHA_DIR = BASE_DATA_DIR / "raw_alphaearth"
PROCESSED_DATA_ROOT = BASE_DATA_DIR / "processed_data"

# 数据元信息
REF_PATH, REF_ROW = 132, 33
NUM_ALPHA_BANDS_TO_STACK = 64
ALPHAEARTH_BANDS_TO_USE = [f'A{i:02d}' for i in range(NUM_ALPHA_BANDS_TO_STACK)]
LR_PATCH_SIZE, SCALE_FACTOR, HR_PATCH_SIZE = 64, 3, 192

# 数据集划分与并行设置
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
RANDOM_SEED = 42
# 有效像素比例阈值：一个瓦片中有效像素必须达到这个比例才会被保留
VALID_PIXEL_RATIO_THRESHOLD = 0.98 
HR_BAND_REQUIREMENT_RATIO = 1.0 
NUM_WORKERS = 64 # 并行处理的CPU核心数

# ==============================================================================
# 2. "工人" 函数 (用于并行处理，无需修改)
# ==============================================================================

def create_one_tile(args):
    """
    内存高效的函数：只创建一对 LR/HR 瓦片。
    只读取所需的小窗口，绝不加载整个文件。
    接收一个元组作为参数以适配 Pool.starmap。
    """
    landsat_path, alpha_band_paths, r, c = args
    try:
        date_str = re.search(r'_(\d{8})_', Path(landsat_path).name).group(1)
        
        with rasterio.open(landsat_path) as lr_src:
            lr_meta = lr_src.meta
            lr_win = Window(c, r, LR_PATCH_SIZE, LR_PATCH_SIZE)
            lr_tile = lr_src.read(window=lr_win)
            # 检查LR瓦片是否有效
            if np.all(lr_tile == 0) or np.isnan(lr_tile).any(): 
                return None

        with rasterio.open(alpha_band_paths[0]) as first_hr_src:
            hr_meta = first_hr_src.meta

        hr_tile = np.zeros((NUM_ALPHA_BANDS_TO_STACK, HR_PATCH_SIZE, HR_PATCH_SIZE), dtype=hr_meta['dtype'])
        hr_r, hr_c = r * SCALE_FACTOR, c * SCALE_FACTOR
        hr_win = Window(hr_c, hr_r, HR_PATCH_SIZE, HR_PATCH_SIZE)

        for i, band_path in enumerate(alpha_band_paths):
            with rasterio.open(band_path) as band_src:
                hr_tile[i, :, :] = band_src.read(1, window=hr_win)
        
        # 检查HR瓦片是否有效
        if np.isnan(hr_tile).any(): 
            return None

        # 瓦片命名规则：日期_tile_行号_列号.tif
        fname = f"{date_str}_tile_{r//LR_PATCH_SIZE}_{c//LR_PATCH_SIZE}.tif"
        
        # 更新瓦片的元数据
        lr_tile_meta = lr_meta.copy()
        lr_tile_meta.update({
            'height': LR_PATCH_SIZE, 'width': LR_PATCH_SIZE,
            'transform': rasterio.windows.transform(lr_win, lr_meta['transform'])
        })

        hr_tile_meta = hr_meta.copy()
        hr_tile_meta.update({
            'count': NUM_ALPHA_BANDS_TO_STACK, 'height': HR_PATCH_SIZE, 'width': HR_PATCH_SIZE,
            'transform': rasterio.windows.transform(hr_win, hr_meta['transform'])
        })
        
        return (fname, lr_tile, lr_tile_meta, hr_tile, hr_tile_meta)

    except Exception as e:
        # 在并行任务中打印错误，但主进程会继续
        print(f"Warning: Error processing tile at (r={r}, c={c}) for {landsat_path}: {e}", file=sys.stderr)
        return None

# ==============================================================================
# 3. 主处理函数 (新的并行调度核心)
# ==============================================================================

def main_processing():
    """
    采用两阶段并行处理，并融合您脚本中的所有检查逻辑。
    """
    # 清理旧数据
    if PROCESSED_DATA_ROOT.exists():
        print(f"[INFO] Old processed data found. Removing '{PROCESSED_DATA_ROOT}' for a clean run.")
        shutil.rmtree(PROCESSED_DATA_ROOT)

    # 创建临时目录用于存放所有生成的瓦片
    temp_dir = PROCESSED_DATA_ROOT / "all_tiles_temp"
    (temp_dir / "LR").mkdir(parents=True)
    (temp_dir / "HR").mkdir(parents=True)
    print(f"[INFO] Created temporary directory: {temp_dir}")

    print("\n===== STAGE 1: Planning all tiling tasks (this is fast) =====")
    tasks = []
    
    all_landsat_files = sorted(RAW_LANDSAT_DIR.glob(f'L8_{REF_PATH}{REF_ROW}_*_Masked.tif'))
    if not all_landsat_files:
        print(f"❌ CRITICAL: No Landsat files found in {RAW_LANDSAT_DIR}. Please check the path.", file=sys.stderr)
        return

    print(f"[INFO] Found {len(all_landsat_files)} time steps to process.")
    
    # 遍历所有时相，生成有效瓦片的坐标
    for landsat_path in tqdm(all_landsat_files, desc="Scanning dates for task planning"):
        date_str = re.search(r'_(\d{8})_', landsat_path.name).group(1)
        
        # 检查对应的HR波段是否存在
        alpha_paths_str = [str(p) for p in [next(iter(RAW_ALPHA_DIR.glob(f"AlphaEarth_Path{REF_PATH}_Row{REF_ROW}_reprojected_{b}.tif")), None) for b in ALPHAEARTH_BANDS_TO_USE]]
        if any(p == 'None' for p in alpha_paths_str):
            print(f"⚠️ Warning: Incomplete HR bands for date {date_str}. Skipping this date.", file=sys.stderr)
            continue
            
        # 生成有效像素掩码
        with rasterio.open(landsat_path) as lr_src, rasterio.open(alpha_paths_str[0]) as first_hr_src:
            lr_h_orig, lr_w_orig = lr_src.height, lr_src.width
            hr_h_orig, hr_w_orig = first_hr_src.height, first_hr_src.width
            
            proc_h = min(lr_h_orig, hr_h_orig // SCALE_FACTOR)
            proc_w = min(lr_w_orig, hr_w_orig // SCALE_FACTOR)
            
            lr_data = lr_src.read(window=Window(0, 0, proc_w, proc_h))
            lr_valid_mask = ~np.any(np.isnan(lr_data), axis=0) & np.all(lr_data != 0, axis=0)
            
            valid_band_count = np.zeros((hr_h_orig, hr_w_orig), dtype=np.uint8)
            for p in alpha_paths_str:
                with rasterio.open(p) as band_src:
                    valid_band_count += (~np.isnan(band_src.read(1))).astype(np.uint8)
            
            required_bands = int(NUM_ALPHA_BANDS_TO_STACK * HR_BAND_REQUIREMENT_RATIO)
            hr_valid_mask_full_res = valid_band_count >= required_bands
            hr_valid_mask_lr_res = hr_valid_mask_full_res[:proc_h*SCALE_FACTOR, :proc_w*SCALE_FACTOR].reshape(proc_h, SCALE_FACTOR, proc_w, SCALE_FACTOR).all(axis=(1,3))
            
            combined_mask = lr_valid_mask & hr_valid_mask_lr_res

        # 根据掩码生成任务列表
        for r in range(0, proc_h - LR_PATCH_SIZE + 1, LR_PATCH_SIZE):
            for c in range(0, proc_w - LR_PATCH_SIZE + 1, LR_PATCH_SIZE):
                if np.count_nonzero(combined_mask[r:r+LR_PATCH_SIZE, c:c+LR_PATCH_SIZE]) / (LR_PATCH_SIZE**2) >= VALID_PIXEL_RATIO_THRESHOLD:
                    tasks.append((str(landsat_path), alpha_paths_str, r, c))
    
    if not tasks:
        print("❌ CRITICAL: No valid tiles found to process. Check data quality or thresholds.", file=sys.stderr)
        return
        
    print(f"\n✅ Task planning complete. Found {len(tasks)} potential tiles to create across all time steps.")

    effective_workers = min(NUM_WORKERS, os.cpu_count()) if NUM_WORKERS > 0 else os.cpu_count()
    print(f"\n===== STAGE 2: Starting parallel tiling with {effective_workers} CPU cores... =====")

    # 使用多进程并行创建瓦片
    successful_count = 0
    with Pool(processes=effective_workers) as pool:
        for result in tqdm(pool.imap_unordered(create_one_tile, tasks), total=len(tasks), desc="Processing Tiles"):
            if result:
                fname, lr_tile, lr_meta, hr_tile, hr_meta = result
                with rasterio.open(temp_dir / "LR" / fname, 'w', **lr_meta) as dst: dst.write(lr_tile)
                with rasterio.open(temp_dir / "HR" / fname, 'w', **hr_meta) as dst: dst.write(hr_tile)
                successful_count += 1

    print(f"\n✅ Parallel processing complete. Successfully created {successful_count} tiles.")

    print("\n===== STAGE 3: Shuffling and splitting the dataset... =====")
    # 注意：这里的划分是基于瓦片ID，而不是单个文件，以确保同一地理位置的所有时相都在同一个数据集中
    all_tile_ids = set()
    print("\n--- [DEBUG] STARTING STAGE 3 FILE SCAN ---")
    for f in (temp_dir / "LR").glob("*.tif"):
        print(f"[DEBUG] Scanning file: {f.name}")
        match = re.search(r'tile_(\d+_\d+)\.tif', f.name)
        if match:
            all_tile_ids.add(match.group(1))
            print(f"   -->SUCCESS: Found tile ID:{match.group(1)}")
    print(f"[DEBUG] Finish Scan. Total unique IDs found: {len(all_tile_ids)} ---\n")
    all_tile_ids = sorted(list(all_tile_ids))
    if not all_tile_ids:
        print("❌ CRITICAL: No tiles were actually generated after processing.", file=sys.stderr)
        return

    print(f"[INFO] Total unique tile locations to be split: {len(all_tile_ids)}")
    random.seed(RANDOM_SEED)
    random.shuffle(all_tile_ids)

    num_tiles = len(all_tile_ids)
    train_end_idx = int(num_tiles * TRAIN_RATIO)
    val_end_idx = train_end_idx + int(num_tiles * VAL_RATIO)
    
    datasets_map = {
        'train': all_tile_ids[:train_end_idx],
        'val': all_tile_ids[train_end_idx:val_end_idx],
        'test': all_tile_ids[val_end_idx:]
    }
    
    print(f"[INFO] Splitting dataset into:\n  - {len(datasets_map['train'])} locations for training\n  - {len(datasets_map['val'])} locations for validation\n  - {len(datasets_map['test'])} locations for testing")

    for name, tile_ids_split in datasets_map.items():
        if not tile_ids_split: continue
        dest_dir = PROCESSED_DATA_ROOT / name
        (dest_dir / "LR").mkdir(parents=True)
        (dest_dir / "HR").mkdir(parents=True)
        
        print(f"\n[INFO] Moving files for '{name}' set...")
        for tile_id in tqdm(tile_ids_split, desc=f"Moving {name} tiles"):
            # 移动所有与该tile_id相关的时相文件
            for lr_file_path in (temp_dir / "LR").glob(f"*_tile_{tile_id}.tif"):
                hr_file_path = temp_dir / "HR" / lr_file_path.name
                shutil.move(str(lr_file_path), str(dest_dir / "LR" / lr_file_path.name))
                if hr_file_path.exists():
                    shutil.move(str(hr_file_path), str(dest_dir / "HR" / hr_file_path.name))

    # 清理临时目录
    shutil.rmtree(temp_dir)
    print("\n✅ All data preprocessing and splitting is complete!")

# ==============================================================================
# 4. 脚本执行入口
# ==============================================================================
if __name__ == '__main__':
    main_processing()
