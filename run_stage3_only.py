# ==============================================================================
# 文件名: run_stage3_only.py
#
# 功能:
#   - 跳过耗时的 Stage 1 和 Stage 2，专门用于调试和执行 Stage 3。
#   - 直接利用已存在的 'all_tiles_temp' 文件夹中的瓦片进行数据集划分。
#
# 使用场景:
#   - 当 Stage 2 已成功运行 (生成了 all_tiles_temp)，但 Stage 3 划分失败时。
#   - 这样可以节省大量时间，无需重新生成瓦片，直接调试划分逻辑。
#
# 如何运行:
#   - 将此文件放在与 'prepare_data_local.py' 相同的目录下。
#   - 确保 './data/processed_data/all_tiles_temp' 文件夹及其中的文件存在。
#   - 在终端运行: python run_stage3_only.py
# ==============================================================================

import os
import re
import glob
import random
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

# --- 1. 核心参数配置 (从原脚本复制，确保与原脚本一致) ---
# 请确保这些配置与您运行 prepare_data_local.py 时使用的配置完全相同

# 路径配置
# 使用绝对路径或确保从正确的项目根目录运行
# SCRIPT_DIR = Path(__file__).resolve().parent
# BASE_DATA_DIR = SCRIPT_DIR/ "data"  #Path("./data") 
ABSOLUTE_PATH="/mnt/lm_data_afs/wangzining/charles/AEF_swinir/data"
BASE_DATA_DIR=Path(ABSOLUTE_PATH)
RAW_LANDSAT_DIR = BASE_DATA_DIR / "raw_landsat"
RAW_ALPHA_DIR = BASE_DATA_DIR / "raw_alphaearth"
PROCESSED_DATA_ROOT = BASE_DATA_DIR / "processed_data"
# 数据集划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
RANDOM_SEED = 42

# --- 2. 主处理函数 (只包含 Stage 3 逻辑) ---

def run_stage3_and_cleanup():
    """
    一个只执行划分、移动和清理任务的函数。
    """
    temp_dir = PROCESSED_DATA_ROOT / "all_tiles_temp"

    # --- 安全检查：确保临时文件夹存在 ---
    if not temp_dir.exists() or not (temp_dir / "LR").exists():
        print(f"❌ CRITICAL: Temporary directory '{temp_dir}/LR' not found!", file=sys.stderr)
        print("Please run the full 'prepare_data_local.py' script first to generate temporary tiles.", file=sys.stderr)
        return

    print("✅ Found existing temporary directory. Skipping Stage 1 & 2.")
    print("\n===== STARTING STAGE 3: Shuffling and splitting the dataset... =====")

    # --- 这是与原脚本完全相同的 Stage 3 逻辑，但加入了调试打印 ---
    all_tile_ids = set()

    # 加上一个开始标记
    print("\n--- [DEBUG] STARTING STAGE 3 FILE SCAN ---")
    
    # glob()会返回一个生成器，我们先把它转成列表，方便查看数量
    lr_files_to_scan = list((temp_dir / "LR").glob("*.tif"))
    
    if not lr_files_to_scan:
        print("❌ CRITICAL: The 'all_tiles_temp/LR' directory is empty. No files to process.", file=sys.stderr)
        return
        
    print(f"[INFO] Found {len(lr_files_to_scan)} files in the temporary LR directory to scan.")

    for f in tqdm(lr_files_to_scan, desc="Scanning temp files"):
        # [新增] 打印出脚本实际看到的文件名
        # 为了避免刷屏，我们可以只在循环的前几次打印
        if tqdm.format_sizeof(lr_files_to_scan, 0) < 10: # 只打印前10个
             print(f"[DEBUG] Scanning file: {f.name}")

        match = re.search(r'tile_(\d+_\d+)\.tif', f.name)
        if match:
            tile_id = match.group(1)
            all_tile_ids.add(tile_id)
            # [新增] 打印出成功匹配到的ID
            if tqdm.format_sizeof(lr_files_to_scan, 0) < 10: # 只打印前10个
                print(f"    --> SUCCESS: Found tile ID: {tile_id}")

    # 加上一个结束标记
    print(f"--- [DEBUG] FINISHED SCAN. Total unique IDs found: {len(all_tile_ids)} ---\n")

    # --- 后续逻辑与原脚本完全一样 ---
    all_tile_ids = sorted(list(all_tile_ids))
    if not all_tile_ids:
        print("❌ CRITICAL: No tile IDs were extracted after scanning all files.", file=sys.stderr)
        print("Please check the [DEBUG] output above. The file names might not match the expected pattern '...tile_ID.tif'.", file=sys.stderr)
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

    # 在移动文件前，先清理可能存在的旧的 train/val/test 文件夹
    for name in datasets_map.keys():
        if (PROCESSED_DATA_ROOT / name).exists():
            print(f"[INFO] Removing old '{name}' directory.")
            shutil.rmtree(PROCESSED_DATA_ROOT / name)


    print(f"[INFO] Splitting dataset into:\n  - {len(datasets_map['train'])} locations for training\n  - {len(datasets_map['val'])} locations for validation\n  - {len(datasets_map['test'])} locations for testing")
    for name, tile_ids_split in datasets_map.items():
        if not tile_ids_split: continue
        dest_dir = PROCESSED_DATA_ROOT / name
        (dest_dir / "LR").mkdir(parents=True)
        (dest_dir / "HR").mkdir(parents=True)

        print(f"\n[INFO] Moving files for '{name}' set...")
        for tile_id in tqdm(tile_ids_split, desc=f"Moving {name} tiles"):
            for lr_file_path in (temp_dir / "LR").glob(f"*_tile_{tile_id}.tif"):
                hr_file_path = temp_dir / "HR" / lr_file_path.name
                shutil.move(str(lr_file_path), str(dest_dir / "LR" / lr_file_path.name))
                if hr_file_path.exists():
                    shutil.move(str(hr_file_path), str(dest_dir / "HR" / hr_file_path.name))

    # 清理临时目录
    print("[INFO] Cleaning up temporary directory...")
    shutil.rmtree(temp_dir)
    print("\n✅ All data splitting and moving is complete!")

# --- 3. 脚本执行入口 ---
if __name__ == '__main__':
    run_stage3_and_cleanup()
