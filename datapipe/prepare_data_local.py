# -*- coding: utf-8 -*-
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
import argparse
import os
import re
import glob
import random
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
import time
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds
from tqdm import tqdm

# --- 1. 核心参数配置 (请根据您的环境进行调整) ---

# 路径配置
SCRIPT_DIR = Path(__file__).resolve().parent

# BASE_DATA_DIR = SCRIPT_DIR/ "data"  #Path("./data")
# ABSOLUTE_PATH="/mnt/lm_data_afs/wangzining/charles/AEF_swinir/data/2019-parsley"
ABSOLUTE_PATH="/mnt/lm_data_afs/wangzining/charles/AEF_swinir/data/2020-charles"
# ABSOLUTE_PATH="/mnt/lm_data_afs/wangzining/charles/AEF_swinir/data/2021-charles"
# ABSOLUTE_PATH="/mnt/lm_data_afs/wangzining/charles/AEF_swinir/data/2022"
# ABSOLUTE_PATH="/mnt/lm_data_afs/wangzining/charles/AEF_swinir/data/2023"
# ABSOLUTE_PATH="/mnt/lm_data_afs/wangzining/charles/AEF_swinir/data/2024"
BASE_DATA_DIR=Path(ABSOLUTE_PATH)
RAW_LANDSAT_DIR = BASE_DATA_DIR / "raw_landsat"
RAW_ALPHA_DIR = BASE_DATA_DIR / "raw_alphaearth"
PROCESSED_DATA_ROOT = BASE_DATA_DIR / "processed_data_SR_10m_32samples"

# ABSOLUTE_PATH="/mnt/lm_data_afs/wangzining/charles/AEF_swinir/data/Cloud_test"
# BASE_DATA_DIR=Path(ABSOLUTE_PATH)
# RAW_LANDSAT_DIR = BASE_DATA_DIR / "raw_landsat_LR_30m"
# RAW_ALPHA_DIR = BASE_DATA_DIR / "raw_alphaearth_HR_10m"
# PROCESSED_DATA_ROOT = BASE_DATA_DIR / "processed_data_SR_10m"

# 数据元信息
REF_PATH, REF_ROW = 132, 33
NUM_ALPHA_BANDS_TO_STACK = 64
ALPHAEARTH_BANDS_TO_USE = [f'A{i:02d}' for i in range(NUM_ALPHA_BANDS_TO_STACK)]
LR_PATCH_SIZE, SCALE_FACTOR, HR_PATCH_SIZE = 64, 3, 192

# 数据集划分与并行设置
TRAIN_RATIO = 0.5
VAL_RATIO = 0.25
RANDOM_SEED = 42
# 有效像素比例阈值：一个瓦片中有效像素必须达到这个比例才会被保留
VALID_PIXEL_RATIO_THRESHOLD = 0.98
HR_BAND_REQUIREMENT_RATIO = 1.0
NUM_WORKERS = max(1, (os.cpu_count() or 1) // 2)
# 与训练配置 `hr_zero_ratio_filter_threshold` 对齐：仅写出 max(LR 零像素占比, HR 零像素占比) 不超过此值的瓦片，
# 这样 patch_stats.log 与磁盘上的 tif 一致，AnytimeTemporalDataset 的 pair-zero 过滤不会把整数据集删光。
PAIR_ZERO_RATIO_MAX = 0.05
# 采样数量：None 表示 STAGE 2 处理全部候选直到结束；正整数表示 STAGE 2 在成功写出 N 对（通过 pair-zero）后停止。
# STAGE 1 始终跑完：枚举所有掩膜通过的候选任务；不再在 STAGE 1 用 N×倍率截断。
SAMPLE_NUM = None

# ==============================================================================
# 2. "工人" 函数 (用于并行处理，无需修改)
# ==============================================================================

def lr_aligned_hr_transform(lr_transform):
    """Return the 10 m HR grid anchored to the LR raster origin."""
    return lr_transform * Affine.scale(1 / SCALE_FACTOR, 1 / SCALE_FACTOR)


def has_same_grid_resolution(src, dst_transform, dst_crs):
    return (
        src.crs == dst_crs
        and np.isclose(src.transform.b, 0.0)
        and np.isclose(src.transform.d, 0.0)
        and np.isclose(dst_transform.b, 0.0)
        and np.isclose(dst_transform.d, 0.0)
        and np.isclose(src.transform.a, dst_transform.a)
        and np.isclose(src.transform.e, dst_transform.e)
    )


def read_alpha_band_on_grid(band_src, dst_shape, dst_transform, dst_crs):
    """
    Read one AlphaEarth band onto the requested LR-aligned HR grid.

    The source AlphaEarth band files may have different origins/sizes even after
    reprojection. Reading through rasterio.warp.reproject keeps every band on the
    same map grid and uses NaN for pixels outside a source band's coverage.
    """
    if has_same_grid_resolution(band_src, dst_transform, dst_crs):
        height, width = dst_shape
        left = dst_transform.c
        top = dst_transform.f
        right = left + dst_transform.a * width
        bottom = top + dst_transform.e * height
        src_window = from_bounds(left, bottom, right, top, band_src.transform)
        band = band_src.read(
            1,
            window=src_window,
            out_shape=dst_shape,
            out_dtype="float32",
            boundless=True,
            fill_value=np.nan,
            masked=True,
            resampling=Resampling.nearest,
        )
        return np.ma.filled(band, np.nan).astype(np.float32, copy=False)

    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=rasterio.band(band_src, 1),
        destination=dst,
        src_transform=band_src.transform,
        src_crs=band_src.crs,
        src_nodata=band_src.nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return dst


def alpha_band_valid_on_grid(band_src, dst_shape, dst_transform, dst_crs):
    """
    Return valid coverage for one band on the destination grid.

    Current AlphaEarth inputs are uint8 with no nodata; for those files, every
    source pixel is finite and validity is just source coverage. Avoid reading
    gigabytes of pixel values during STAGE 1 when the grid math is enough.
    """
    dtype = np.dtype(band_src.dtypes[0])
    can_use_coverage_only = (
        has_same_grid_resolution(band_src, dst_transform, dst_crs)
        and band_src.nodata is None
        and not np.issubdtype(dtype, np.floating)
    )
    if can_use_coverage_only:
        height, width = dst_shape
        col_offset = (dst_transform.c - band_src.transform.c) / band_src.transform.a
        row_offset = (dst_transform.f - band_src.transform.f) / band_src.transform.e
        if np.isclose(col_offset, round(col_offset)) and np.isclose(row_offset, round(row_offset)):
            cols = np.arange(width) + int(round(col_offset))
            rows = np.arange(height) + int(round(row_offset))
            valid_cols = (cols >= 0) & (cols < band_src.width)
            valid_rows = (rows >= 0) & (rows < band_src.height)
            return valid_rows[:, None] & valid_cols[None, :]

    return np.isfinite(read_alpha_band_on_grid(band_src, dst_shape, dst_transform, dst_crs))


def lr_grid_cache_key(lr_src):
    return (
        lr_src.crs.to_string() if lr_src.crs else None,
        lr_src.height,
        lr_src.width,
        tuple(round(v, 9) for v in lr_src.transform),
    )


def build_hr_valid_mask_lr_res(alpha_band_paths, lr_transform, lr_crs, lr_height, lr_width):
    """
    Build the HR validity mask on the LR grid without assuming identical HR shapes.

    A LR pixel is HR-valid only when all pixels in its SCALE_FACTOR x SCALE_FACTOR
    HR block have enough valid AlphaEarth bands.
    """
    required_bands = int(NUM_ALPHA_BANDS_TO_STACK * HR_BAND_REQUIREMENT_RATIO)
    hr_transform = lr_aligned_hr_transform(lr_transform)
    hr_width = lr_width * SCALE_FACTOR
    hr_valid_lr_res = np.zeros((lr_height, lr_width), dtype=bool)

    # Chunk by LR rows to avoid allocating a full float32 HR scene for every band.
    chunk_lr_rows = min(512, lr_height)
    for lr_row in tqdm(
        range(0, lr_height, chunk_lr_rows),
        desc="Building LR-aligned HR mask",
        leave=False,
    ):
        chunk_lr_h = min(chunk_lr_rows, lr_height - lr_row)
        chunk_hr_h = chunk_lr_h * SCALE_FACTOR
        hr_win = Window(0, lr_row * SCALE_FACTOR, hr_width, chunk_hr_h)
        chunk_transform = rasterio.windows.transform(hr_win, hr_transform)
        valid_band_count = np.zeros((chunk_hr_h, hr_width), dtype=np.uint8)

        for band_path in alpha_band_paths:
            with rasterio.open(band_path) as band_src:
                band_valid = alpha_band_valid_on_grid(
                    band_src,
                    (chunk_hr_h, hr_width),
                    chunk_transform,
                    lr_crs,
                )
            valid_band_count += band_valid.astype(np.uint8)

        hr_valid_full = valid_band_count >= required_bands
        hr_valid_lr_res[lr_row:lr_row + chunk_lr_h, :] = hr_valid_full.reshape(
            chunk_lr_h,
            SCALE_FACTOR,
            lr_width,
            SCALE_FACTOR,
        ).all(axis=(1, 3))

    return hr_valid_lr_res

def patch_passes_planning_thresholds(lr_tile, hr_tile):
    """
    与 STAGE 1 的 combined_mask 一致：LR 每像素全波段非 nan 且非 0；
    HR 每像素非 nan 的波段数 >= required_bands；再下采样到 LR 网格，块内 3x3 HR 像素须全为 True。
    最后要求 patch 内 combined 有效像素比例 >= VALID_PIXEL_RATIO_THRESHOLD。
    """
    required_bands = int(NUM_ALPHA_BANDS_TO_STACK * HR_BAND_REQUIREMENT_RATIO)
    lr_valid = (~np.any(np.isnan(lr_tile), axis=0)) & np.all(lr_tile != 0, axis=0)
    valid_hr_full = np.sum(~np.isnan(hr_tile), axis=0) >= required_bands
    ph, pw = lr_valid.shape
    hr_h, hr_w = valid_hr_full.shape
    if hr_h != ph * SCALE_FACTOR or hr_w != pw * SCALE_FACTOR:
        return False
    hr_valid_lr_res = valid_hr_full.reshape(ph, SCALE_FACTOR, pw, SCALE_FACTOR).all(axis=(1, 3))
    combined = lr_valid & hr_valid_lr_res
    return np.count_nonzero(combined) / combined.size >= VALID_PIXEL_RATIO_THRESHOLD


def create_one_tile(args):
    """
    内存高效的函数：只创建一对 LR/HR 瓦片。
    只读取所需的小窗口，绝不加载整个文件。
    args: (landsat_path, alpha_band_paths, r, c[, pair_zero_max]) — pair_zero_max 与训练 hr_zero_ratio_filter_threshold 一致。
    """
    if len(args) == 5:
        landsat_path, alpha_band_paths, r, c, pair_zero_max = args
    else:
        landsat_path, alpha_band_paths, r, c = args
        pair_zero_max = PAIR_ZERO_RATIO_MAX
    try:
        date_str = re.search(r'_(\d{8})_', Path(landsat_path).name).group(1)

        with rasterio.open(landsat_path) as lr_src:
            lr_meta = lr_src.meta
            lr_crs = lr_src.crs
            hr_base_transform = lr_aligned_hr_transform(lr_src.transform)
            lr_win = Window(c, r, LR_PATCH_SIZE, LR_PATCH_SIZE)
            lr_tile = lr_src.read(window=lr_win)

        hr_r, hr_c = r * SCALE_FACTOR, c * SCALE_FACTOR
        hr_win = Window(hr_c, hr_r, HR_PATCH_SIZE, HR_PATCH_SIZE)
        hr_tile_transform = rasterio.windows.transform(hr_win, hr_base_transform)
        hr_tile = np.full(
            (NUM_ALPHA_BANDS_TO_STACK, HR_PATCH_SIZE, HR_PATCH_SIZE),
            np.nan,
            dtype=np.float32,
        )

        for i, band_path in enumerate(alpha_band_paths):
            with rasterio.open(band_path) as band_src:
                hr_tile[i, :, :] = read_alpha_band_on_grid(
                    band_src,
                    (HR_PATCH_SIZE, HR_PATCH_SIZE),
                    hr_tile_transform,
                    lr_crs,
                )

        # STAGE 1 规划掩码（与 patch_passes_planning_thresholds 一致）；pair-zero 在下方单独筛，与训练 manifest 对齐
        if not patch_passes_planning_thresholds(lr_tile, hr_tile):
            return None

        lr_min = np.nanmin(lr_tile)
        lr_max = np.nanmax(lr_tile)
        hr_min = np.nanmin(hr_tile)
        hr_max = np.nanmax(hr_tile)
        lr_nan_ratio = np.isnan(lr_tile).sum() / lr_tile.size
        lr_zero_ratio = np.sum(lr_tile == 0) / lr_tile.size
        hr_nan_ratio = np.isnan(hr_tile).sum() / hr_tile.size
        hr_zero_ratio = np.sum(hr_tile == 0) / hr_tile.size

        # 与 datapipe.datasets._filter_files_by_hr_zero_ratio / manifest 一致：超过阈值则不写盘、不写 patch_stats
        pair_zero_ratio = max(lr_zero_ratio, hr_zero_ratio)
        if pair_zero_ratio > pair_zero_max:
            return None

        # 可选：patch归一化检查（如需归一化可在此处加）
        # 归一化示例（如需）：
        # lr_tile = (lr_tile - lr_min) / (lr_max - lr_min + 1e-8)
        # hr_tile = (hr_tile - hr_min) / (hr_max - hr_min + 1e-8)
        # ================== Landsat8 LR patch物理还原与归一化 ==================
        lr_tile = lr_tile.astype(np.float32)
        n_lr_bands = min(10, lr_tile.shape[0])
        for i in range(n_lr_bands):
            lr_tile[i] = lr_tile[i] - np.nanmin(lr_tile[i])
            bmax = np.nanmax(lr_tile[i])
            lr_tile[i] = lr_tile[i] / (bmax if bmax > 1e-8 else 1.0)
        # ================== END LR归一化 ==================
        # 记录patch统计信息
        try:
            with open("patch_stats.log", "a") as logf:
                logf.write(
                    f"{landsat_path},{r},{c},"
                    f"LR[min={lr_min},max={lr_max},nan={lr_nan_ratio:.2%},zero={lr_zero_ratio:.2%}],"
                    f"HR[min={hr_min},max={hr_max},nan={hr_nan_ratio:.2%},zero={hr_zero_ratio:.2%}]\n"
                )
        except Exception as logerr:
            print(f"[WARN] Failed to write patch_stats.log: {logerr}", file=sys.stderr)

        # 瓦片命名规则：日期_tile_行号_列号.tif
        fname = f"{date_str}_tile_{r//LR_PATCH_SIZE}_{c//LR_PATCH_SIZE}.tif"

        # 更新瓦片的元数据
        lr_tile_meta = lr_meta.copy()
        lr_tile_meta.update({
            'height': LR_PATCH_SIZE, 'width': LR_PATCH_SIZE,
            'transform': rasterio.windows.transform(lr_win, lr_meta['transform'])
        })

        hr_tile_meta = lr_meta.copy()
        hr_tile_meta.update({
            'count': NUM_ALPHA_BANDS_TO_STACK, 'height': HR_PATCH_SIZE, 'width': HR_PATCH_SIZE,
            'transform': hr_tile_transform, 'crs': lr_crs, 'dtype': 'float32', 'nodata': np.nan
        })
        print(f"LR patch: nan_ratio={lr_nan_ratio:.2%}, zero_ratio={lr_zero_ratio:.2%}")
        print(f"HR patch: nan_ratio={hr_nan_ratio:.2%}, zero_ratio={hr_zero_ratio:.2%}")
        return (fname, lr_tile, lr_tile_meta, hr_tile, hr_tile_meta)

    except Exception as e:
        # 在并行任务中打印错误，但主进程会继续
        print(f"Warning: Error processing tile at (r={r}, c={c}) for {landsat_path}: {e}", file=sys.stderr)
        return None

# ==============================================================================
# 3. 主处理函数 (新的并行调度核心)
# ==============================================================================

def main_processing(sample_num=None, pair_zero_max=None):
    """
    采用两阶段并行处理，并融合您脚本中的所有检查逻辑。

    sample_num: 若为 None 则使用脚本中的 SAMPLE_NUM；否则覆盖之。
                None：STAGE 1 枚举全部掩膜候选，STAGE 2 全部处理；正整数：STAGE 1 仍枚举全部候选，
                STAGE 2 依次处理直至成功写出 N 对（通过 pair-zero）后停止。
    pair_zero_max: 默认 PAIR_ZERO_RATIO_MAX，应与 YAML 中 hr_zero_ratio_filter_threshold 一致。
    """
    limit = SAMPLE_NUM if sample_num is None else sample_num
    if limit is not None and limit <= 0:
        print("❌ CRITICAL: sample_num / SAMPLE_NUM must be None (all) or a positive integer.", file=sys.stderr)
        return
    pzm = float(PAIR_ZERO_RATIO_MAX if pair_zero_max is None else pair_zero_max)

    # 清理旧数据
    if PROCESSED_DATA_ROOT.exists():
        print(f"[WARN] Processed data dir '{PROCESSED_DATA_ROOT}' already exists.")
        confirm = input("该目录已存在，是否备份并继续？(y/n): ").strip().lower()
        if confirm not in ("y", "yes"):
            print("操作已取消，无数据被删除。")
            return
        backup_dir = PROCESSED_DATA_ROOT.parent / f"{PROCESSED_DATA_ROOT.name}_backup_{int(time.time())}"
        print(f"自动备份到: {backup_dir}")
        shutil.move(str(PROCESSED_DATA_ROOT), str(backup_dir))

    # 创建临时目录用于存放所有生成的瓦片
    temp_dir = PROCESSED_DATA_ROOT / "all_tiles_temp"
    (temp_dir / "LR").mkdir(parents=True)
    (temp_dir / "HR").mkdir(parents=True)
    print(f"[INFO] Created temporary directory: {temp_dir}")
    print(
        f"[INFO] Pair-zero export filter: max(LR,HR) zero-pixel ratio <= {pzm:.4f} "
        f"(align with config hr_zero_ratio_filter_threshold; rejects candidates before write)"
    )

    stage2_note = (
        f"STAGE 2 will stop after {limit} successful exports (pair-zero OK)."
        if limit is not None
        else "STAGE 2 will process every planned task."
    )
    print(f"\n===== STAGE 1: Planning ALL mask-valid tiling tasks ({stage2_note}) =====")
    tasks = []

    all_landsat_files = sorted(RAW_LANDSAT_DIR.glob(f'L8_{REF_PATH}{REF_ROW}_*_Masked.tif'))
    if not all_landsat_files:
        print(f"❌ CRITICAL: No Landsat files found in {RAW_LANDSAT_DIR}. Please check the path.", file=sys.stderr)
        return

    print(f"[INFO] Found {len(all_landsat_files)} time steps to process.")

    hr_mask_cache = {}

    # 遍历时相，生成所有掩膜通过的瓦片坐标（不根据 sample_num 截断）
    for landsat_path in tqdm(all_landsat_files, desc="Scanning dates for task planning"):
        date_str = re.search(r'_(\d{8})_', landsat_path.name).group(1)

        # 检查对应的HR波段是否存在
        alpha_paths_str = [str(p) for p in [next(iter(RAW_ALPHA_DIR.glob(f"AlphaEarth_Path{REF_PATH}_Row{REF_ROW}_reprojected_{b}.tif")), None) for b in ALPHAEARTH_BANDS_TO_USE]]
        if any(p == 'None' for p in alpha_paths_str):
            print(f"⚠️ Warning: Incomplete HR bands for date {date_str}. Skipping this date.", file=sys.stderr)
            continue

        # 生成有效像素掩码
        with rasterio.open(landsat_path) as lr_src:
            lr_h_orig, lr_w_orig = lr_src.height, lr_src.width
            grid_key = lr_grid_cache_key(lr_src)
            if grid_key not in hr_mask_cache:
                hr_mask_cache[grid_key] = build_hr_valid_mask_lr_res(
                    alpha_paths_str,
                    lr_src.transform,
                    lr_src.crs,
                    lr_src.height,
                    lr_src.width,
                )
            hr_valid_mask_lr_res = hr_mask_cache[grid_key]
            proc_h = min(lr_h_orig, hr_valid_mask_lr_res.shape[0])
            proc_w = min(lr_w_orig, hr_valid_mask_lr_res.shape[1])

            lr_data = lr_src.read(window=Window(0, 0, proc_w, proc_h))
            lr_valid_mask = ~np.any(np.isnan(lr_data), axis=0) & np.all(lr_data != 0, axis=0)
            combined_mask = lr_valid_mask & hr_valid_mask_lr_res[:proc_h, :proc_w]

        # 根据掩码生成任务列表
        for r in range(0, proc_h - LR_PATCH_SIZE + 1, LR_PATCH_SIZE):
            for c in range(0, proc_w - LR_PATCH_SIZE + 1, LR_PATCH_SIZE):
                if np.count_nonzero(combined_mask[r:r+LR_PATCH_SIZE, c:c+LR_PATCH_SIZE]) / (LR_PATCH_SIZE**2) >= VALID_PIXEL_RATIO_THRESHOLD:
                    tasks.append((str(landsat_path), alpha_paths_str, r, c, pzm))

    if not tasks:
        print("❌ CRITICAL: No valid tiles found to process. Check data quality or thresholds.", file=sys.stderr)
        return

    print(
        f"\n✅ STAGE 1 done. Planned {len(tasks)} candidate tasks (mask-valid). "
        + (f"STAGE 2 target: {limit} exports passing pair-zero filter." if limit is not None else "STAGE 2: process all.")
    )

    _cpu = os.cpu_count() or 1
    effective_workers = int(min(NUM_WORKERS, _cpu) if NUM_WORKERS > 0 else _cpu)
    print(f"\n===== STAGE 2: Starting parallel tiling with {effective_workers} CPU cores... =====")

    successful_count = 0
    early_stop_success = False
    pool = Pool(processes=effective_workers)
    try:
        for result in tqdm(pool.imap_unordered(create_one_tile, tasks), total=len(tasks), desc="Processing Tiles"):
            if result:
                fname, lr_tile, lr_meta, hr_tile, hr_meta = result
                with rasterio.open(temp_dir / "LR" / fname, "w", **lr_meta) as dst:
                    dst.write(lr_tile)
                with rasterio.open(temp_dir / "HR" / fname, "w", **hr_meta) as dst:
                    dst.write(hr_tile)
                successful_count += 1
                if limit is not None and successful_count >= limit:
                    print(f"\n✅ STAGE 2 early stop: got {limit} tiles passing pair-zero filter.")
                    early_stop_success = True
                    break
    finally:
        if early_stop_success:
            pool.terminate()
        else:
            pool.close()
        pool.join()

    print(f"\n✅ STAGE 2 complete. Successfully created {successful_count} tiles.")
    if limit is not None and successful_count < limit:
        print(
            f"⚠️ [WARN] Only {successful_count}/{limit} tiles passed pair-zero filter (<= {pzm:.4f}) "
            f"after scanning {len(tasks)} candidates. Add more scenes/tiles or raise --pair-zero-max / training threshold.",
            file=sys.stderr,
        )
    if successful_count == 0:
        print(
            f"❌ CRITICAL: No tiles passed pair-zero filter (<= {pzm:.4f}). "
            "Your HR/LR patches may have too many literal zeros; try a higher --pair-zero-max or check data nodata handling.",
            file=sys.stderr,
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

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
    parser = argparse.ArgumentParser(description="Prepare local LR/HR tiles from Landsat + AlphaEarth.")
    parser.add_argument(
        "--sample-num",
        type=int,
        default=None,
        metavar="N",
        help="Target successful LR/HR pairs after pair-zero filter: STAGE 1 plans ALL mask-valid tasks, STAGE 2 stops after N successes (overrides SAMPLE_NUM). Omit = use SAMPLE_NUM; both unset = export all passing.",
    )
    parser.add_argument(
        "--pair-zero-max",
        type=float,
        default=None,
        metavar="T",
        help=f"Max allowed max(LR,HR) zero-pixel ratio for export (default: {PAIR_ZERO_RATIO_MAX}, same as training hr_zero_ratio_filter_threshold).",
    )
    args = parser.parse_args()
    main_processing(sample_num=args.sample_num, pair_zero_max=args.pair_zero_max)
