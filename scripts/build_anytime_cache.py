#!/usr/bin/env python3
import argparse
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datapipe.datasets import _compute_hard_valid_mask, _select_reflectance_bands, robust_per_image_normalize


def group_lr_files_by_tile(lr_dir: Path):
    grouped = defaultdict(list)
    for p in lr_dir.glob("*_tile_*.tif"):
        m = re.search(r"tile_(\d+_\d+)\.tif", p.name)
        if m is None:
            continue
        grouped[m.group(1)].append(p)
    for k in list(grouped.keys()):
        grouped[k].sort()
    return grouped


def compute_day_of_year(file_name: str):
    date_str = file_name.split("_")[0]
    y = int(date_str[0:4])
    m = int(date_str[4:6])
    d = int(date_str[6:8])
    # numpy datetime trick for DOY
    dt = np.datetime64(f"{y:04d}-{m:02d}-{d:02d}")
    year_start = np.datetime64(f"{y:04d}-01-01")
    return int((dt - year_start).astype(int) + 1)


def build_split_cache(
    lr_dir: Path,
    out_dir: Path,
    max_tiles: int = 0,
    valid_mask_band_count: int = 0,
    reflectance_band_count: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_lr_files_by_tile(lr_dir)
    tile_ids = sorted(grouped.keys())
    if max_tiles and max_tiles > 0:
        tile_ids = tile_ids[:max_tiles]

    print(f"[cache] lr_dir={lr_dir}")
    print(f"[cache] out_dir={out_dir}")
    print(f"[cache] tiles={len(tile_ids)}")

    for tile_id in tqdm(tile_ids, desc=f"build-cache:{lr_dir.parent.name}/{lr_dir.name}", dynamic_ncols=True):
        build_one_tile_cache(
            tile_id,
            [str(p) for p in grouped[tile_id]],
            str(out_dir),
            valid_mask_band_count,
            reflectance_band_count,
        )


def build_split_cache_parallel(
    lr_dir: Path,
    out_dir: Path,
    max_tiles: int = 0,
    num_workers: int = 4,
    valid_mask_band_count: int = 0,
    reflectance_band_count: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_lr_files_by_tile(lr_dir)
    tile_ids = sorted(grouped.keys())
    if max_tiles and max_tiles > 0:
        tile_ids = tile_ids[:max_tiles]

    print(f"[cache] lr_dir={lr_dir}")
    print(f"[cache] out_dir={out_dir}")
    print(f"[cache] tiles={len(tile_ids)}")
    print(f"[cache] num_workers={num_workers}")

    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for tile_id in tile_ids:
            files = [str(p) for p in grouped[tile_id]]
            futures.append(
                ex.submit(
                    build_one_tile_cache,
                    tile_id,
                    files,
                    str(out_dir),
                    valid_mask_band_count,
                    reflectance_band_count,
                )
            )

        pbar = tqdm(total=len(futures), desc=f"build-cache:{lr_dir.parent.name}/{lr_dir.name}", dynamic_ncols=True)
        try:
            for fut in as_completed(futures):
                tile_id, err = fut.result()
                if err is not None:
                    print(f"[cache][warn] tile={tile_id} failed: {err}")
                pbar.update(1)
        finally:
            pbar.close()


def build_one_tile_cache(
    tile_id: str,
    file_paths,
    out_dir: str,
    valid_mask_band_count: int = 0,
    reflectance_band_count: int = 0,
):
    try:
        file_names = []
        refl_list = []
        hard_mask_list = []
        doy_list = []

        for p_str in file_paths:
            p = Path(p_str)
            with rasterio.open(p) as src:
                arr = src.read()  # (C,H,W)
            refl_input = _select_reflectance_bands(arr, reflectance_band_count=reflectance_band_count)
            refl = robust_per_image_normalize(refl_input).astype(np.float32)
            hard_mask = _compute_hard_valid_mask(arr, valid_mask_band_count=valid_mask_band_count)
            doy = compute_day_of_year(p.name)

            file_names.append(p.name)
            refl_list.append(refl)
            hard_mask_list.append(hard_mask)
            doy_list.append(doy)

        if not refl_list:
            return tile_id, None

        base = Path(out_dir) / f"tile_{tile_id}"
        np.save(str(base) + "_reflectance.npy", np.stack(refl_list, axis=0).astype(np.float16))
        np.save(str(base) + "_hard_valid_mask.npy", np.stack(hard_mask_list, axis=0).astype(np.float16))
        np.save(str(base) + "_day_of_year.npy", np.array(doy_list, dtype=np.int16))
        with (Path(out_dir) / f"tile_{tile_id}_file_names.txt").open("w", encoding="utf-8") as f:
            for n in file_names:
                f.write(n + "\n")
        return tile_id, None
    except Exception as e:
        return tile_id, str(e)


def main():
    parser = argparse.ArgumentParser(description="Build offline cache for AnytimeTemporalDataset")
    parser.add_argument("--lr_dir", type=str, required=True, help="Path to LR directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Cache output directory")
    parser.add_argument("--max_tiles", type=int, default=0, help="Only cache first N tiles (0 means all)")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel workers for cache build")
    parser.add_argument(
        "--valid_mask_band_count",
        type=int,
        default=0,
        help="Leading LR bands used for hard valid mask; 0 auto-detects and ignores trailing QA/mask-like bands.",
    )
    parser.add_argument(
        "--reflectance_band_count",
        type=int,
        default=0,
        help="Leading LR bands stored as model reflectance input; 0 keeps all LR bands.",
    )
    args = parser.parse_args()
    if args.num_workers <= 1:
        build_split_cache(
            Path(args.lr_dir),
            Path(args.out_dir),
            max_tiles=args.max_tiles,
            valid_mask_band_count=args.valid_mask_band_count,
            reflectance_band_count=args.reflectance_band_count,
        )
    else:
        build_split_cache_parallel(
            Path(args.lr_dir),
            Path(args.out_dir),
            max_tiles=args.max_tiles,
            num_workers=args.num_workers,
            valid_mask_band_count=args.valid_mask_band_count,
            reflectance_band_count=args.reflectance_band_count,
        )


if __name__ == "__main__":
    main()
