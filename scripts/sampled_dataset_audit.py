#!/usr/bin/env python3
import argparse
import random
import re
from pathlib import Path

import numpy as np
import rasterio


def random_sample_files(dir_path: Path, n: int):
    files = [p for p in dir_path.glob("*.tif")]
    if not files:
        return []
    n = min(n, len(files))
    return random.sample(files, n)


def read_stats(tif_path: Path):
    with rasterio.open(tif_path) as ds:
        arr = ds.read()
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "nan_count": int(np.isnan(arr).sum()),
            "height": ds.height,
            "width": ds.width,
            "count": ds.count,
        }


def normalize_per_channel(arr: np.ndarray, lo=1, hi=99):
    arr = arr.astype(np.float32)
    out = []
    for i in range(arr.shape[0]):
        band = arr[i]
        if np.all(band == 0):
            out.append(np.zeros_like(band, dtype=np.float32))
            continue
        lo_p, hi_p = np.percentile(band, (lo, hi))
        if hi_p - lo_p < 1e-6:
            out.append(np.zeros_like(band, dtype=np.float32))
            continue
        clipped = np.clip(band, lo_p, hi_p)
        norm = (clipped - lo_p) / (hi_p - lo_p)
        out.append(norm * 2.0 - 1.0)
    return np.stack(out, axis=0)


def check_raw(raw_dir: Path, path_row: str, expected_year: str, samples: int):
    print("\n==== RAW LANDSAT CHECK ====")
    picked = random_sample_files(raw_dir, samples)
    if not picked:
        print(f"[ERROR] No raw tif files found in {raw_dir}")
        return

    pat = re.compile(r"^L8_(\d{5})_(\d{8})_Masked\.tif$")
    for p in picked:
        m = pat.match(p.name)
        if not m:
            print(f"[WARN] Unexpected raw filename: {p.name}")
            continue
        pr, ymd = m.group(1), m.group(2)
        year = ymd[:4]
        pr_ok = pr == path_row
        year_ok = year == expected_year
        print(f"[RAW] {p.name} | pathrow_ok={pr_ok} year_ok={year_ok}")


def check_split(split_dir: Path, samples: int):
    lr_dir = split_dir / "LR"
    hr_dir = split_dir / "HR"
    split_name = split_dir.name

    print(f"\n==== SPLIT CHECK: {split_name} ====")
    if not lr_dir.exists() or not hr_dir.exists():
        print(f"[ERROR] Missing LR or HR dir under {split_dir}")
        return

    picked_lr = random_sample_files(lr_dir, samples)
    if not picked_lr:
        print(f"[ERROR] No LR tif files in {lr_dir}")
        return

    for lr in picked_lr:
        hr = hr_dir / lr.name
        if not hr.exists():
            print(f"[ERROR] Missing HR pair for {lr.name}")
            continue

        # filename/date sanity
        m = re.match(r"^(\d{8})_tile_(\d+_\d+)\.tif$", lr.name)
        if not m:
            print(f"[WARN] Unexpected processed filename: {lr.name}")
            continue

        ymd = m.group(1)
        lr_stats = read_stats(lr)
        hr_stats = read_stats(hr)

        h_ratio = hr_stats["height"] / max(lr_stats["height"], 1)
        w_ratio = hr_stats["width"] / max(lr_stats["width"], 1)

        print(
            f"[PAIR] {lr.name} | date={ymd} | "
            f"LR CxHxW={lr_stats['shape']} HR CxHxW={hr_stats['shape']} | "
            f"scale_h={h_ratio:.2f} scale_w={w_ratio:.2f}"
        )
        print(
            f"       LR dtype={lr_stats['dtype']} min={lr_stats['min']:.4f} max={lr_stats['max']:.4f} nan={lr_stats['nan_count']} | "
            f"HR dtype={hr_stats['dtype']} min={hr_stats['min']:.4f} max={hr_stats['max']:.4f} nan={hr_stats['nan_count']}"
        )

        # normalization sanity check (same style as dataset)
        with rasterio.open(lr) as ds:
            lr_raw = ds.read()
        with rasterio.open(hr) as ds:
            hr_raw = ds.read()
        lr_norm = normalize_per_channel(lr_raw)
        hr_norm = normalize_per_channel(hr_raw)

        print(
            f"       NormRange LR=[{float(np.min(lr_norm)):.4f}, {float(np.max(lr_norm)):.4f}] "
            f"HR=[{float(np.min(hr_norm)):.4f}, {float(np.max(hr_norm)):.4f}]"
        )


def main():
    parser = argparse.ArgumentParser(description="Sample-based dataset audit for Landsat/processed_data")
    parser.add_argument("--raw-dir", type=str, default="data/raw_landsat")
    parser.add_argument("--processed-root", type=str, default="data/processed_data")
    parser.add_argument("--path-row", type=str, default="13233")
    parser.add_argument("--expected-year", type=str, default="2018")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    raw_dir = Path(args.raw_dir)
    processed_root = Path(args.processed_root)

    print("==== SAMPLE DATASET AUDIT START ====")
    print(f"raw_dir={raw_dir}")
    print(f"processed_root={processed_root}")
    print(f"samples={args.samples}, seed={args.seed}")

    check_raw(raw_dir, args.path_row, args.expected_year, args.samples)

    for split in ["train", "val", "test"]:
        check_split(processed_root / split, args.samples)

    print("\n==== SAMPLE DATASET AUDIT DONE ====")


if __name__ == "__main__":
    main()
