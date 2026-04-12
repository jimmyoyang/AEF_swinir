#!/usr/bin/env python3
import argparse
import random
import re
from pathlib import Path

import numpy as np
import rasterio


def parse_date_from_processed_name(name: str):
    m = re.match(r"^(\d{8})_tile_\d+_\d+\.tif$", name)
    return m.group(1) if m else None


def parse_date_from_raw_name(name: str):
    m = re.match(r"^L8_(\d{5})_(\d{8})_Masked\.tif$", name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def list_tifs(dir_path: Path):
    return sorted([p for p in dir_path.glob("*.tif") if p.is_file()])


def check_split(root: Path, split: str):
    lr_dir = root / split / "LR"
    hr_dir = root / split / "HR"

    lr_files = list_tifs(lr_dir)
    hr_files = list_tifs(hr_dir)

    lr_names = {p.name for p in lr_files}
    hr_names = {p.name for p in hr_files}

    matched = lr_names & hr_names
    only_lr = lr_names - hr_names
    only_hr = hr_names - lr_names

    dates = sorted({parse_date_from_processed_name(n) for n in matched if parse_date_from_processed_name(n)})

    return {
        "split": split,
        "lr_count": len(lr_files),
        "hr_count": len(hr_files),
        "matched_count": len(matched),
        "only_lr_count": len(only_lr),
        "only_hr_count": len(only_hr),
        "dates": dates,
        "sample_names": sorted(list(matched))[:5],
        "lr_dir": lr_dir,
        "hr_dir": hr_dir,
    }


def read_stats(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read()
        return {
            "path": str(path),
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "count": ds.count,
            "height": ds.height,
            "width": ds.width,
        }


def robust_per_image_normalize(img, lo=1, hi=99):
    img_float = img.astype(np.float32)
    normalized_bands = []
    for i in range(img_float.shape[0]):
        band = img_float[i]
        if np.all(band == 0):
            normalized_bands.append(band.astype(np.float32))
            continue

        lo_p, hi_p = np.percentile(band, (lo, hi))
        if hi_p - lo_p < 1e-6:
            normalized_band = np.zeros_like(band, dtype=np.float32)
        else:
            clipped_band = np.clip(band, lo_p, hi_p)
            normalized_band = (clipped_band - lo_p) / (hi_p - lo_p)

        normalized_bands.append(normalized_band * 2 - 1)

    return np.stack(normalized_bands, axis=0)


def check_loader_normalization(root: Path, split: str, sample_name: str):
    lr_path = root / split / "LR" / sample_name
    hr_path = root / split / "HR" / sample_name

    with rasterio.open(lr_path) as ds:
        lr_raw = ds.read()
    with rasterio.open(hr_path) as ds:
        hr_raw = ds.read()

    lr_norm = robust_per_image_normalize(lr_raw)
    hr_norm = robust_per_image_normalize(hr_raw)

    return {
        "lr_norm_min": float(np.min(lr_norm)),
        "lr_norm_max": float(np.max(lr_norm)),
        "hr_norm_min": float(np.min(hr_norm)),
        "hr_norm_max": float(np.max(hr_norm)),
        "lr_norm_shape": lr_norm.shape,
        "hr_norm_shape": hr_norm.shape,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate processed dataset integrity and basic statistics.")
    parser.add_argument("--processed-root", type=str, default="data/processed_data")
    parser.add_argument("--raw-root", type=str, default="data/raw_landsat")
    parser.add_argument("--path-row", type=str, default="13233", help="Expected WRS path/row code in raw file names.")
    parser.add_argument("--expected-year", type=str, default="2018")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    processed_root = Path(args.processed_root)
    raw_root = Path(args.raw_root)

    print("==== 1) Split file integrity ====")
    split_reports = [check_split(processed_root, s) for s in ["train", "val", "test"]]
    for rep in split_reports:
        print(f"[{rep['split']}] LR={rep['lr_count']} HR={rep['hr_count']} matched={rep['matched_count']} only_lr={rep['only_lr_count']} only_hr={rep['only_hr_count']}")

    all_dates = sorted({d for rep in split_reports for d in rep["dates"]})
    print(f"[all splits] unique processed dates={len(all_dates)}")
    if all_dates:
        print(f"[all splits] date range: {all_dates[0]} -> {all_dates[-1]}")

    wrong_year = [d for d in all_dates if not d.startswith(args.expected_year)]
    if wrong_year:
        print(f"[WARN] Found dates not in year {args.expected_year}: {wrong_year[:10]}")
    else:
        print(f"[OK] All processed dates are in year {args.expected_year}.")

    print("\n==== 2) Raw source consistency (Path/Row + date overlap) ====")
    raw_files = sorted(raw_root.glob("L8_*_*_Masked.tif"))
    raw_dates = []
    bad_pr = []
    for p in raw_files:
        pr, d = parse_date_from_raw_name(p.name)
        if pr is None:
            continue
        if pr != args.path_row:
            bad_pr.append(p.name)
        raw_dates.append(d)

    raw_dates = sorted(set(raw_dates))
    print(f"[raw] files={len(raw_files)} parsed_dates={len(raw_dates)}")
    if bad_pr:
        print(f"[WARN] Found raw files not matching path/row {args.path_row}: {bad_pr[:5]}")
    else:
        print(f"[OK] Raw files match path/row={args.path_row}.")

    proc_set = set(all_dates)
    raw_set = set(raw_dates)
    miss_in_raw = sorted(proc_set - raw_set)
    miss_in_processed = sorted(raw_set - proc_set)

    print(f"[date overlap] processed_not_in_raw={len(miss_in_raw)}, raw_not_in_processed={len(miss_in_processed)}")
    if miss_in_raw:
        print(f"[WARN] processed dates missing in raw: {miss_in_raw[:10]}")
    if miss_in_processed:
        print(f"[INFO] raw dates not used in processed (possibly filtered): {miss_in_processed[:10]}")

    print("\n==== 3) Random pair stats (channels, shapes, value range) ====")
    # Sample from val+test to focus on evaluation data
    candidate_names = []
    for split in ["val", "test"]:
        rep = next(x for x in split_reports if x["split"] == split)
        names = [p.name for p in list_tifs(rep["lr_dir"]) if (rep["hr_dir"] / p.name).exists()]
        candidate_names.extend([(split, n) for n in names])

    if not candidate_names:
        print("[ERROR] No matched pairs in val/test.")
        return

    k = min(args.samples, len(candidate_names))
    picks = random.sample(candidate_names, k=k)

    for i, (split, name) in enumerate(picks, start=1):
        lr_path = processed_root / split / "LR" / name
        hr_path = processed_root / split / "HR" / name

        lr_s = read_stats(lr_path)
        hr_s = read_stats(hr_path)
        norm_s = check_loader_normalization(processed_root, split, name)

        scale_h = hr_s["height"] / lr_s["height"]
        scale_w = hr_s["width"] / lr_s["width"]

        print(f"\nSample {i}: {split}/{name}")
        print(f"  LR shape={lr_s['shape']} dtype={lr_s['dtype']} raw[min,max]=[{lr_s['min']:.4f},{lr_s['max']:.4f}] mean={lr_s['mean']:.4f}")
        print(f"  HR shape={hr_s['shape']} dtype={hr_s['dtype']} raw[min,max]=[{hr_s['min']:.4f},{hr_s['max']:.4f}] mean={hr_s['mean']:.4f}")
        print(f"  Channels LR={lr_s['count']} HR={hr_s['count']} | spatial scale h={scale_h:.2f} w={scale_w:.2f}")
        print(f"  After robust_per_image_normalize: LR[min,max]=[{norm_s['lr_norm_min']:.4f},{norm_s['lr_norm_max']:.4f}] HR[min,max]=[{norm_s['hr_norm_min']:.4f},{norm_s['hr_norm_max']:.4f}]")

    print("\n==== 4) Expected checks summary ====")
    print("[PASS criteria]")
    print("  - train/val/test matched_count > 0 and only_lr/only_hr = 0")
    print(f"  - all processed dates in {args.expected_year}")
    print(f"  - raw path/row = {args.path_row}")
    print("  - LR/HR channel counts are consistent with your task definition")
    print("  - normalized ranges approximately within [-1, 1]")


if __name__ == "__main__":
    main()
