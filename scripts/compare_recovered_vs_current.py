#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import rasterio

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


def norm_stem(path: Path) -> str:
    name = path.stem
    if name.startswith("pred_"):
        name = name[5:]
    return name


def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read().astype(np.float32)


def calc_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    data_range = float(gt.max() - gt.min())
    if data_range <= 1e-12:
        data_range = 1.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def calc_ssim_mean(pred: np.ndarray, gt: np.ndarray) -> float:
    if not HAS_SKIMAGE:
        return float("nan")

    scores = []
    channels = pred.shape[0]
    for i in range(channels):
        p = pred[i]
        g = gt[i]
        data_range = float(g.max() - g.min())
        if data_range <= 1e-12:
            data_range = 1.0
        scores.append(ssim_fn(g, p, data_range=data_range))
    return float(np.mean(scores))


def evaluate_dir(pred_dir: Path, gt_dir: Path):
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    gt_map = {norm_stem(p): p for p in gt_dir.glob("*.tif")}
    pred_files = sorted(pred_dir.glob("*.tif"))

    rows = []
    for pred_path in pred_files:
        key = norm_stem(pred_path)
        gt_path = gt_map.get(key)
        if gt_path is None:
            continue

        pred = read_tif(pred_path)
        gt = read_tif(gt_path)

        if pred.shape != gt.shape:
            print(
                f"[WARN] shape mismatch, skip {pred_path.name}: "
                f"pred={pred.shape}, gt={gt.shape}"
            )
            continue

        rows.append(
            {
                "file": pred_path.name,
                "psnr": calc_psnr(pred, gt),
                "ssim": calc_ssim_mean(pred, gt),
            }
        )

    if not rows:
        raise RuntimeError(f"No matched TIFF pairs found under: {pred_dir}")

    mean_psnr = float(np.mean([r["psnr"] for r in rows]))
    if HAS_SKIMAGE:
        mean_ssim = float(np.mean([r["ssim"] for r in rows]))
    else:
        mean_ssim = float("nan")

    return rows, mean_psnr, mean_ssim


def main():
    parser = argparse.ArgumentParser(
        description="Compare recovered vs current prediction directories against GT."
    )
    parser.add_argument(
        "--recovered-dir",
        type=str,
        required=True,
        help="Recovered model prediction directory containing *.tif",
    )
    parser.add_argument(
        "--current-dir",
        type=str,
        required=True,
        help="Current model prediction directory containing *.tif",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="GT HR directory containing *.tif",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="debug_output/compare_3b_recovered_vs_current.csv",
        help="Output CSV file path",
    )
    args = parser.parse_args()

    recovered_dir = Path(args.recovered_dir)
    current_dir = Path(args.current_dir)
    gt_dir = Path(args.gt_dir)
    out_csv = Path(args.output)

    _, rec_psnr, rec_ssim = evaluate_dir(recovered_dir, gt_dir)
    _, cur_psnr, cur_ssim = evaluate_dir(current_dir, gt_dir)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mean_psnr", "mean_ssim"])
        writer.writerow(["recovered", f"{rec_psnr:.4f}", f"{rec_ssim:.4f}"])
        writer.writerow(["current", f"{cur_psnr:.4f}", f"{cur_ssim:.4f}"])
        writer.writerow(
            [
                "delta_recovered_minus_current",
                f"{(rec_psnr - cur_psnr):+.4f}",
                f"{(rec_ssim - cur_ssim):+.4f}",
            ]
        )

    print("===== Recovered vs Current =====")
    print(f"Recovered: PSNR={rec_psnr:.4f}, SSIM={rec_ssim:.4f}")
    print(f"Current  : PSNR={cur_psnr:.4f}, SSIM={cur_ssim:.4f}")
    print(f"Delta    : PSNR={(rec_psnr - cur_psnr):+.4f}, SSIM={(rec_ssim - cur_ssim):+.4f}")
    print(f"CSV saved: {out_csv}")
    if not HAS_SKIMAGE:
        print("[INFO] skimage unavailable, SSIM may be NaN.")


if __name__ == "__main__":
    main()
