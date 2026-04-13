#!/usr/bin/env python3
"""
调试 LR/HR GeoTIFF：统计各波段、与训练一致的 robust_per_image_normalize 后是否含 NaN/Inf。
用于排查训练 loss=NaN（常见原因：原始数据含 NaN、掩膜全零仍算 SSIM、常数波段等）。

示例:
  python test_lr_tile_values.py /path/to/a.tif
  python test_lr_tile_values.py --lr-dir /path/to/train/LR --limit 20
  python test_lr_tile_values.py --lr-dir data/.../LR --hr-dir data/.../HR --limit 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datapipe.datasets import robust_per_image_normalize


def describe_tif(path: Path, norm: bool = False) -> None:
    print("=" * 72)
    print(f"path: {path}")
    if not path.is_file():
        print("  [MISSING] file does not exist")
        return

    with rasterio.open(path) as src:
        arr = src.read().astype(np.float64)
        print(f"  shape (C,H,W): {arr.shape}")
        print(f"  dtype (on disk): {src.dtypes}")
        print(f"  nodata: {src.nodatavals}")

    if norm:
        arr_norm = robust_per_image_normalize(arr.astype(np.float32))
        finite = np.isfinite(arr_norm)
        n_bad = int((~finite).sum())
        print(f"  [after robust_per_image_normalize] non-finite count: {n_bad} / {arr_norm.size}")
        if n_bad:
            print("  -> 该文件在数据集归一化后会产生 NaN/Inf，易导致 MixedLoss/SSIM 为 NaN")
        per_ch = []
        for c in range(arr_norm.shape[0]):
            sl = arr_norm[c].ravel()
            f = np.isfinite(sl)
            if f.any():
                s = sl[f]
                per_ch.append((c, float(np.min(s)), float(np.max(s)), float(np.mean(s))))
            else:
                per_ch.append((c, float("nan"), float("nan"), float("nan")))
        print("  per-channel (finite only) min/max/mean:", per_ch[: min(6, len(per_ch))], "...")

    flat = arr.reshape(arr.shape[0], -1)
    for c in range(arr.shape[0]):
        band = flat[c]
        finite = np.isfinite(band)
        n_nan = int(np.isnan(band).sum())
        n_inf = int(np.isinf(band).sum())
        if finite.any():
            b = band[finite]
            print(
                f"  band {c:2d}: min={np.min(b):.8g} max={np.max(b):.8g} "
                f"mean={np.mean(b):.8g} std={np.std(b):.8g} | "
                f"nan={n_nan} inf={n_inf}"
            )
        else:
            print(f"  band {c:2d}: no finite values | nan={n_nan} inf={n_inf}")

    h, w = arr.shape[1], arr.shape[2]
    h0, w0 = min(4, h), min(4, w)
    print(f"  sample band0[:{h0}, :{w0}] (top-left):")
    sample = arr[0, :h0, :w0]
    for row in sample:
        print("   ", " ".join(f"{v:12.6g}" for v in row))


def collect_tifs(lr_dir: Path, limit: int, pattern: str) -> list[Path]:
    files = sorted(lr_dir.glob(pattern))
    if limit > 0:
        files = files[:limit]
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect LR/HR GeoTIFF stats and dataset-normalized NaN risk (same as training)."
    )
    parser.add_argument("paths", nargs="*", type=str, help="Optional list of .tif paths")
    parser.add_argument("--lr-dir", type=str, default=None, help="Scan this LR directory (e.g. train/LR)")
    parser.add_argument("--hr-dir", type=str, default=None, help="Also scan HR directory")
    parser.add_argument("--limit", type=int, default=32, help="Max files per directory when using --lr-dir/--hr-dir (0=all)")
    parser.add_argument("--glob", type=str, default="*.tif", help="Glob under lr/hr dir")
    parser.add_argument("--no-norm-pass", action="store_true", help="Skip second pass with robust_per_image_normalize")
    args = parser.parse_args()

    paths: list[Path] = [Path(p) for p in args.paths]

    if args.lr_dir:
        paths.extend(collect_tifs(Path(args.lr_dir), args.limit, args.glob))
    if args.hr_dir:
        paths.extend(collect_tifs(Path(args.hr_dir), args.limit, args.glob))

    if not paths:
        paths = [
            REPO_ROOT / "data/Cloud_test/processed_data_SR_10m/train/LR/20180101_tile_0_1.tif",
            REPO_ROOT / "data/processed_data/train/LR/20180101_tile_1_29.tif",
        ]
        print("[info] No paths given; using defaults (may be missing):", file=sys.stderr)

    seen = set()
    unique = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    bad_after_norm = 0
    for p in unique:
        describe_tif(p, norm=False)
        if not args.no_norm_pass:
            describe_tif(p, norm=True)
            with rasterio.open(p) as src:
                arr = src.read().astype(np.float32)
            out = robust_per_image_normalize(arr)
            if not np.isfinite(out).all():
                bad_after_norm += 1

    print("=" * 72)
    print(f"Summary: {len(unique)} file(s), {bad_after_norm} with non-finite values after dataset normalization.")
    return 1 if bad_after_norm else 0


if __name__ == "__main__":
    raise SystemExit(main())
