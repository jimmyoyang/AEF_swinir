#!/usr/bin/env python3
"""
Filter processed dataset by exact temporal length T per tile.

For each split (train/val/test):
1) Group files by tile_id parsed from "<date>_tile_<r>_<c>.tif"
2) Keep tile_ids whose LR file count is exactly --target-t
3) Keep only LR/HR matched names under those tile_ids
4) Write filtered files to output root, preserving split/LR and split/HR layout

Default paths are aligned with this project:
  input:  data/processed_data_SR_10m_32samples
  output: data/processed_data_SR_10m_filer
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path


FILE_RE = re.compile(r"^(\d{8})_tile_(\d+_\d+)\.tif$")


def parse_name(name: str):
    m = FILE_RE.match(name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def collect_split(split_dir: Path):
    lr_dir = split_dir / "LR"
    hr_dir = split_dir / "HR"
    if not lr_dir.is_dir() or not hr_dir.is_dir():
        raise FileNotFoundError(f"Missing LR/HR directory in split: {split_dir}")

    lr_by_tile = defaultdict(list)
    hr_names = set()
    bad_lr = 0
    bad_hr = 0

    for p in lr_dir.glob("*.tif"):
        _, tile_id = parse_name(p.name)
        if tile_id is None:
            bad_lr += 1
            continue
        lr_by_tile[tile_id].append(p)

    for p in hr_dir.glob("*.tif"):
        _, tile_id = parse_name(p.name)
        if tile_id is None:
            bad_hr += 1
            continue
        hr_names.add(p.name)

    for tile_id in lr_by_tile:
        lr_by_tile[tile_id].sort(key=lambda x: x.name)

    return lr_by_tile, hr_names, bad_lr, bad_hr


def filter_split(
    input_root: Path,
    output_root: Path,
    split: str,
    target_t: int,
    mode: str,
    clear_output_split: bool,
    dry_run: bool,
):
    split_in = input_root / split
    split_out = output_root / split
    out_lr = split_out / "LR"
    out_hr = split_out / "HR"

    if not dry_run:
        if clear_output_split and split_out.exists():
            shutil.rmtree(split_out)
        safe_mkdir(out_lr)
        safe_mkdir(out_hr)

    lr_by_tile, hr_names, bad_lr, bad_hr = collect_split(split_in)
    total_tiles = len(lr_by_tile)
    total_lr_files = sum(len(v) for v in lr_by_tile.values())

    kept_tile_ids = []
    kept_lr_files = []
    missing_hr_count = 0

    for tile_id, lr_files in lr_by_tile.items():
        if len(lr_files) != target_t:
            continue
        matched = []
        for p in lr_files:
            if p.name in hr_names:
                matched.append(p)
            else:
                missing_hr_count += 1
        if len(matched) != target_t:
            continue
        kept_tile_ids.append(tile_id)
        kept_lr_files.extend(matched)

    kept_lr_files.sort(key=lambda x: x.name)

    copied = 0
    for lr_src in kept_lr_files:
        hr_src = split_in / "HR" / lr_src.name
        if not dry_run:
            lr_dst = out_lr / lr_src.name
            hr_dst = out_hr / lr_src.name
            copy_or_link(lr_src, lr_dst, mode)
            copy_or_link(hr_src, hr_dst, mode)
        copied += 1

    return {
        "split": split,
        "target_t": target_t,
        "total_tiles": total_tiles,
        "total_lr_files": total_lr_files,
        "kept_tiles": len(kept_tile_ids),
        "kept_pairs": copied,  # LR/HR pairs
        "bad_lr_names": bad_lr,
        "bad_hr_names": bad_hr,
        "missing_hr_count": missing_hr_count,
        "output_lr_dir": out_lr,
        "output_hr_dir": out_hr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter processed dataset by exact T per tile and export a new dataset root."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/processed_data_SR_10m_32samples"),
        help="Source processed dataset root (contains train/val/test).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed_data_SR_10m_filer"),
        help="Destination root for filtered dataset.",
    )
    parser.add_argument(
        "--target-t",
        type=int,
        default=20,
        help="Keep tile_ids with exactly this number of LR timesteps.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to process.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to write filtered files.",
    )
    parser.add_argument(
        "--clear-output-split",
        action="store_true",
        help="Remove output split directory before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count remaining tiles/files; do not write output files.",
    )
    args = parser.parse_args()

    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    if args.target_t <= 0:
        raise ValueError("--target-t must be positive")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No valid split in --splits")

    safe_mkdir(args.output_root)

    print(f"[INFO] input_root : {args.input_root}")
    print(f"[INFO] output_root: {args.output_root}")
    print(f"[INFO] target_t   : {args.target_t}")
    print(f"[INFO] splits     : {splits}")
    print(f"[INFO] mode       : {args.mode}")
    print(f"[INFO] dry_run    : {args.dry_run}")

    reports = []
    for split in splits:
        report = filter_split(
            input_root=args.input_root,
            output_root=args.output_root,
            split=split,
            target_t=args.target_t,
            mode=args.mode,
            clear_output_split=args.clear_output_split,
            dry_run=args.dry_run,
        )
        reports.append(report)
        print(
            f"[{split}] tiles: {report['total_tiles']} -> {report['kept_tiles']} | "
            f"pairs kept: {report['kept_pairs']} | "
            f"bad names (LR/HR): {report['bad_lr_names']}/{report['bad_hr_names']} | "
            f"missing_hr_hits: {report['missing_hr_count']}"
        )
        print(f"        output LR: {report['output_lr_dir']}")
        print(f"        output HR: {report['output_hr_dir']}")

    total_pairs = sum(r["kept_pairs"] for r in reports)
    total_tiles = sum(r["kept_tiles"] for r in reports)
    print("\n[SUMMARY]")
    print(f"  total kept tiles: {total_tiles}")
    print(f"  total kept LR/HR pairs: {total_pairs}")


if __name__ == "__main__":
    main()

