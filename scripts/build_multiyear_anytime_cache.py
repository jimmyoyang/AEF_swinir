#!/usr/bin/env python3
import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_anytime_cache import build_one_tile_cache, group_lr_files_by_tile, tile_cache_complete


DEFAULT_DATASETS = {
    "2018": "2018_T20",
    "2019": "2019_T23",
    "2021": "2021_T21",
    "2022": "2022_T19",
    "2023": "2023_T22",
    "2024": "2024_T22",
}


def parse_csv(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [x.strip() for x in str(value).split(",") if x.strip()]


def build_task(task):
    year, split, tile_id, file_paths, out_dir, valid_mask_band_count, reflectance_band_count = task
    _, err = build_one_tile_cache(
        tile_id,
        file_paths,
        out_dir,
        valid_mask_band_count=valid_mask_band_count,
        reflectance_band_count=reflectance_band_count,
    )
    return year, split, tile_id, err


def collect_tasks(args):
    years = parse_csv(args.years)
    splits = parse_csv(args.splits)
    tasks = []
    summary = []

    for year in years:
        dataset_name = DEFAULT_DATASETS.get(year)
        if dataset_name is None:
            raise ValueError(f"Unsupported year '{year}'. Known years: {','.join(DEFAULT_DATASETS)}")

        root = Path(args.data_root) / year / "processed_data_SR_10m_filer"
        for split in splits:
            lr_dir = root / split / "LR"
            out_dir = root / "cache" / split
            if not lr_dir.is_dir():
                print(f"[cache][warn] missing LR dir: year={year} split={split} path={lr_dir}")
                continue

            grouped = group_lr_files_by_tile(lr_dir)
            tile_ids = sorted(grouped.keys())
            if args.max_tiles_per_dataset and args.max_tiles_per_dataset > 0:
                tile_ids = tile_ids[: args.max_tiles_per_dataset]

            skipped = 0
            selected = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            for tile_id in tile_ids:
                if args.skip_existing and tile_cache_complete(out_dir, tile_id):
                    skipped += 1
                    continue
                tasks.append(
                    (
                        year,
                        split,
                        tile_id,
                        [str(p) for p in grouped[tile_id]],
                        str(out_dir),
                        args.valid_mask_band_count,
                        args.reflectance_band_count,
                    )
                )
                selected += 1

            summary.append((year, dataset_name, split, len(tile_ids), skipped, selected, str(out_dir)))

    return tasks, summary


def main():
    parser = argparse.ArgumentParser(description="Build per-year AnytimeTemporalDataset cache for multiple years/splits.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Root containing <year>/processed_data_SR_10m_filer directories.",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2018,2019,2021,2022,2023,2024",
        help="Comma-separated years to process.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated splits to process. train,val is enough for bash train.sh 0 in train mode.",
    )
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max_tiles_per_dataset", type=int, default=0)
    parser.add_argument("--valid_mask_band_count", type=int, default=9)
    parser.add_argument("--reflectance_band_count", type=int, default=9)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    tasks, summary = collect_tasks(args)
    print("[cache] plan")
    for year, name, split, total, skipped, selected, out_dir in summary:
        print(
            f"  - {year} {name} {split}: total_tiles={total} "
            f"skipped_existing={skipped} build={selected} out={out_dir}"
        )
    print(f"[cache] total_build_tasks={len(tasks)} num_workers={args.num_workers}")

    if args.dry_run or not tasks:
        return

    started = time.time()
    failures = []
    done_by_split = Counter()
    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futures = [ex.submit(build_task, task) for task in tasks]
        with tqdm(total=len(futures), desc="build-multiyear-cache", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                year, split, tile_id, err = fut.result()
                if err is not None:
                    failures.append((year, split, tile_id, err))
                    print(f"[cache][warn] year={year} split={split} tile={tile_id} failed: {err}")
                else:
                    done_by_split[(year, split)] += 1
                pbar.update(1)

    elapsed = time.time() - started
    print(f"[cache] finished elapsed_sec={elapsed:.1f} failures={len(failures)}")
    for (year, split), count in sorted(done_by_split.items()):
        print(f"[cache] built year={year} split={split} tiles={count}")
    if failures:
        fail_path = PROJECT_ROOT / "tmp" / f"cache_failures_{int(time.time())}.tsv"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_path.open("w", encoding="utf-8") as f:
            for year, split, tile_id, err in failures:
                f.write(f"{year}\t{split}\t{tile_id}\t{err}\n")
        print(f"[cache] failure_log={fail_path}")


if __name__ == "__main__":
    main()
