#!/usr/bin/env python3
"""
验证 raw_landsat / raw_alphaearth 与 prepare_data_local 产物是否一致，并辅助定位训练侧
「variable T + default collate」类错误。

1) 几何 / HR 重建 / 静态 HR（原有）
2) --audit-temporal：按 tile_id 统计每个瓦片的 LR 时序条数 T 的分布；可选 pair-zero manifest
   模拟 AnytimeTemporalDataset 过滤后的 T。用于解释：
   RuntimeError: stack expects each tensor to be equal size, but got [T1,C,H,W] and [T2,C,H,W]
3) --smoke-dataloader：按 YAML 实例化 Dataset + DataLoader(batch>1)，直接复现 collate 是否失败

说明：
- datapipe.datasets.AnytimeTemporalDataset 不在 __getitem__ 里 pad 时间维；
  若 train.batch[0]>1 且不同 tile 的 T 不同，默认 collate 会报错。
- datapipe.datasets_with_lr_norm.AnytimeTemporalDataset 会对齐 fixed_temporal_len（默认 20）。

依赖: numpy, rasterio；--smoke-dataloader 另需 torch, pyyaml。
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_processed_name(name: str):
    m = re.match(r"^(\d{8})_tile_(\d+)_(\d+)\.tif$", name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def landsat_path_for_date(raw_landsat: Path, path_row: str, yyyymmdd: str) -> Path | None:
    pat = f"L8_{path_row}_{yyyymmdd}_Masked.tif"
    p = raw_landsat / pat
    return p if p.is_file() else None


def alpha_paths(raw_alpha: Path, path: int, row: int, num_bands: int) -> list[Path]:
    paths = []
    for i in range(num_bands):
        b = f"A{i:02d}"
        g = list(raw_alpha.glob(f"AlphaEarth_Path{path}_Row{row}_reprojected_{b}.tif"))
        if not g:
            raise FileNotFoundError(f"Missing HR band file for {b}")
        paths.append(g[0])
    return paths


def proc_overlap_dims(lr_h: int, lr_w: int, hr_h: int, hr_w: int, scale: int) -> tuple[int, int]:
    """与 prepare_data_local.py 一致。"""
    proc_h = min(lr_h, hr_h // scale)
    proc_w = min(lr_w, hr_w // scale)
    return proc_h, proc_w


def reconstruct_hr_from_raw(
    alpha_band_paths: list[Path],
    grid_i: int,
    grid_j: int,
    lr_patch: int,
    scale: int,
    hr_patch: int,
) -> np.ndarray:
    """与 create_one_tile 中 HR 读取方式一致：像素 r,c = grid*64，HR 窗口从 hr_c, hr_r 起。"""
    r = grid_i * lr_patch
    c = grid_j * lr_patch
    hr_r, hr_c = r * scale, c * scale
    win = Window(hr_c, hr_r, hr_patch, hr_patch)
    out = np.zeros((len(alpha_band_paths), hr_patch, hr_patch), dtype=np.float32)
    for i, bp in enumerate(alpha_band_paths):
        with rasterio.open(bp) as src:
            out[i] = src.read(1, window=win).astype(np.float32)
    return out


def _tile_id_from_lr_filename(name: str) -> str | None:
    m = re.search(r"tile_(\d+_\d+)\.tif$", name)
    return m.group(1) if m else None


def scan_lr_paths_by_tile(lr_dir: Path) -> dict[str, list[Path]]:
    """单次扫描 LR 目录，按 tile_id 分组（与 AnytimeTemporalDataset 索引逻辑一致）。"""
    by_tile: dict[str, list[Path]] = defaultdict(list)
    for p in lr_dir.glob("*_tile_*.tif"):
        if not p.is_file():
            continue
        tid = _tile_id_from_lr_filename(p.name)
        if tid is None:
            continue
        by_tile[tid].append(p)
    for tid in by_tile:
        by_tile[tid].sort(key=lambda x: x.name)
    return by_tile


def summarize_t_distribution(counts: dict[str, int]) -> dict:
    if not counts:
        return {"n_tiles": 0}
    vals = list(counts.values())
    hist = Counter(vals)
    return {
        "n_tiles": len(vals),
        "T_min": min(vals),
        "T_max": max(vals),
        "T_mean": float(np.mean(vals)),
        "T_median": float(np.median(vals)),
        "T_std": float(np.std(vals)),
        "hist_top": hist.most_common(12),
        "variable_T": min(vals) != max(vals),
    }


def print_temporal_audit_section(title: str, raw_by_tile: dict[str, int] | None, filt_by_tile: dict[str, int] | None):
    print(f"\n=== {title} ===")
    if raw_by_tile is not None:
        s = summarize_t_distribution(raw_by_tile)
        print(f"[raw file counts per tile] tiles={s['n_tiles']} T_min={s['T_min']} T_max={s['T_max']} "
              f"T_mean={s['T_mean']:.2f} T_median={s['T_median']:.1f} std={s['T_std']:.2f}")
        print(f"  T histogram (value -> count): {s['hist_top']}")
        if s.get("variable_T"):
            print("  -> 不同 tile 的时序长度 T 不一致。")
        else:
            print("  -> 所有 tile 的 T 相同。")

    if filt_by_tile is not None:
        s2 = summarize_t_distribution(filt_by_tile)
        print(f"[after pair-zero filter sim] tiles={s2['n_tiles']} T_min={s2['T_min']} T_max={s2['T_max']} "
              f"T_mean={s2['T_mean']:.2f} T_median={s2['T_median']:.1f}")
        print(f"  T histogram: {s2['hist_top']}")
        if s2.get("variable_T"):
            print("  -> 过滤后 T 仍随 tile 变化：默认 DataLoader collate + batch_size>1 会 stack 失败。")

    print("\n[训练报错定位]")
    print("  若出现: RuntimeError: stack expects each tensor to be equal size, but got [T1,C,H,W] and [T2,...]")
    print("  根因通常是: datapipe.datasets.AnytimeTemporalDataset 返回的 lr_sequence 第一维 T 随样本变化，")
    print("  而 torch.utils.data.DataLoader 默认 collate 会把 batch 维 stack 起来，要求各样本 T 相同。")
    print("  可行修复（选一）:")
    print("  - train/val 的 batch 设为 1；或实现自定义 collate_fn（pad/truncate 到统一 T）。")
    print("  - 换用 datapipe.datasets_with_lr_norm.AnytimeTemporalDataset（__getitem__ 内 _pad_or_truncate_temporal_sample）。")
    print("  - 在 datapipe.datasets.AnytimeTemporalDataset 返回前同样 pad/truncate（与 lr_norm 版对齐）。")


def run_temporal_audit(
    processed_root: Path,
    processed_subdir: str,
    splits: list[str],
    manifest: Path | None,
    threshold: float,
):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        from datapipe.datasets import _filter_files_by_hr_zero_ratio, _load_hr_zero_ratio_manifest
    except ImportError as e:
        print(f"ERROR: cannot import datapipe.datasets for manifest simulation: {e}", file=sys.stderr)
        print("  Run from repo root or set PYTHONPATH.", file=sys.stderr)
        sys.exit(1)

    zr_map = None
    if manifest is not None:
        if not manifest.exists():
            print(f"[WARN] manifest not found: {manifest} (pair-zero simulation skipped)", file=sys.stderr)
        else:
            zr_map = _load_hr_zero_ratio_manifest(manifest)

    for split in splits:
        lr_dir = processed_root / processed_subdir / split / "LR"
        if not lr_dir.is_dir():
            print(f"[skip] {lr_dir} not found")
            continue
        print(f"\n[scan] {lr_dir} ...")
        by_tile = scan_lr_paths_by_tile(lr_dir)
        raw_counts = {tid: len(paths) for tid, paths in by_tile.items()}
        filt_counts: dict[str, int] | None = None
        if zr_map is not None:
            filt_counts = {}
            for tid, paths in by_tile.items():
                kept, _, _ = _filter_files_by_hr_zero_ratio(paths, zr_map, threshold)
                filt_counts[tid] = len(kept)

        print_temporal_audit_section(f"Temporal audit: {split}", raw_counts, filt_counts)


def _probe_dataset_temporal_lengths(dataset, probe_samples: int):
    """直接逐样本读取，输出 T 分布和样本元信息（不经过 DataLoader collate）。"""
    n = min(int(probe_samples), len(dataset))
    rows = []
    t_hist = Counter()
    for i in range(n):
        sample = dataset[i]
        lr_seq = sample.get("lr_sequence")
        if lr_seq is None:
            continue
        t_len = int(lr_seq.shape[0])
        t_hist[t_len] += 1
        tile_id = None
        if hasattr(dataset, "tile_ids") and i < len(dataset.tile_ids):
            tile_id = dataset.tile_ids[i]
        path = sample.get("path")
        rows.append(
            {
                "index": i,
                "tile_id": tile_id,
                "T": t_len,
                "path": str(path) if path is not None else "",
            }
        )
    return rows, t_hist


def run_smoke_dataloader(
    config_path: Path,
    split: str,
    batch_size: int,
    disable_zero_filter: bool,
    probe_samples: int,
):
    try:
        import torch
        import yaml
    except ImportError as e:
        print(f"ERROR: --smoke-dataloader needs torch and yaml: {e}", file=sys.stderr)
        sys.exit(1)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from datapipe.datasets import create_dataset

    with Path(config_path).open("r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    if split not in configs.get("data", {}):
        print(f"ERROR: data.{split} missing in config", file=sys.stderr)
        sys.exit(1)

    ds_cfg = dict(configs["data"][split])
    params = dict(ds_cfg.get("params", {}))
    if disable_zero_filter:
        params["enable_hr_zero_filter"] = False
    # 便于在 smoke 阶段打印更明确的样本定位信息
    params["need_path"] = True
    ds_cfg["params"] = params

    print(f"\n=== Smoke DataLoader (config={config_path}, split={split}, batch_size={batch_size}) ===")
    print(f"  dataset target: {ds_cfg.get('target')}")
    if disable_zero_filter:
        print("  (overrides enable_hr_zero_filter=False for faster init)")

    dataset = create_dataset(ds_cfg, parent_configs=configs)
    print(f"  len(dataset)={len(dataset)}")
    if probe_samples > 0:
        rows, t_hist = _probe_dataset_temporal_lengths(dataset, probe_samples)
        if rows:
            t_values = [r["T"] for r in rows]
            print(
                f"  pre-collate probe: n={len(rows)} T_min={min(t_values)} T_max={max(t_values)} "
                f"T_mean={float(np.mean(t_values)):.2f} hist={t_hist.most_common(8)}"
            )
            if min(t_values) != max(t_values):
                print("  -> 采样范围内已出现 variable-T，batch_size>1 时默认 collate 高概率报错。")
        else:
            print("  [WARN] pre-collate probe did not collect lr_sequence info.")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    try:
        batch = next(iter(loader))
        print("  OK: first batch collated successfully.")
        for k, v in batch.items():
            if torch.is_tensor(v):
                print(f"    {k}: {tuple(v.shape)}")
    except RuntimeError as e:
        print(f"  FAIL: {e}")
        if len(dataset) > 0:
            print("  failing batch candidates (first batch, shuffle=False):")
            for i in range(min(batch_size, len(dataset))):
                s = dataset[i]
                t_len = int(s["lr_sequence"].shape[0]) if "lr_sequence" in s else -1
                tile_id = dataset.tile_ids[i] if hasattr(dataset, "tile_ids") and i < len(dataset.tile_ids) else "?"
                p = s.get("path", "")
                print(f"    entry{i}: idx={i} tile_id={tile_id} T={t_len} path={p}")
        print("  这与 variable-T + default collate 一致；见上方 --audit-temporal 建议。")
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser(
        description="Validate raw vs processed tiles + optional temporal/DataLoader diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-root", type=Path, default=None, help="含 raw_landsat / raw_alphaearth / processed_*")
    ap.add_argument("--processed-subdir", type=str, default="processed_data_SR_10m_32samples")
    ap.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    ap.add_argument("--path", type=int, default=132)
    ap.add_argument("--row", type=int, default=33)
    ap.add_argument("--path-row", type=str, default="", help="L8 文件名 pathrow；默认 path+row")
    ap.add_argument("--num-hr-bands", type=int, default=64)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--lr-patch", type=int, default=64)
    ap.add_argument("--hr-patch", type=int, default=192)
    ap.add_argument("--check-static-hr", action="store_true")
    ap.add_argument("--reconstruct-hr", action="store_true")
    ap.add_argument("--one-name", type=str, default="", help="指定 processed LR 文件名；默认取目录中第一个")
    ap.add_argument("--skip-geometry", action="store_true", help="跳过 raw/单 patch 几何检查（仅跑 temporal/smoke）")
    ap.add_argument("--audit-temporal", action="store_true", help="按 tile 统计时序长度 T，解释 variable-T collate 报错")
    ap.add_argument("--all-splits", action="store_true", help="与 --audit-temporal 联用：扫描 train/val/test")
    ap.add_argument("--pair-zero-manifest", type=Path, default=None, help="patch_stats.log，模拟 enable_hr_zero_filter 后的 T")
    ap.add_argument("--pair-zero-threshold", type=float, default=0.05)
    ap.add_argument("--smoke-dataloader", type=Path, default=None, help="YAML 配置路径：DataLoader batch>1 冒烟测试")
    ap.add_argument("--smoke-split", type=str, default="train", choices=("train", "val", "test"))
    ap.add_argument("--smoke-batch-size", type=int, default=2)
    ap.add_argument("--smoke-probe-samples", type=int, default=16, help="smoke 前先直接读取前 N 个样本统计 T 分布")
    ap.add_argument("--smoke-no-zero-filter", action="store_true", help="冒烟测试时关闭 HR zero filter 以加速")
    args = ap.parse_args()

    if (
        args.data_root is None
        and not args.skip_geometry
        and not args.audit_temporal
        and args.smoke_dataloader is None
    ):
        ap.error("需要 --data-root，或使用 --skip-geometry + --audit-temporal/--smoke-dataloader 等组合")

    if args.skip_geometry:
        if args.audit_temporal:
            if args.data_root is None:
                ap.error("--audit-temporal 需要 --data-root")
            splits = ["train", "val", "test"] if args.all_splits else [args.split]
            run_temporal_audit(
                args.data_root,
                args.processed_subdir,
                splits,
                args.pair_zero_manifest,
                args.pair_zero_threshold,
            )
        if args.smoke_dataloader is not None:
            run_smoke_dataloader(
                args.smoke_dataloader,
                args.smoke_split,
                args.smoke_batch_size,
                disable_zero_filter=args.smoke_no_zero_filter,
                probe_samples=args.smoke_probe_samples,
            )
        print("\nDone (--skip-geometry).")
        return

    if args.data_root is None:
        if args.smoke_dataloader is not None:
            run_smoke_dataloader(
                args.smoke_dataloader,
                args.smoke_split,
                args.smoke_batch_size,
                disable_zero_filter=args.smoke_no_zero_filter,
                probe_samples=args.smoke_probe_samples,
            )
        if args.audit_temporal:
            ap.error("--audit-temporal 需要 --data-root（或配合 --skip-geometry 仅跑 temporal）")
        print("\nDone.")
        return

    if args.audit_temporal:
        splits = ["train", "val", "test"] if args.all_splits else [args.split]
        run_temporal_audit(
            args.data_root,
            args.processed_subdir,
            splits,
            args.pair_zero_manifest,
            args.pair_zero_threshold,
        )

    if args.smoke_dataloader is not None:
        run_smoke_dataloader(
            args.smoke_dataloader,
            args.smoke_split,
            args.smoke_batch_size,
            disable_zero_filter=args.smoke_no_zero_filter,
            probe_samples=args.smoke_probe_samples,
        )

    path_row = args.path_row or f"{args.path}{args.row}"
    raw_lr_dir = args.data_root / "raw_landsat"
    raw_hr_dir = args.data_root / "raw_alphaearth"
    proc_lr_dir = args.data_root / args.processed_subdir / args.split / "LR"
    proc_hr_dir = args.data_root / args.processed_subdir / args.split / "HR"

    if not proc_lr_dir.is_dir():
        print(f"ERROR: {proc_lr_dir} not found", file=sys.stderr)
        sys.exit(1)

    print("=== Raw geometry & overlap (prepare_data_local 一致) ===")
    lr_files = sorted(raw_lr_dir.glob(f"L8_{path_row}_*_Masked.tif"))
    if not lr_files:
        print(f"ERROR: no Landsat L8_{path_row}_*_Masked.tif under {raw_lr_dir}", file=sys.stderr)
        sys.exit(1)
    sample_lr = lr_files[0]
    alpha_list = alpha_paths(raw_hr_dir, args.path, args.row, args.num_hr_bands)

    with rasterio.open(sample_lr) as lr:
        lr_h, lr_w = lr.height, lr.width
        lr_crs = lr.crs
        lr_tf = lr.transform
    with rasterio.open(alpha_list[0]) as a0:
        hr_h, hr_w = a0.height, a0.width
        hr_crs = a0.crs
        hr_tf = a0.transform

    ph, pw = proc_overlap_dims(lr_h, lr_w, hr_h, hr_w, args.scale)
    print(f"Landsat sample: {sample_lr.name} size {lr_h}x{lr_w}")
    print(f"AlphaEarth A00: size {hr_h}x{hr_w}")
    print(f"proc_h, proc_w = min(lr, hr//scale) -> {ph}, {pw}")
    print(f"LR CRS: {lr_crs} | HR CRS: {hr_crs}")
    if lr_crs != hr_crs:
        print("WARN: CRS differs; prepare assumes aligned grids (same CRS / same grid).")
    print(f"LR transform: {lr_tf}")
    print(f"HR transform: {hr_tf}")

    if args.one_name:
        one = args.one_name
        if not (proc_lr_dir / one).is_file():
            print(f"ERROR: {proc_lr_dir / one} missing", file=sys.stderr)
            sys.exit(1)
    else:
        found = next(proc_lr_dir.glob("*_tile_*_*.tif"), None)
        if found is None:
            print(f"ERROR: no tif under {proc_lr_dir}", file=sys.stderr)
            sys.exit(1)
        one = found.name

    parsed = parse_processed_name(one)
    if not parsed:
        print(f"ERROR: cannot parse processed name: {one}", file=sys.stderr)
        sys.exit(1)
    date_str, gi, gj = parsed
    print(f"\n=== Processed sample: {args.split}/LR/{one} ===")

    lr_p = proc_lr_dir / one
    hr_p = proc_hr_dir / one
    if not hr_p.is_file():
        print(f"ERROR: missing HR {hr_p}", file=sys.stderr)
        sys.exit(1)

    with rasterio.open(lr_p) as ds:
        plr = ds.read()
        print(f"Processed LR shape {plr.shape} dtype {plr.dtype} range [{np.nanmin(plr):.6g}, {np.nanmax(plr):.6g}]")
    with rasterio.open(hr_p) as ds:
        phr = ds.read()
        print(f"Processed HR shape {phr.shape} dtype {phr.dtype} range [{np.nanmin(phr):.6g}, {np.nanmax(phr):.6g}]")

    if plr.ndim != 3 or phr.ndim != 3:
        print("ERROR: expected CHW arrays", file=sys.stderr)
        sys.exit(1)
    sh = phr.shape[1] / plr.shape[1]
    sw = phr.shape[2] / plr.shape[2]
    if abs(sh - args.scale) > 1e-6 or abs(sw - args.scale) > 1e-6:
        print(f"WARN: HR/LR spatial scale not {args.scale}: got h={sh}, w={sw}")

    raw_lp = landsat_path_for_date(raw_lr_dir, path_row, date_str)
    if raw_lp:
        print(f"Matching raw Landsat: {raw_lp.name} exists.")
    else:
        print(f"WARN: no raw Landsat for date {date_str} pathrow {path_row}")

    if args.reconstruct_hr:
        print("\n=== Reconstruct HR from raw AlphaEarth (same window as prepare) ===")
        recon = reconstruct_hr_from_raw(
            alpha_list, gi, gj, args.lr_patch, args.scale, args.hr_patch
        )
        phr_f = phr.astype(np.float32)
        d = np.nanmax(np.abs(recon - phr_f))
        print(f"max abs(processed_HR - recon_from_raw) = {d:.6g}")
        if d < 1e-3:
            print("OK: HR patch matches raw stack (within 1e-3).")
        elif d < 1.0:
            print("OK-ish: small mismatch (dtype/rounding); inspect if worried.")
        else:
            print("WARN: large mismatch — CRS/grid/tile index 或 band 顺序可能有问题。")

    if args.check_static_hr:
        print("\n=== Static HR across dates (same tile_ij) ===")
        tile_suffix = f"_tile_{gi}_{gj}.tif"
        same_tile = sorted(proc_hr_dir.glob(f"*{tile_suffix}"))
        if len(same_tile) < 2:
            print(f"Only {len(same_tile)} HR file(s) for tile_{gi}_{gj}; skip or use more data.")
        else:
            with rasterio.open(same_tile[0]) as ds:
                ref = ds.read()
            maxdiff = 0.0
            for p in same_tile[1:11]:
                with rasterio.open(p) as ds:
                    d = np.nanmax(np.abs(ds.read().astype(np.float32) - ref.astype(np.float32)))
                maxdiff = max(maxdiff, d)
            print(f"Compared first vs up to 10 others for tile_{gi}_{gj}: max abs diff = {maxdiff:.6g}")
            if maxdiff == 0.0:
                print("OK: HR identical across sampled dates (静态 AlphaEarth 预期).")
            else:
                print("INFO: HR varies across dates — 检查是否多套 HR 或窗口不一致。")

    print("\nDone.")


if __name__ == "__main__":
    main()
