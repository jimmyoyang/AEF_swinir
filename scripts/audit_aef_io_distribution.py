#!/usr/bin/env python3
"""Audit model input/output distributions for Landsat -> AlphaEarth training.

This script is intentionally small and empirical: it builds the configured
dataset, samples a few items, and reports what the model actually receives.
For AlphaEarth HR files exported with quantize_aef(), it also reports the raw
uint8 target and the dequantized embedding distribution.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datapipe.datasets import (  # noqa: E402
    _l2_normalize_embedding_chw,
    aef_uint8_dequantize,
    create_dataset,
)


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _sample_flat(arr: np.ndarray, max_values: int) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if max_values <= 0 or flat.size <= max_values:
        return flat
    step = max(1, flat.size // max_values)
    return flat[::step][:max_values]


class SampledStats:
    def __init__(self, max_values_per_field: int):
        self.max_values_per_field = int(max_values_per_field)
        self._values: Dict[str, List[np.ndarray]] = {}

    def add(self, name: str, arr, max_values_per_sample: int) -> None:
        values = _sample_flat(_to_numpy(arr), max_values=max_values_per_sample)
        if values.size == 0:
            return
        bucket = self._values.setdefault(name, [])
        current = sum(v.size for v in bucket)
        remaining = self.max_values_per_field - current
        if remaining <= 0:
            return
        bucket.append(values[:remaining])

    def summarize(self) -> List[Dict[str, float]]:
        rows = []
        for name, chunks in sorted(self._values.items()):
            if not chunks:
                continue
            values = np.concatenate(chunks, axis=0)
            if values.size == 0:
                continue
            rows.append(
                {
                    "name": name,
                    "n": int(values.size),
                    "min": float(np.min(values)),
                    "p01": float(np.percentile(values, 1)),
                    "p50": float(np.percentile(values, 50)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "p99": float(np.percentile(values, 99)),
                    "max": float(np.max(values)),
                }
            )
        return rows


def _as_sample_path(value) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def _choose_indices(length: int, max_samples: int, seed: int, random_sample: bool) -> List[int]:
    n = min(int(max_samples), int(length))
    if n <= 0:
        return []
    if not random_sample:
        return list(range(n))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(length, size=n, replace=False))


def _set_need_path(cfg, split: str) -> None:
    if not hasattr(cfg.data, split):
        raise KeyError(f"Config has no data.{split}")
    params = cfg.data[split].setdefault("params", {})
    params["need_path"] = True


def audit(args: argparse.Namespace) -> List[Dict[str, float]]:
    cfg = OmegaConf.load(args.cfg_path)
    _set_need_path(cfg, args.split)

    data_cfg = cfg.data[args.split]
    ds = create_dataset(data_cfg, parent_configs=cfg)
    indices = _choose_indices(len(ds), args.max_samples, args.seed, args.random)

    stats = SampledStats(max_values_per_field=args.max_values_per_field)
    base_band_count = int(args.base_band_count)
    if base_band_count <= 0:
        try:
            base_band_count = int(data_cfg.params.get("reflectance_band_count", 9) or 9)
        except Exception:
            base_band_count = 9

    for idx in indices:
        sample = ds[idx]
        lr = sample["lr_sequence"]
        gt = sample["gt"]

        stats.add("dataset.lr_all", lr, args.max_values_per_sample)
        stats.add("dataset.lr_base_reflectance", lr[:, :base_band_count], args.max_values_per_sample)
        if lr.shape[1] > base_band_count:
            stats.add("dataset.lr_extra_bands", lr[:, base_band_count:], args.max_values_per_sample)
        stats.add("dataset.gt", gt, args.max_values_per_sample)

        gt_np = _to_numpy(gt)
        stats.add("dataset.gt_pixel_l2_norm", np.linalg.norm(gt_np, axis=0), args.max_values_per_sample)

        if "mask" in sample:
            stats.add("dataset.temporal_mask", sample["mask"], args.max_values_per_sample)
        if "indicating_mask" in sample:
            stats.add("dataset.indicating_mask", sample["indicating_mask"], args.max_values_per_sample)

        hr_path = _as_sample_path(sample.get("path"))
        if hr_path is not None and args.audit_raw_hr:
            with rasterio.open(hr_path) as src:
                raw_hr = src.read().astype(np.float32)
            deq = aef_uint8_dequantize(raw_hr, power=args.dequantize_power)
            deq_l2 = _l2_normalize_embedding_chw(deq, eps=args.eps)

            stats.add("raw_hr.uint8_values", raw_hr, args.max_values_per_sample)
            stats.add("raw_hr.dequantized", deq, args.max_values_per_sample)
            stats.add("raw_hr.dequantized_pixel_l2_norm", np.linalg.norm(deq, axis=0), args.max_values_per_sample)
            stats.add("raw_hr.dequantized_l2", deq_l2, args.max_values_per_sample)
            stats.add(
                "raw_hr.dequantized_l2_pixel_l2_norm",
                np.linalg.norm(deq_l2, axis=0),
                args.max_values_per_sample,
            )

    return stats.summarize()


def print_rows(rows: Iterable[Dict[str, float]]) -> None:
    print(
        f"{'name':42s} {'n':>9s} {'min':>10s} {'p01':>10s} {'p50':>10s} "
        f"{'mean':>10s} {'std':>10s} {'p99':>10s} {'max':>10s}"
    )
    for r in rows:
        print(
            f"{r['name']:42s} {int(r['n']):9d} "
            f"{r['min']:10.5f} {r['p01']:10.5f} {r['p50']:10.5f} "
            f"{r['mean']:10.5f} {r['std']:10.5f} {r['p99']:10.5f} {r['max']:10.5f}"
        )


def write_csv(rows: Iterable[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AEF LR/HR distribution after dataset transforms.")
    parser.add_argument("--cfg_path", default="configs/ablation/aef_time_aligned_cosine_default.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random", action="store_true", help="Sample random dataset indices instead of first N.")
    parser.add_argument("--base_band_count", type=int, default=0)
    parser.add_argument("--max_values_per_sample", type=int, default=200000)
    parser.add_argument("--max_values_per_field", type=int, default=2000000)
    parser.add_argument("--audit_raw_hr", action="store_true", default=True)
    parser.add_argument("--no_audit_raw_hr", dest="audit_raw_hr", action="store_false")
    parser.add_argument("--dequantize_power", type=float, default=2.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()

    rows = audit(args)
    print_rows(rows)
    if args.out_csv:
        write_csv(rows, Path(args.out_csv))
        print(f"[audit] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
