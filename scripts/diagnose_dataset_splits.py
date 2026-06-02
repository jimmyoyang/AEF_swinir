#!/usr/bin/env python3
import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datapipe.datasets import create_dataset


def _as_dict_config(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


def _as_list(value):
    if value is None:
        return []
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def _phase_dir_pairs(params):
    roots = _as_list(params.get("dataset_roots", params.get("roots", params.get("data_roots", None))))
    split = params.get("split", None)
    if roots and split:
        return [
            (Path(str(root)) / str(split) / "LR", Path(str(root)) / str(split) / "HR")
            for root in roots
        ]

    lr_dirs = _as_list(params.get("lr_dirs", None))
    hr_dirs = _as_list(params.get("hr_dirs", None))
    if lr_dirs or hr_dirs:
        return [(Path(str(lr)), Path(str(hr))) for lr, hr in zip(lr_dirs, hr_dirs)]

    lr_dir = params.get("lr_dir", None)
    hr_dir = params.get("hr_dir", None)
    if lr_dir is not None and hr_dir is not None:
        return [(Path(str(lr_dir)), Path(str(hr_dir)))]

    return []


def _fmt_dir_pairs(dir_pairs, max_items=3):
    if not dir_pairs:
        return "dirs=n/a"
    preview = [f"{lr.parent.parent.name}:{lr}|{hr}" for lr, hr in dir_pairs[:max_items]]
    suffix = "" if len(dir_pairs) <= max_items else f" ... +{len(dir_pairs) - max_items} more"
    return "; ".join(preview) + suffix


def _tensor_stats(tensor):
    if not torch.is_tensor(tensor):
        return None
    tf = tensor.detach().cpu().float()
    finite = torch.isfinite(tf)
    out = {
        "shape": tuple(tensor.shape),
        "finite_ratio": float(finite.float().mean().item()) if tf.numel() else 0.0,
    }
    if finite.any():
        vals = tf[finite]
        out.update(
            min=float(vals.min().item()),
            mean=float(vals.mean().item()),
            std=float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
            max=float(vals.max().item()),
            zero_ratio=float((vals == 0).float().mean().item()),
        )
    return out


def _fmt_stats(name, stats):
    if not stats:
        return f"{name}: n/a"
    bits = [
        f"{name}: shape={stats['shape']}",
        f"finite={stats['finite_ratio']:.4f}",
    ]
    for key in ("min", "mean", "std", "max", "zero_ratio"):
        if key in stats:
            bits.append(f"{key}={stats[key]:.4f}")
    return " ".join(bits)


def _infer_test_dirs_from_train(cfg, params):
    existing_pairs = _phase_dir_pairs(params)
    if existing_pairs and all(lr.exists() and hr.exists() for lr, hr in existing_pairs):
        return params

    lr_dir = Path(str(params.get("lr_dir", "")))
    hr_dir = Path(str(params.get("hr_dir", "")))
    if lr_dir.exists() and hr_dir.exists():
        return params

    train_params = cfg.data.train.params
    if not hasattr(train_params, "lr_dir") or not hasattr(train_params, "hr_dir"):
        return params

    train_lr = Path(str(train_params.lr_dir))
    train_hr = Path(str(train_params.hr_dir))
    if train_lr.parent.name == "train" and train_hr.parent.name == "train":
        candidate_lr = train_lr.parent.parent / "test" / "LR"
        candidate_hr = train_hr.parent.parent / "test" / "HR"
        if candidate_lr.exists() and candidate_hr.exists():
            params["lr_dir"] = str(candidate_lr)
            params["hr_dir"] = str(candidate_hr)
            print(
                f"[split-diff][INFO] test dirs inferred from train root: "
                f"lr_dir={candidate_lr} hr_dir={candidate_hr}"
            )
    return params


def _make_phase_cfg(cfg, phase):
    if not hasattr(cfg.data, phase):
        return None
    phase_cfg = _as_dict_config(getattr(cfg.data, phase))
    params = dict(phase_cfg.get("params", {}))
    if phase == "test":
        params = _infer_test_dirs_from_train(cfg, params)
    params.setdefault("need_path", True)
    params["sample_cache_mem_size"] = int(params.get("sample_cache_mem_size", 0) or 0)
    params["slow_sample_warn_sec"] = 0
    phase_cfg["params"] = params
    return phase_cfg


def _concat_stats(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _fmt_scalar_stats(name, stats):
    if not stats:
        return f"{name}: n/a"
    return (
        f"{name}: n={stats['n']} min={stats['min']:.4f} mean={stats['mean']:.4f} "
        f"std={stats['std']:.4f} median={stats['median']:.4f} max={stats['max']:.4f}"
    )


def _sample_indices(length, max_samples):
    if length <= 0:
        return []
    count = min(length, max_samples)
    if count == length:
        return list(range(length))
    return sorted(set(int(round(x)) for x in np.linspace(0, length - 1, count)))


def summarize_phase(cfg, phase, max_samples):
    phase_cfg = _make_phase_cfg(cfg, phase)
    if phase_cfg is None:
        print(f"[split-diff][WARN] phase={phase} missing in config")
        return None

    params = phase_cfg.get("params", {})
    dir_pairs = _phase_dir_pairs(params)
    missing_pairs = [(lr, hr) for lr, hr in dir_pairs if not lr.exists() or not hr.exists()]
    if not dir_pairs or missing_pairs:
        if missing_pairs:
            lr_dir, hr_dir = missing_pairs[0]
            missing_msg = (
                f"lr_dir={lr_dir} exists={lr_dir.exists()} "
                f"hr_dir={hr_dir} exists={hr_dir.exists()}"
            )
        else:
            missing_msg = "no lr/hr dirs configured"
        print(
            f"[split-diff][WARN] phase={phase} skipped because dirs are missing: "
            f"{missing_msg}"
        )
        return None

    print(f"[split-diff][INFO] phase={phase} build dataset")
    ds = create_dataset(phase_cfg, parent_configs=cfg)
    n = len(ds)
    indices = _sample_indices(n, max_samples)
    print(
        f"[split-diff][INFO] phase={phase} len={n} sampled={len(indices)} "
        f"{_fmt_dir_pairs(dir_pairs)}"
    )

    lr_means, gt_means, lr_stds, gt_stds = [], [], [], []
    mask_means, mask_valid_ratios, temporal_valid_ratios = [], [], []
    timestamps_min, timestamps_max = [], []
    first_paths = []
    key_stats = {}

    for idx in indices:
        sample = ds[idx]
        if sample.get("path") is not None and len(first_paths) < 5:
            first_paths.append(str(sample["path"]))

        for key in ("lr_sequence", "gt", "mask", "mask_prob", "indicating_mask", "timestamps"):
            if key in sample and key not in key_stats:
                key_stats[key] = _tensor_stats(sample[key])

        lr = sample.get("lr_sequence")
        gt = sample.get("gt")
        if torch.is_tensor(lr):
            lr_f = lr.float()
            lr_means.append(float(lr_f.mean().item()))
            lr_stds.append(float(lr_f.std(unbiased=False).item()))
        if torch.is_tensor(gt):
            gt_f = gt.float()
            gt_means.append(float(gt_f.mean().item()))
            gt_stds.append(float(gt_f.std(unbiased=False).item()))

        mask = sample.get("mask")
        if torch.is_tensor(mask):
            temporal_valid_ratios.append(float(mask.float().mean().item()))

        for key in ("mask_prob", "indicating_mask"):
            mt = sample.get(key)
            if torch.is_tensor(mt):
                mf = mt.float().clamp(0, 1)
                mask_means.append(float(mf.mean().item()))
                mask_valid_ratios.append(float((mf > 0.5).float().mean().item()))

        ts = sample.get("timestamps")
        if torch.is_tensor(ts):
            timestamps_min.append(float(ts.float().min().item()))
            timestamps_max.append(float(ts.float().max().item()))

    summary = {
        "len": n,
        "sampled": len(indices),
        "lr_mean": _concat_stats(lr_means),
        "gt_mean": _concat_stats(gt_means),
        "lr_std": _concat_stats(lr_stds),
        "gt_std": _concat_stats(gt_stds),
        "mask_mean": _concat_stats(mask_means),
        "mask_valid_ratio": _concat_stats(mask_valid_ratios),
        "temporal_valid_ratio": _concat_stats(temporal_valid_ratios),
        "timestamp_min": _concat_stats(timestamps_min),
        "timestamp_max": _concat_stats(timestamps_max),
        "key_stats": key_stats,
        "paths": first_paths,
    }

    print(f"[split-diff][SUMMARY] phase={phase}")
    for key, stats in key_stats.items():
        print("[split-diff] " + _fmt_stats(f"{phase}.{key}", stats))
    for key in (
        "lr_mean",
        "lr_std",
        "gt_mean",
        "gt_std",
        "mask_mean",
        "mask_valid_ratio",
        "temporal_valid_ratio",
        "timestamp_min",
        "timestamp_max",
    ):
        print("[split-diff] " + _fmt_scalar_stats(f"{phase}.{key}", summary[key]))
    if first_paths:
        print(f"[split-diff] {phase}.paths_preview={first_paths}")
    return summary


def compare_pair(left_name, left, right_name, right):
    if left is None or right is None:
        return
    print(f"[split-diff][COMPARE] {left_name} vs {right_name}")
    for key in ("lr_mean", "lr_std", "gt_mean", "gt_std", "mask_mean", "mask_valid_ratio", "temporal_valid_ratio"):
        lstat = left.get(key)
        rstat = right.get(key)
        if not lstat or not rstat:
            continue
        delta = rstat["mean"] - lstat["mean"]
        denom = abs(lstat["mean"]) + 1e-8
        rel = delta / denom
        level = "WARN" if abs(rel) > 0.20 and abs(delta) > 0.02 else "INFO"
        print(
            f"[split-diff][{level}] {key}: {left_name}_mean={lstat['mean']:.4f} "
            f"{right_name}_mean={rstat['mean']:.4f} delta={delta:.4f} rel={rel:.2%}"
        )


def main():
    parser = argparse.ArgumentParser(description="CPU-only dataset split diagnostics.")
    parser.add_argument("--cfg_path", required=True)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--phases", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    cfg = OmegaConf.load(args.cfg_path)
    print(f"[split-diff][INFO] cfg_path={args.cfg_path}")
    print(f"[split-diff][INFO] max_samples={args.max_samples} phases={args.phases}")

    summaries = {}
    for phase in args.phases:
        try:
            summaries[phase] = summarize_phase(cfg, phase, args.max_samples)
        except Exception as exc:
            summaries[phase] = None
            print(f"[split-diff][ERROR] phase={phase} failed: {exc}")

    compare_pair("train", summaries.get("train"), "val", summaries.get("val"))
    compare_pair("train", summaries.get("train"), "test", summaries.get("test"))
    compare_pair("val", summaries.get("val"), "test", summaries.get("test"))
    print("[split-diff][INFO] done")


if __name__ == "__main__":
    main()
