#!/usr/bin/env python3
"""Compute cosine similarity for samples from older train.sh runs.

Default behavior targets the old train.sh default experiment:
  training_logs/experiments/ablation_4f_multiyear_2018_2024

Examples:
  python scripts/eval_previous_train_sh_sample_cosine.py

  python scripts/eval_previous_train_sh_sample_cosine.py \
    --run_dir training_logs/experiments/ablation_4f_multiyear_2018_2024/2026-05-26_13-39-38 \
    --ckpt latest --max_samples 8
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datapipe.datasets import create_dataset
from utils import util_common, util_net


YAML_END_MARKERS = (
    "📊 Dataset",
    "✅ Dynamically detected",
    "📝 Instantiating loss",
    "🎨 Generating baseline visualization",
    "📈 Iter ",
)


def read_yaml_config_from_training_log(training_log: Path) -> Dict[str, Any]:
    if not training_log.exists():
        raise FileNotFoundError(f"training.log not found: {training_log}")

    lines: List[str] = []
    with training_log.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if any(line.startswith(marker) for marker in YAML_END_MARKERS):
                break
            lines.append(raw)

    while lines and lines[-1].strip() == "":
        lines.pop()

    cfg = yaml.safe_load("".join(lines))
    if not isinstance(cfg, dict) or "model" not in cfg or "data" not in cfg:
        raise ValueError(f"Could not parse config YAML from {training_log}")
    return cfg


def list_run_dirs(experiment_root: Path) -> List[Path]:
    if not experiment_root.exists():
        return []
    runs = [p for p in experiment_root.iterdir() if p.is_dir() and (p / "training.log").is_file()]
    return sorted(runs, key=lambda p: p.stat().st_mtime)


def has_any_checkpoint(run_dir: Path) -> bool:
    ckpt_dir = run_dir / "ckpts"
    if (ckpt_dir / "model_best.pth").is_file():
        return True
    return any(re.fullmatch(r"model_\d+\.pth", p.name) for p in ckpt_dir.glob("model_*.pth"))


def resolve_run_dir(run_dir: Optional[str], experiment_root: str) -> Path:
    if run_dir:
        out = Path(run_dir)
        if not out.is_dir():
            raise FileNotFoundError(f"--run_dir does not exist: {out}")
        return out

    root = Path(experiment_root)
    runs = list_run_dirs(root)
    if not runs:
        raise FileNotFoundError(f"No run dirs with training.log found under {root}")
    runs_with_ckpts = [run for run in runs if has_any_checkpoint(run)]
    return (runs_with_ckpts or runs)[-1]


def resolve_ckpt(run_dir: Path, ckpt: str) -> Tuple[str, Path]:
    raw = str(ckpt)
    direct = Path(raw)
    if direct.is_file():
        return direct.stem, direct

    ckpt_dir = run_dir / "ckpts"
    if raw in {"best", "model_best"}:
        path = ckpt_dir / "model_best.pth"
        if path.is_file():
            return "best", path
        raw = "latest"

    if raw == "latest":
        numbered = []
        for path in ckpt_dir.glob("model_*.pth"):
            match = re.fullmatch(r"model_(\d+)", path.stem)
            if match:
                numbered.append((int(match.group(1)), path))
        if not numbered:
            raise FileNotFoundError(f"No numbered checkpoints found under {ckpt_dir}")
        iteration, path = max(numbered, key=lambda x: x[0])
        return f"iter_{iteration}", path

    if re.fullmatch(r"\d+", raw):
        path = ckpt_dir / f"model_{raw}.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return f"iter_{raw}", path

    raise ValueError(f"Unsupported --ckpt value: {ckpt}")


def detect_in_chans(ds) -> int:
    sample = ds[0]
    lr_seq = sample["lr_sequence"]
    if lr_seq.dim() != 4:
        raise ValueError(f"Expected lr_sequence shape (T,C,H,W), got {tuple(lr_seq.shape)}")
    return int(lr_seq.shape[1])


def load_model(cfg: Dict[str, Any], ckpt_path: Path, device: torch.device, strict: bool = False) -> torch.nn.Module:
    model_cls = util_common.get_obj_from_str(cfg["model"]["target"])
    model_params = dict(cfg["model"].get("params", {}))
    model = model_cls(**model_params).to(device)
    model.eval()

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    load_info = util_net.reload_model(model, state_dict, strict=strict)
    print(
        f"[load] loaded={load_info.get('loaded')} "
        f"missing={len(load_info.get('missing', []))} "
        f"shape_mismatch={len(load_info.get('shape_mismatch', []))}"
    )
    return model


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.dtype == torch.float16 and device.type == "cpu":
                value = value.float()
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def autocast_ctx(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "autocast"):
        return amp.autocast(device_type="cuda")
    return torch.cuda.amp.autocast()


def normalize_indicating_mask(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    mask = mask.float()
    if mask.ndim == 5:
        if mask.shape[1] == 1:
            mask = mask.squeeze(1)
        elif mask.shape[2] == 1:
            mask = mask.squeeze(2)
        else:
            raise ValueError(f"Unsupported indicating_mask shape: {tuple(mask.shape)}")
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f"Unsupported indicating_mask shape: {tuple(mask.shape)}")

    if mask.shape[0] != batch_size:
        if mask.shape[0] == 1:
            mask = mask.expand(batch_size, -1, -1, -1)
        else:
            raise ValueError(f"Batch mismatch for indicating_mask: {tuple(mask.shape)} vs B={batch_size}")
    return mask


def reduce_temporal_mask(mask_bt_hw: torch.Tensor, strategy: str = "prob_or") -> torch.Tensor:
    mask_bt_hw = mask_bt_hw.clamp(0.0, 1.0)
    if mask_bt_hw.shape[1] == 1:
        return mask_bt_hw
    if strategy == "mean":
        return mask_bt_hw.mean(dim=1, keepdim=True)
    if strategy == "max":
        return mask_bt_hw.max(dim=1, keepdim=True).values
    if strategy == "prob_or":
        return 1.0 - torch.prod(1.0 - mask_bt_hw, dim=1, keepdim=True)
    raise ValueError(f"Unknown mask reduce strategy: {strategy}")


def sample_cosine_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    indicating_mask: Optional[torch.Tensor],
    mask_reduce: str,
    eps: float,
) -> List[Dict[str, float]]:
    pred = pred.float()
    gt = gt.float()
    pred_norm = F.normalize(pred, p=2, dim=1, eps=eps)
    gt_norm = F.normalize(gt, p=2, dim=1, eps=eps)
    cosine_map = (pred_norm * gt_norm).sum(dim=1).clamp(-1.0, 1.0)
    angle_map = torch.rad2deg(torch.acos(cosine_map))
    valid = gt.norm(p=2, dim=1) > eps

    spatial_mask = None
    if indicating_mask is not None:
        mask = normalize_indicating_mask(indicating_mask, batch_size=gt.shape[0])
        spatial_mask = reduce_temporal_mask(mask, strategy=mask_reduce)
        if spatial_mask.shape[-2:] != gt.shape[-2:]:
            spatial_mask = F.interpolate(spatial_mask, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        spatial_mask = spatial_mask[:, 0] > 0.5

    rows: List[Dict[str, float]] = []
    for i in range(gt.shape[0]):
        valid_i = valid[i]
        if valid_i.any():
            cosine = float(cosine_map[i][valid_i].mean().item())
            angle = float(angle_map[i][valid_i].mean().item())
        else:
            cosine = float("nan")
            angle = float("nan")

        masked_cosine = float("nan")
        if spatial_mask is not None:
            masked_valid = valid_i & spatial_mask[i]
            if masked_valid.any():
                masked_cosine = float(cosine_map[i][masked_valid].mean().item())

        rows.append(
            {
                "cosine": cosine,
                "angle_deg": angle,
                "masked_cosine": masked_cosine,
                "gt_norm_mean": float(gt[i].norm(p=2, dim=0).mean().item()),
                "pred_norm_mean": float(pred[i].norm(p=2, dim=0).mean().item()),
            }
        )
    return rows


def finite_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def finite_percentile(values: Iterable[float], q: float) -> float:
    vals = [v for v in values if math.isfinite(float(v))]
    return float(np.percentile(vals, q)) if vals else float("nan")


def path_from_batch(batch: Dict[str, Any], batch_idx: int) -> str:
    paths = batch.get("path", "")
    if isinstance(paths, (list, tuple)):
        if batch_idx < len(paths):
            return str(paths[batch_idx])
        return ""
    return str(paths) if paths else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_root",
        type=str,
        default="training_logs/experiments/ablation_4f_multiyear_2018_2024",
        help="Root containing older train.sh run dirs. Ignored when --run_dir is set.",
    )
    parser.add_argument("--run_dir", type=str, default=None, help="Specific run dir containing training.log.")
    parser.add_argument("--phase", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", type=str, default="best", help="'best', 'latest', iteration number, or .pth path.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="How many samples to evaluate. Default uses train.val_max_samples from the run config, else 8.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Default uses validation batch size from config.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--strict", action="store_true", help="Strict checkpoint loading.")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--list_runs", action="store_true", help="List discovered run dirs and exit.")
    args = parser.parse_args()

    if args.list_runs:
        for run in list_run_dirs(Path(args.experiment_root)):
            print(run)
        return

    run_dir = resolve_run_dir(args.run_dir, args.experiment_root)
    ckpt_tag, ckpt_path = resolve_ckpt(run_dir, args.ckpt)
    cfg = read_yaml_config_from_training_log(run_dir / "training.log")

    if args.phase not in cfg.get("data", {}):
        raise KeyError(f"Config has no data.{args.phase}")
    data_cfg = cfg["data"][args.phase]
    data_cfg.setdefault("params", {})
    data_cfg["params"]["need_path"] = True

    max_samples = args.max_samples
    if max_samples is None:
        max_samples = int(cfg.get("train", {}).get("val_max_samples", 8) or 8)

    print(f"[run] {run_dir}")
    print(f"[ckpt] {ckpt_tag}: {ckpt_path}")
    print(f"[phase] {args.phase} | max_samples={max_samples}")

    ds = create_dataset(data_cfg, parent_configs=cfg)
    in_chans = detect_in_chans(ds)
    cfg.setdefault("model", {}).setdefault("params", {})["in_chans"] = in_chans

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = load_model(cfg, ckpt_path=ckpt_path, device=device, strict=args.strict)

    default_batch_size = 1
    train_batch = cfg.get("train", {}).get("batch", None)
    if isinstance(train_batch, list) and len(train_batch) > 1:
        default_batch_size = int(train_batch[1])
    batch_size = int(args.batch_size or default_batch_size or 1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    mask_reduce = str(cfg.get("train", {}).get("indicating_mask_reduce", "prob_or"))
    eps = float(cfg.get("train", {}).get("metric_cosine_eps", 1e-8) or 1e-8)

    rows: List[Dict[str, Any]] = []
    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="cosine"):
            batch = to_device(batch, device)
            with autocast_ctx(device):
                pred = model(batch)

            gt = batch["gt"]
            metrics = sample_cosine_metrics(
                pred=pred,
                gt=gt,
                indicating_mask=batch.get("indicating_mask", None),
                mask_reduce=mask_reduce,
                eps=eps,
            )

            for bi, metric in enumerate(metrics):
                if seen >= max_samples:
                    break
                row = {
                    "idx": seen,
                    "path": path_from_batch(batch, bi),
                    **metric,
                }
                rows.append(row)
                seen += 1

            if seen >= max_samples:
                break

    out_csv = Path(args.out_csv) if args.out_csv else (run_dir / "analysis" / f"sample_cosine_{args.phase}_{ckpt_tag}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "path",
                "cosine",
                "angle_deg",
                "masked_cosine",
                "gt_norm_mean",
                "pred_norm_mean",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    cosines = [float(r["cosine"]) for r in rows]
    masked = [float(r["masked_cosine"]) for r in rows]
    print(
        "[summary] "
        f"n={len(rows)} "
        f"cos_mean={finite_mean(cosines):.6f} "
        f"cos_median={finite_percentile(cosines, 50):.6f} "
        f"cos_min={finite_percentile(cosines, 0):.6f} "
        f"cos_max={finite_percentile(cosines, 100):.6f} "
        f"masked_cos_mean={finite_mean(masked):.6f}"
    )
    print(f"[save] {out_csv}")

    if rows:
        worst = min(rows, key=lambda r: float(r["cosine"]) if math.isfinite(float(r["cosine"])) else float("inf"))
        best = max(rows, key=lambda r: float(r["cosine"]) if math.isfinite(float(r["cosine"])) else float("-inf"))
        print(f"[worst] idx={worst['idx']} cosine={float(worst['cosine']):.6f} path={worst['path']}")
        print(f"[best] idx={best['idx']} cosine={float(best['cosine']):.6f} path={best['path']}")


if __name__ == "__main__":
    main()
