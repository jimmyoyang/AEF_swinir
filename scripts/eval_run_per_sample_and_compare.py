#!/usr/bin/env python3
"""Evaluate per-sample metrics for a given training run directory and compare checkpoints.

This script is meant for quick, reproducible debugging when training logs show
counter-intuitive trends (e.g., visual quality improves but PSNR drops).

It:
1) Parses the YAML config dump from <run_dir>/training.log
2) Builds the requested dataset split (val by default)
3) Detects per-timestep input channels C and injects model.params.in_chans (SwinIR)
4) For each checkpoint, computes *per-sample* metrics
   (PSNR/SSIM/ERGAS/SAM/Cosine/Angle/Masked-PSNR/Masked-Cosine)
   and saves a CSV.
5) For selected sample indices, saves side-by-side comparison figures across checkpoints.

Example:
  python scripts/eval_run_per_sample_and_compare.py \
    --run_dir training_logs/experiments/ablation_4f_mask_prob_or_learnable_pos_no_cross/2026-04-22_06-12-22 \
    --ckpt_iters 500 20000 \
    --sample_idxs 0

Outputs are written under <run_dir>/analysis/ by default.
"""

from __future__ import annotations

# Allow running via: `python scripts/xxx.py ...`
# In that case, Python puts `scripts/` on sys.path, not the repo root.
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sewar.full_ref import ergas as sewar_ergas
from sewar.full_ref import psnr as sewar_psnr
from sewar.full_ref import sam as sewar_sam
from sewar.full_ref import ssim as sewar_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datapipe.datasets import create_dataset
from utils import util_common, util_net


class _TensorCacheDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._samples[index]


def _autocast_ctx(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    # Prefer the new API when available.
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "autocast"):
        return amp.autocast(device_type="cuda")
    return torch.cuda.amp.autocast()


def _default_tensor_cache_path(out_dir: Path, phase: str) -> Path:
    return out_dir / f"tensor_cache_{phase}.pt"


def _squeeze_batch_dim(sample: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sample.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1:
            out[k] = v[0].contiguous().cpu()
        elif torch.is_tensor(v):
            out[k] = v.contiguous().cpu()
        else:
            out[k] = v
    return out


def _build_tensor_cache(
    ds,
    out_path: Path,
    num_workers: int,
    max_samples: int = 0,
    save_dtype: str = "float16",
) -> Path:
    """Run the dataset pipeline once and dump tensors to disk for fast multi-ckpt eval.

    This is specifically to avoid repeated rasterio + (optional) advanced mask processing
    when you evaluate many checkpoints.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)

    samples: List[Dict[str, Any]] = []
    limit = max_samples if (max_samples and max_samples > 0) else None
    for i, batch in enumerate(tqdm(loader, desc=f"cache:{out_path.name}")):
        if limit is not None and i >= limit:
            break
        sample = _squeeze_batch_dim(batch)

        # Optional: compress to float16 to reduce disk footprint.
        if save_dtype == "float16":
            for key in ("lr_sequence", "gt", "timestamps", "mask", "mask_prob", "indicating_mask"):
                if key in sample and torch.is_tensor(sample[key]) and sample[key].dtype in (torch.float32, torch.float64):
                    sample[key] = sample[key].to(torch.float16)

        # Make sure path is a plain str.
        if "path" in sample and isinstance(sample["path"], (list, tuple)):
            sample["path"] = str(sample["path"][0])
        elif "path" in sample:
            sample["path"] = str(sample["path"])

        samples.append(sample)

    payload = {
        "version": 1,
        "num_samples": len(samples),
        "samples": samples,
    }
    torch.save(payload, str(out_path))
    return out_path


def _load_tensor_cache(path: Path) -> _TensorCacheDataset:
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict) or "samples" not in obj:
        raise ValueError(f"Invalid tensor cache file: {path}")
    samples = obj["samples"]
    if not isinstance(samples, list):
        raise ValueError(f"Invalid tensor cache payload: {path}")
    return _TensorCacheDataset(samples)


_YAML_END_MARKERS = (
    "📊 Dataset",
    "✅ Dynamically detected",
    "📝 Instantiating loss",
    "🎨 Generating baseline visualization",
    "📈 Iter ",
)


def _read_yaml_config_from_training_log(training_log: Path) -> Dict[str, Any]:
    """Extract the initial YAML config block from training.log."""
    if not training_log.exists():
        raise FileNotFoundError(f"training.log not found: {training_log}")

    lines: List[str] = []
    with training_log.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if any(line.startswith(m) for m in _YAML_END_MARKERS):
                break
            lines.append(raw)

    # Strip trailing whitespace-only lines to keep YAML clean.
    while lines and lines[-1].strip() == "":
        lines.pop()

    cfg = yaml.safe_load("".join(lines))
    if not isinstance(cfg, dict) or "model" not in cfg or "data" not in cfg:
        raise ValueError(
            "Failed to parse YAML config from training.log. "
            "Expected a dict containing at least 'model' and 'data'."
        )
    return cfg


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            # Tensor-cache may store float16 for disk efficiency.
            # For CPU inference (no autocast), conv2d requires input and bias dtypes match.
            if v.dtype == torch.float16:
                v = v.float()
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _norm_01(x: torch.Tensor) -> np.ndarray:
    """Map [-1,1] -> [0,1] for metric/visualization."""
    x = x.clamp(-1, 1)
    x = (x + 1.0) / 2.0
    return x.detach().cpu().numpy()


def _pick_rgb_channels(num_channels: int, rgb_chn: Sequence[int]) -> Sequence[int]:
    if num_channels >= 3 and max(rgb_chn, default=2) < num_channels:
        return list(rgb_chn)
    if num_channels >= 3:
        return [0, 1, 2]
    return [0]


def _aggregate_indicating_mask(indicating_mask: Optional[torch.Tensor], target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if indicating_mask is None:
        return None

    # Accept shapes: (B,T,H,W), (T,H,W), (B,1,H,W), (H,W)
    mask = indicating_mask
    if mask.dim() == 4:  # (B,T,H,W) or (B,1,H,W)
        if mask.shape[1] > 1:
            mask = mask.max(dim=1, keepdim=True).values  # (B,1,H,W)
        # take first batch
        mask = mask[0, 0]
    elif mask.dim() == 3:  # (T,H,W)
        mask = mask.max(dim=0).values
    elif mask.dim() == 2:
        pass
    else:
        return None

    mask = mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    H, W = target_hw
    if tuple(mask.shape[-2:]) != (H, W):
        mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)

    return mask[0, 0].detach().cpu().numpy()


def _masked_psnr(gt_hwc: np.ndarray, pred_hwc: np.ndarray, indicating_mask: Optional[torch.Tensor]) -> Optional[float]:
    """Compute per-band PSNR only over pixels where aggregated indicating_mask is true."""
    mask = _aggregate_indicating_mask(indicating_mask, gt_hwc.shape[:2])
    if mask is None:
        return None

    m = mask > 0.5
    if m.sum() == 0:
        return None

    per_band = []
    for b in range(C):
        diff = gt_hwc[:, :, b][m] - pred_hwc[:, :, b][m]
        mse = float(np.mean(diff * diff))
        if mse <= 0:
            continue
        per_band.append(20.0 * np.log10(1.0 / np.sqrt(mse)))

    if not per_band:
        return None
    return float(np.mean(per_band))


def _embedding_cosine_metrics(
    gt_chw: torch.Tensor,
    pred_chw: torch.Tensor,
    indicating_mask: Optional[torch.Tensor],
    eps: float = 1e-8,
) -> Tuple[float, float, Optional[float]]:
    gt = gt_chw.float()
    pred = pred_chw.float()
    gt_norm = F.normalize(gt, p=2, dim=0, eps=eps)
    pred_norm = F.normalize(pred, p=2, dim=0, eps=eps)
    cosine_map = (gt_norm * pred_norm).sum(dim=0).clamp(-1.0, 1.0)
    valid = gt.norm(p=2, dim=0) > eps

    if valid.any():
        cosine = float(cosine_map[valid].mean().item())
        angle_deg = float(torch.rad2deg(torch.acos(cosine_map[valid])).mean().item())
    else:
        cosine = float("nan")
        angle_deg = float("nan")

    masked_cosine = None
    mask = _aggregate_indicating_mask(indicating_mask, tuple(gt.shape[-2:]))
    if mask is not None:
        mask_tensor = torch.from_numpy(mask > 0.5).to(valid.device)
        masked_valid = valid & mask_tensor
        if masked_valid.any():
            masked_cosine = float(cosine_map[masked_valid].mean().item())

    return cosine, angle_deg, masked_cosine


@dataclass
class SampleMetrics:
    idx: int
    path: str
    psnr: float
    ssim: float
    ergas: float
    sam: float
    cosine: float
    angle_deg: float
    masked_psnr: Optional[float]
    masked_cosine: Optional[float]


def _compute_metrics(gt_chw: torch.Tensor, pred_chw: torch.Tensor, indicating_mask: Optional[torch.Tensor]) -> SampleMetrics:
    gt_01 = _norm_01(gt_chw)  # (C,H,W)
    pred_01 = _norm_01(pred_chw)

    gt_hwc = np.transpose(gt_01, (1, 2, 0))
    pred_hwc = np.transpose(pred_01, (1, 2, 0))

    # Per-band PSNR/SSIM, then mean.
    psnrs = []
    ssims = []
    for b in range(gt_hwc.shape[-1]):
        psnrs.append(float(sewar_psnr(gt_hwc[:, :, b], pred_hwc[:, :, b], MAX=1.0)))
        ssims.append(float(sewar_ssim(gt_hwc[:, :, b], pred_hwc[:, :, b], MAX=1.0)[0]))

    mpsnr = _masked_psnr(gt_hwc, pred_hwc, indicating_mask)
    cosine, angle_deg, masked_cosine = _embedding_cosine_metrics(gt_chw, pred_chw, indicating_mask)

    return SampleMetrics(
        idx=-1,
        path="",
        psnr=float(np.mean(psnrs)),
        ssim=float(np.mean(ssims)),
        ergas=float(sewar_ergas(gt_hwc, pred_hwc)),
        sam=float(sewar_sam(gt_hwc, pred_hwc)),
        cosine=cosine,
        angle_deg=angle_deg,
        masked_psnr=mpsnr,
        masked_cosine=masked_cosine,
    )


def _detect_in_chans_from_dataset(ds) -> int:
    sample = ds[0]
    lr_seq = sample["lr_sequence"]
    if lr_seq.dim() != 4:
        raise ValueError(f"Expected lr_sequence to be (T,C,H,W), got {tuple(lr_seq.shape)}")
    _, C, _, _ = lr_seq.shape
    return int(C)


def _load_model_from_config(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_target = cfg["model"]["target"]
    model_params = dict(cfg["model"].get("params", {}))
    model_cls = util_common.get_obj_from_str(model_target)
    model = model_cls(**model_params).to(device)
    model.eval()
    return model


def _load_ckpt_into_model(model: torch.nn.Module, ckpt_path: Path, strict: bool = False) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Tolerate DDP prefixes, compile wrappers, and partial matches.
    load_info = util_net.reload_model(model, state_dict, strict=strict)
    return load_info


def _resolve_ckpt_paths(run_dir: Path, ckpt_iters: Sequence[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for raw in ckpt_iters:
        # allow direct path
        p = Path(raw)
        if p.exists():
            tag = p.stem
            out.append((tag, p))
            continue

        if raw in {"best", "model_best"}:
            p2 = run_dir / "ckpts" / "model_best.pth"
            if not p2.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p2}")
            out.append(("best", p2))
            continue

        # numeric iteration
        if re.fullmatch(r"\d+", raw):
            p3 = run_dir / "ckpts" / f"model_{raw}.pth"
            if not p3.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p3}")
            out.append((f"iter_{raw}", p3))
            continue

        raise ValueError(f"Unrecognized --ckpt_iters entry: {raw}")

    return out


def _save_metrics_csv(rows: List[SampleMetrics], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "idx", "path", "psnr", "ssim", "ergas", "sam",
            "cosine", "angle_deg", "masked_psnr", "masked_cosine",
        ])
        for r in rows:
            w.writerow([
                r.idx,
                r.path,
                f"{r.psnr:.6f}",
                f"{r.ssim:.6f}",
                f"{r.ergas:.6f}",
                f"{r.sam:.6f}",
                f"{r.cosine:.6f}",
                f"{r.angle_deg:.6f}",
                "" if r.masked_psnr is None else f"{r.masked_psnr:.6f}",
                "" if r.masked_cosine is None else f"{r.masked_cosine:.6f}",
            ])


def _summarize(rows: List[SampleMetrics]) -> Dict[str, float]:
    def mean_of(name: str) -> float:
        vals = [getattr(r, name) for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    out = {
        "psnr": mean_of("psnr"),
        "ssim": mean_of("ssim"),
        "ergas": mean_of("ergas"),
        "sam": mean_of("sam"),
        "cosine": mean_of("cosine"),
        "angle_deg": mean_of("angle_deg"),
    }
    mps = [r.masked_psnr for r in rows if r.masked_psnr is not None]
    if mps:
        out["masked_psnr"] = float(np.mean(mps))
    mcos = [r.masked_cosine for r in rows if r.masked_cosine is not None]
    if mcos:
        out["masked_cosine"] = float(np.mean(mcos))
    return out


def _plot_compare(
    cfg: Dict[str, Any],
    sample: Dict[str, Any],
    preds_by_tag: List[Tuple[str, torch.Tensor]],
    out_path: Path,
) -> None:
    lr_seq = sample["lr_sequence"]  # (T,C,H,W)
    gt = sample["gt"]  # (C,H,W)

    rgb_chn = cfg.get("train", {}).get("rgb_chn", [0, 1, 2])

    # Visualize first timestep LR.
    lr0 = lr_seq[0]  # (C,H,W)

    lr_vis = _norm_01(lr0)
    gt_vis = _norm_01(gt)

    lr_hwc = np.transpose(lr_vis, (1, 2, 0))
    gt_hwc = np.transpose(gt_vis, (1, 2, 0))

    rgb_lr = _pick_rgb_channels(lr_hwc.shape[-1], rgb_chn)
    rgb_hr = _pick_rgb_channels(gt_hwc.shape[-1], rgb_chn)

    n_pred = len(preds_by_tag)
    fig, axes = plt.subplots(2, 2 + n_pred, figsize=(4 * (2 + n_pred), 8))

    # Row 0: LR, HR, predictions
    axes[0, 0].imshow(lr_hwc[:, :, rgb_lr]); axes[0, 0].set_title("Input LR (t=0)")
    axes[0, 1].imshow(gt_hwc[:, :, rgb_hr]); axes[0, 1].set_title("Ground Truth (HR)")

    axes[1, 0].axis("off")
    axes[1, 1].axis("off")

    for j, (tag, pred) in enumerate(preds_by_tag):
        pred_vis = _norm_01(pred)
        pred_hwc = np.transpose(pred_vis, (1, 2, 0))
        rgb_pred = _pick_rgb_channels(pred_hwc.shape[-1], rgb_chn)

        err = np.abs(pred_hwc - gt_hwc).mean(axis=2)

        axes[0, 2 + j].imshow(pred_hwc[:, :, rgb_pred])
        axes[0, 2 + j].set_title(f"Prediction ({tag})")

        im = axes[1, 2 + j].imshow(err, cmap="hot")
        axes[1, 2 + j].set_title("Abs Error (mean over C)")
        fig.colorbar(im, ax=axes[1, 2 + j], fraction=0.046, pad=0.04)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Checkpoint Comparison | sample_idx={int(sample.get('__idx__', -1))}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Run directory containing training.log and ckpts/")
    p.add_argument(
        "--phase",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (val by default)",
    )
    p.add_argument(
        "--ckpt_iters",
        type=str,
        nargs="+",
        required=True,
        help="Checkpoint iterations (e.g. 500 20000), 'best', or direct .pth paths",
    )
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <run_dir>/analysis)")
    p.add_argument(
        "--sample_idxs",
        type=int,
        nargs="*",
        default=[0],
        help="Sample indices to render comparison figures for (default: 0)",
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--strict", action="store_true", help="Use strict checkpoint loading")

    p.add_argument(
        "--tensor_cache",
        type=str,
        default=None,
        help=(
            "Optional .pt cache of fully-processed dataset samples. If provided and the file exists, "
            "the script will load samples from it (fast). If it doesn't exist, pass --build_tensor_cache."
        ),
    )
    p.add_argument(
        "--build_tensor_cache",
        action="store_true",
        help="Build the tensor cache by running the dataset pipeline once (recommended for CPU/I/O heavy configs).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Limit number of samples when building tensor cache (0 means all).",
    )
    p.add_argument(
        "--tensor_cache_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Dtype to store tensors in the cache (float16 saves disk and is usually fine for eval).",
    )

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    training_log = run_dir / "training.log"

    cfg = _read_yaml_config_from_training_log(training_log)

    # Ensure dataset returns file paths if supported.
    data_cfg = cfg.get("data", {})
    if args.phase in data_cfg and isinstance(data_cfg[args.phase], dict):
        params = data_cfg[args.phase].setdefault("params", {})
        params.setdefault("need_path", True)

    # Build dataset and inject in_chans if needed.
    ds = create_dataset(cfg["data"][args.phase], parent_configs=cfg)
    in_chans = _detect_in_chans_from_dataset(ds)
    cfg.setdefault("model", {}).setdefault("params", {})["in_chans"] = in_chans

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_config(cfg, device=device)

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = _resolve_ckpt_paths(run_dir, args.ckpt_iters)

    # Optional: build/load a tensor cache to avoid repeated rasterio + processor overhead.
    tensor_cache_path = Path(args.tensor_cache) if args.tensor_cache else _default_tensor_cache_path(out_dir, args.phase)
    if args.build_tensor_cache:
        if not tensor_cache_path.exists():
            print(f"[cache] building tensor cache: {tensor_cache_path}")
        else:
            print(f"[cache] tensor cache already exists, will rebuild: {tensor_cache_path}")
        _build_tensor_cache(
            ds,
            tensor_cache_path,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            save_dtype=args.tensor_cache_dtype,
        )

    if tensor_cache_path.exists():
        print(f"[cache] using tensor cache: {tensor_cache_path}")
        ds = _load_tensor_cache(tensor_cache_path)

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Pre-fetch sample(s) for visualization (CPU).
    samples_for_vis: Dict[int, Dict[str, Any]] = {}
    for idx in set(args.sample_idxs):
        if 0 <= idx < len(ds):
            sample = ds[idx]
            sample["__idx__"] = idx
            samples_for_vis[idx] = sample

    for tag, ckpt_path in ckpts:
        load_info = _load_ckpt_into_model(model, ckpt_path, strict=args.strict)
        print(f"[load] {tag}: loaded={load_info.get('loaded')} missing={len(load_info.get('missing', []))} shape_mismatch={len(load_info.get('shape_mismatch', []))}")

        rows: List[SampleMetrics] = []

        pbar = tqdm(loader, desc=f"eval {tag}")
        for i, batch in enumerate(pbar):
            batch = _to_device(batch, device)

            with torch.no_grad():
                with _autocast_ctx(device):
                    pred = model(batch)

            gt = batch["gt"][0]
            pred0 = pred[0]
            ind_mask = batch.get("indicating_mask", None)
            if ind_mask is not None:
                ind_mask = ind_mask[0] if ind_mask.dim() >= 4 else ind_mask

            m = _compute_metrics(gt, pred0, ind_mask)
            m.idx = i
            m.path = str(batch.get("path", [""])[0]) if "path" in batch else ""
            rows.append(m)

        summary = _summarize(rows)
        print(
            f"[mean {tag}] PSNR={summary['psnr']:.4f} "
            f"Cosine={summary['cosine']:.4f} Angle={summary['angle_deg']:.2f} "
            f"SSIM={summary['ssim']:.4f} ERGAS={summary['ergas']:.4f} SAM={summary['sam']:.4f}"
            + (f" Masked-PSNR={summary.get('masked_psnr', float('nan')):.4f}" if 'masked_psnr' in summary else "")
            + (f" Masked-Cosine={summary.get('masked_cosine', float('nan')):.4f}" if 'masked_cosine' in summary else "")
        )

        out_csv = out_dir / f"per_sample_metrics_{tag}.csv"
        _save_metrics_csv(rows, out_csv)
        print(f"[save] {out_csv}")

        # Render comparison figures for selected samples for this checkpoint only (optional).
        # We'll do combined comparisons after all checkpoints are evaluated.

    # Combined comparison figures across checkpoints.
    preds_for_vis: Dict[int, List[Tuple[str, torch.Tensor]]] = {idx: [] for idx in samples_for_vis}
    for tag, ckpt_path in ckpts:
        _load_ckpt_into_model(model, ckpt_path, strict=args.strict)
        for idx, sample in samples_for_vis.items():
            batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in sample.items() if k != "__idx__"}
            batch = _to_device(batch, device)
            with torch.no_grad():
                with _autocast_ctx(device):
                    pred = model(batch)
            preds_for_vis[idx].append((tag, pred[0].detach().cpu()))

    for idx, sample in samples_for_vis.items():
        out_png = out_dir / f"compare_sample_{idx}__{'__'.join([t for t,_ in ckpts])}.png"
        _plot_compare(cfg, sample, preds_for_vis[idx], out_png)
        print(f"[save] {out_png}")


if __name__ == "__main__":
    main()
