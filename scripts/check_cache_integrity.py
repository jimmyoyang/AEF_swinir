#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datapipe.datasets import _compute_hard_valid_mask, _select_reflectance_bands


def _load_config(cfg_path: Path) -> Dict:
    # Prefer OmegaConf, fallback to yaml.safe_load.
    try:
        from omegaconf import OmegaConf  # type: ignore

        cfg = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)
        if not isinstance(cfg, dict):
            raise ValueError("Config root is not a mapping.")
        return cfg
    except Exception:
        import yaml  # type: ignore

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config root is not a mapping.")
        return cfg


def _robust_per_image_normalize(img: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    img_float = img.astype(np.float32)
    normalized_bands = []
    for i in range(img_float.shape[0]):
        band = img_float[i]
        if np.all(band == 0):
            normalized_bands.append(band.astype(np.float32))
            continue
        lo_p, hi_p = np.percentile(band, (lo, hi))
        if hi_p - lo_p < 1e-6:
            normalized = np.zeros_like(band, dtype=np.float32)
        else:
            clipped = np.clip(band, lo_p, hi_p)
            normalized = (clipped - lo_p) / (hi_p - lo_p)
        normalized_bands.append((normalized * 2.0 - 1.0).astype(np.float32))
    return np.stack(normalized_bands, axis=0)


def _day_of_year_from_name(fname: str) -> int:
    date_str = fname.split("_")[0]
    y = int(date_str[0:4])
    m = int(date_str[4:6])
    d = int(date_str[6:8])
    dt = np.datetime64(f"{y:04d}-{m:02d}-{d:02d}")
    year_start = np.datetime64(f"{y:04d}-01-01")
    return int((dt - year_start).astype(int) + 1)


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def _discover_tile_ids(cache_dir: Path) -> List[str]:
    tile_ids = set()
    for p in cache_dir.glob("tile_*_reflectance.npy"):
        m = re.match(r"tile_(.+)_reflectance\.npy$", p.name)
        if m:
            tile_ids.add(m.group(1))
    return sorted(tile_ids)


def _check_tile_files_exist(cache_dir: Path, tile_id: str) -> Tuple[bool, List[str]]:
    expected = [
        cache_dir / f"tile_{tile_id}_reflectance.npy",
        cache_dir / f"tile_{tile_id}_hard_valid_mask.npy",
        cache_dir / f"tile_{tile_id}_day_of_year.npy",
        cache_dir / f"tile_{tile_id}_file_names.txt",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    return (len(missing) == 0, missing)


def _hard_mask_cache_key(valid_mask_band_count: int) -> str:
    payload = {
        "version": 2,
        "valid_mask_band_count": int(valid_mask_band_count),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def _load_hard_mask_sidecar(tile_id: str, names: List[str], params: Dict, valid_mask_band_count: int):
    sidecar_dir = params.get("hard_mask_cache_dir") or params.get("mask_cache_dir")
    if not sidecar_dir:
        return None

    sidecar_dir = Path(sidecar_dir)
    key = _hard_mask_cache_key(valid_mask_band_count)
    base = sidecar_dir / f"tile_{tile_id}_hardmask_{key}"
    mask_path = Path(str(base) + "_valid.npy")
    names_path = Path(str(base) + "_names.txt")
    if not mask_path.exists() or not names_path.exists():
        return None

    try:
        cached_names = _read_lines(names_path)
        if cached_names != list(names):
            return None
        return np.load(mask_path, allow_pickle=False)
    except Exception as e:
        print(f"[WARN] tile={tile_id} failed loading hard-mask sidecar: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Anytime cache integrity.")
    parser.add_argument("--cfg_path", type=str, default="configs/config_swinir.yaml")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--lr_dir", type=str, default=None)
    parser.add_argument("--hr_dir", type=str, default=None)
    parser.add_argument("--target_hr_rule", type=str, default="latest", choices=["latest", "earliest"])
    parser.add_argument("--spot_check", type=int, default=0, help="Number of timesteps to verify against raw LR per tile.")
    parser.add_argument("--max_tiles", type=int, default=0, help="Only check first N tiles (0 means all).")
    parser.add_argument(
        "--valid_mask_band_count",
        type=int,
        default=0,
        help="Leading LR bands used for hard valid mask; 0 auto-detects trailing QA/mask-like bands.",
    )
    parser.add_argument(
        "--reflectance_band_count",
        type=int,
        default=0,
        help="Leading LR bands expected in reflectance cache; 0 uses config value, then all bands.",
    )
    parser.add_argument("--tol_reflectance", type=float, default=2e-2)
    parser.add_argument("--tol_mask", type=float, default=1e-6)
    args = parser.parse_args()

    cfg = _load_config(Path(args.cfg_path))
    data_cfg = cfg.get("data", {})
    phase_cfg = data_cfg.get(args.phase, {})
    params = phase_cfg.get("params", {}) if isinstance(phase_cfg, dict) else {}

    cache_dir = Path(args.cache_dir or params.get("cache_dir", ""))
    lr_dir = Path(args.lr_dir or params.get("lr_dir", ""))
    hr_dir = Path(args.hr_dir or params.get("hr_dir", ""))

    if not cache_dir.exists():
        print(f"[ERROR] cache_dir not found: {cache_dir}")
        return 2
    if lr_dir and not lr_dir.exists():
        print(f"[WARN] lr_dir not found: {lr_dir}")
    if hr_dir and not hr_dir.exists():
        print(f"[WARN] hr_dir not found: {hr_dir}")

    tile_ids = _discover_tile_ids(cache_dir)
    if args.max_tiles > 0:
        tile_ids = tile_ids[: args.max_tiles]
    if not tile_ids:
        print(f"[ERROR] No cache tiles found in: {cache_dir}")
        return 2

    errors = 0
    warnings = 0
    checked_timesteps = 0
    spot_checked = 0

    print(f"[INFO] phase={args.phase}")
    print(f"[INFO] cache_dir={cache_dir}")
    print(f"[INFO] lr_dir={lr_dir}")
    print(f"[INFO] hr_dir={hr_dir}")
    print(f"[INFO] tiles={len(tile_ids)}")

    rng = np.random.default_rng(0)
    valid_mask_band_count = int(args.valid_mask_band_count or params.get("valid_mask_band_count", 0) or 0)
    reflectance_band_count = int(args.reflectance_band_count or params.get("reflectance_band_count", 0) or 0)

    for tile_id in tile_ids:
        ok, missing = _check_tile_files_exist(cache_dir, tile_id)
        if not ok:
            errors += 1
            print(f"[ERROR] tile={tile_id} missing files: {missing}")
            continue

        refl_path = cache_dir / f"tile_{tile_id}_reflectance.npy"
        mask_path = cache_dir / f"tile_{tile_id}_hard_valid_mask.npy"
        doy_path = cache_dir / f"tile_{tile_id}_day_of_year.npy"
        names_path = cache_dir / f"tile_{tile_id}_file_names.txt"

        try:
            refl = np.load(refl_path, allow_pickle=False)
            mask = np.load(mask_path, allow_pickle=False)
            doy = np.load(doy_path, allow_pickle=False)
            names = _read_lines(names_path)
        except Exception as e:
            errors += 1
            print(f"[ERROR] tile={tile_id} failed loading cache arrays: {e}")
            continue

        active_mask = mask
        sidecar_mask = _load_hard_mask_sidecar(tile_id, names, params, valid_mask_band_count)
        if sidecar_mask is not None:
            active_mask = sidecar_mask

        # Shape checks.
        if refl.ndim != 4:
            errors += 1
            print(f"[ERROR] tile={tile_id} reflectance ndim={refl.ndim}, expected 4")
            continue
        if mask.ndim != 3:
            errors += 1
            print(f"[ERROR] tile={tile_id} hard_valid_mask ndim={mask.ndim}, expected 3")
            continue
        if doy.ndim != 1:
            errors += 1
            print(f"[ERROR] tile={tile_id} day_of_year ndim={doy.ndim}, expected 1")
            continue

        t_refl, c_refl, h_refl, w_refl = refl.shape
        if active_mask.shape != (t_refl, h_refl, w_refl):
            errors += 1
            print(
                f"[ERROR] tile={tile_id} shape mismatch: reflectance={refl.shape}, active_hard_valid_mask={active_mask.shape}"
            )
        if len(doy) != t_refl or len(names) != t_refl:
            errors += 1
            print(
                f"[ERROR] tile={tile_id} T mismatch: reflectance={t_refl}, doy={len(doy)}, file_names={len(names)}"
            )

        # Numeric sanity.
        if not np.isfinite(refl).all():
            errors += 1
            print(f"[ERROR] tile={tile_id} reflectance contains NaN/Inf")
        if not np.isfinite(active_mask).all():
            errors += 1
            print(f"[ERROR] tile={tile_id} hard_valid_mask contains NaN/Inf")
        if refl.size > 0:
            rmin, rmax = float(np.min(refl)), float(np.max(refl))
            if rmin < -1.2 or rmax > 1.2:
                warnings += 1
                print(f"[WARN] tile={tile_id} reflectance out of expected range [-1,1]: [{rmin:.4f},{rmax:.4f}]")
        mmin, mmax = float(np.min(active_mask)), float(np.max(active_mask))
        if mmin < -1e-6 or mmax > 1.0 + 1e-6:
            errors += 1
            print(f"[ERROR] tile={tile_id} hard_valid_mask out of range [0,1]: [{mmin:.6f},{mmax:.6f}]")

        # File order and DOY checks.
        sorted_names = sorted(names)
        if names != sorted_names:
            warnings += 1
            print(f"[WARN] tile={tile_id} file_names are not sorted; temporal order may be unstable")

        for i, fname in enumerate(names):
            checked_timesteps += 1
            if lr_dir and (lr_dir / fname).exists() is False:
                errors += 1
                print(f"[ERROR] tile={tile_id} missing LR file referenced by cache: {fname}")
            try:
                expected_doy = _day_of_year_from_name(fname)
                if int(doy[i]) != int(expected_doy):
                    errors += 1
                    print(
                        f"[ERROR] tile={tile_id} DOY mismatch at t={i}: cached={int(doy[i])}, expected={expected_doy}, file={fname}"
                    )
            except Exception:
                warnings += 1
                print(f"[WARN] tile={tile_id} cannot parse date from file name: {fname}")

        # HR target existence check aligned with rule.
        if hr_dir and names:
            target_name = names[-1] if args.target_hr_rule == "latest" else names[0]
            if not (hr_dir / target_name).exists():
                errors += 1
                print(f"[ERROR] tile={tile_id} missing target HR file ({args.target_hr_rule}): {target_name}")

        # Optional raw spot check.
        if args.spot_check > 0 and lr_dir and lr_dir.exists():
            idxs = np.arange(t_refl)
            if len(idxs) > args.spot_check:
                idxs = rng.choice(idxs, size=args.spot_check, replace=False)
            for i in np.sort(idxs):
                fname = names[int(i)]
                lr_path = lr_dir / fname
                if not lr_path.exists():
                    continue
                try:
                    with rasterio.open(lr_path) as src:
                        raw = src.read()
                except Exception as e:
                    errors += 1
                    print(f"[ERROR] tile={tile_id} failed reading raw LR {fname}: {e}")
                    continue

                recomputed_mask = _compute_hard_valid_mask(
                    raw,
                    valid_mask_band_count=valid_mask_band_count,
                )
                cached_mask = active_mask[int(i)].astype(np.float32)
                mask_diff = float(np.max(np.abs(recomputed_mask - cached_mask)))
                if mask_diff > args.tol_mask:
                    errors += 1
                    print(
                        f"[ERROR] tile={tile_id} t={int(i)} hard_mask mismatch max_abs_diff={mask_diff:.6g} file={fname}"
                    )

                raw_reflectance = _select_reflectance_bands(
                    raw,
                    reflectance_band_count=reflectance_band_count,
                )
                recomputed_reflectance = _robust_per_image_normalize(raw_reflectance)
                cached_reflectance = _select_reflectance_bands(
                    refl[int(i)].astype(np.float32),
                    reflectance_band_count=reflectance_band_count,
                )
                if recomputed_reflectance.shape != cached_reflectance.shape:
                    errors += 1
                    print(
                        f"[ERROR] tile={tile_id} t={int(i)} reflectance shape mismatch "
                        f"recomputed={recomputed_reflectance.shape} cached={cached_reflectance.shape} file={fname}"
                    )
                else:
                    ref_diff = float(np.max(np.abs(recomputed_reflectance - cached_reflectance)))
                    if ref_diff > args.tol_reflectance:
                        errors += 1
                        print(
                            f"[ERROR] tile={tile_id} t={int(i)} reflectance mismatch "
                            f"max_abs_diff={ref_diff:.6g} tol={args.tol_reflectance} file={fname}"
                        )
                spot_checked += 1

    print("-" * 80)
    print(f"[SUMMARY] tiles_checked={len(tile_ids)} timesteps_checked={checked_timesteps} spot_checked={spot_checked}")
    print(f"[SUMMARY] errors={errors} warnings={warnings}")
    if errors > 0:
        print("[RESULT] FAIL")
        return 2
    print("[RESULT] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
