#!/usr/bin/env python3
"""
Legacy-compatible Predictor module.

This file preserves the historical public interface:
    Predictor(config_path, ckpt_path)
    .run_on_batch(...)
    .save_prediction(...)
    .run_inference(...)

Main training/testing flow uses inference.py, but keeping this adapter avoids
breaking older scripts that still import predictor.py directly.
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))
from utils.util_common import get_obj_from_str


class Predictor:
    """Backward-compatible predictor class."""

    def __init__(self, config_path, ckpt_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs = OmegaConf.load(config_path)

        print(f"[Predictor] Initializing model: {self.configs.model.target}")
        self.model = get_obj_from_str(self.configs.model.target)(**self.configs.model.params).to(self.device)

        print(f"[Predictor] Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt)

        # Be tolerant to DDP prefixes in historical checkpoints.
        normalized_state = {}
        for k, v in state_dict.items():
            normalized_state[k[7:] if k.startswith("module.") else k] = v

        self.model.load_state_dict(normalized_state, strict=False)
        self.model.eval()
        print("[Predictor] Model loaded successfully.")

    @torch.no_grad()
    def run_on_batch(self, data_batch):
        inputs = {k: v.to(self.device) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            predictions = self.model(inputs)
        return predictions.cpu()

    def save_prediction(self, prediction_tensor, original_hr_path, output_path):
        pred_numpy = (prediction_tensor.squeeze(0).numpy() + 1.0) / 2.0 * 65535.0
        pred_numpy = np.clip(pred_numpy, 0, 65535).astype(np.uint16)

        with rasterio.open(original_hr_path) as src:
            profile = src.profile
        profile.update({"driver": "GTiff", "count": pred_numpy.shape[0], "dtype": pred_numpy.dtype})

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(pred_numpy)

    def run_inference(self, test_loader, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pbar = tqdm(test_loader, desc=f"Inference on {out_dir.name}")
        for data_batch in pbar:
            try:
                predictions = self.run_on_batch(data_batch)
                original_hr_path = data_batch["path"][0]
                output_filename = f"pred_{Path(original_hr_path).name}"
                output_path = out_dir / output_filename
                self.save_prediction(predictions, original_hr_path, output_path)
            except Exception as e:
                print(f"\nWARNING: Failed to process batch for {data_batch.get('path', ['N/A'])[0]}. Error: {e}")
                continue
        print(f"\nInference complete! Results saved to: {out_dir}")


__all__ = ["Predictor"]
