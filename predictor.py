# 文件路径: inference.py (推荐将 Predictor 放入此文件)

import os, sys, yaml, torch
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import rasterio
from tqdm import tqdm

# --- 环境设置 ---
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))
from utils.util_common import get_obj_from_str

class Predictor:
    def __init__(self, config_path, ckpt_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.configs = OmegaConf.load(config_path)
        
        print(f"[Predictor] Initializing model: {self.configs.model.target}")
        self.model = get_obj_from_str(self.configs.model.target)(**self.configs.model.params).to(self.device)
        
        print(f"[Predictor] Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("[Predictor] Model loaded successfully.")

    @torch.no_grad()
    def run_on_batch(self, data_batch):
        inputs = {k: v.to(self.device) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
        with torch.cuda.amp.autocast(enabled=True):
            predictions = self.model(inputs)
        return predictions.cpu()

    def save_prediction(self, prediction_tensor, original_hr_path, output_path):
        pred_numpy = (prediction_tensor.squeeze(0).numpy() + 1) / 2.0 * 65535 # 假设反归一化到16-bit
        pred_numpy = pred_numpy.astype(np.uint16)

        with rasterio.open(original_hr_path) as src:
            profile = src.profile
        
        profile.update({'driver': 'GTiff', 'count': pred_numpy.shape[0], 'dtype': pred_numpy.dtype})

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(pred_numpy)

    def run_inference(self, test_loader, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        pbar = tqdm(test_loader, desc=f'Inference on {out_dir.name}')
        for data_batch in pbar:
            try:
                predictions = self.run_on_batch(data_batch)
                original_hr_path = data_batch['path'][0]
                output_filename = f"pred_{Path(original_hr_path).name}"
                output_path = out_dir / output_filename
                self.save_prediction(predictions, original_hr_path, output_path)
            except Exception as e:
                print(f"\n⚠️ WARNING: Failed to process batch for {data_batch.get('path', ['N/A'])[0]}. Error: {e}")
                continue
        print(f"\n🎉 Inference complete! Results saved to: {out_dir}")
