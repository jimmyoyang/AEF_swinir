# 文件路径: inference.py
# (一个统一的、生产级别的推理脚本)

import os
import sys
import yaml
import torch
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import rasterio
from tqdm import tqdm
import argparse

# --- 环境设置 ---
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))
from utils.util_common import get_obj_from_str
from datapipe.datasets import create_dataset

# ==============================================================================
# 1. 推理器类 (Predictor)
# ==============================================================================

class Predictor:
    """
    一个封装了模型加载和推理逻辑的可重用类。
    """
    def __init__(self, config_path, ckpt_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        self.configs = OmegaConf.load(config_path)
        
        # 初始化模型
        print(f"[Predictor] Initializing model: {self.configs.model.target}")
        self.model = get_obj_from_str(self.configs.model.target)(**self.configs.model.params).to(self.device)
        
        # 加载模型权重
        print(f"[Predictor] Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("[Predictor] Model loaded successfully.")

    @torch.no_grad()
    def run_on_batch(self, data_batch):
        """对一个数据批次进行推理。"""
        inputs = {k: v.to(self.device) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
        
        # 使用混合精度以提高效率
        with torch.cuda.amp.autocast(enabled=True):
            predictions = self.model(inputs)
            
        return predictions.cpu()

    def save_prediction(self, prediction_tensor, original_hr_path, output_path):
        """将预测结果保存为GeoTIFF文件，并继承原始地理信息。"""
        # 反归一化 (假设训练时归一化到 [-1, 1])
        # 注意：这里的反归一化逻辑需要与您的数据预处理严格对应
        # 这是一个简化的示例，可能需要根据您的 robust_normalize 进行调整
        pred_numpy = (prediction_tensor.squeeze(0).numpy() + 1) / 2.0 * 255 # 假设原始范围是0-255
        pred_numpy = pred_numpy.astype(np.uint8)

        # 从原始HR文件读取地理元数据
        with rasterio.open(original_hr_path) as src:
            profile = src.profile
        
        profile.update({
            'driver': 'GTiff',
            'count': pred_numpy.shape[0],
            'dtype': pred_numpy.dtype
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(pred_numpy)

# ==============================================================================
# 2. 主执行逻辑
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Inference Script for AnytimeSwinIR")
    parser.add_argument('--config', type=str, required=True, help="Path to the model's training config file (.yaml).")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model checkpoint file (.pth).")
    parser.add_argument('--input', type=str, required=True, help="Path to the input data directory (e.g., './data/processed_data/test').")
    parser.add_argument('--output', type=str, required=True, help="Path to the directory to save inference results.")
    args = parser.parse_args()

    # --- 1. 初始化 ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🚀 Starting Inference...")
    print(f" - Model Config: {args.config}")
    print(f" - Checkpoint: {args.ckpt}")
    print(f" - Input Data: {args.input}")
    print(f" - Output Dir: {args.output}")
    print("="*80)

    # --- 2. 创建推理器 ---
    try:
        predictor = Predictor(args.config, args.ckpt)
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize predictor. Error: {e}")
        sys.exit(1)

    # --- 3. 创建数据集和加载器 ---
    try:
        # 使用与训练时相同的配置来创建数据集
        data_config = {
            'target': 'datapipe.datasets.AnytimeTemporalDataset',
            'params': {
                'lr_dir': str(Path(args.input) / 'LR'),
                'hr_dir': str(Path(args.input) / 'HR'),
                'need_path': True # 确保返回路径以便保存
            }
        }
        dataset = create_dataset(data_config, parent_configs=predictor.configs)
        # 推理时 batch_size 通常为 1
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"❌ CRITICAL: Failed to create data loader. Error: {e}")
        sys.exit(1)

    # --- 4. 循环执行推理 ---
    pbar = tqdm(loader, desc="Running inference on dataset")
    for data_batch in pbar:
        try:
            # 执行推理
            predictions = predictor.run_on_batch(data_batch)
            
            # 获取原始HR路径用于保存和命名
            original_hr_path = data_batch['path'][0]
            output_filename = f"pred_{Path(original_hr_path).name}"
            output_path = output_dir / output_filename
            
            # 保存预测结果
            predictor.save_prediction(predictions, original_hr_path, output_path)
            
        except Exception as e:
            print(f"\n⚠️ WARNING: Failed to process batch for {data_batch.get('path', ['N/A'])[0]}. Error: {e}")
            continue
            
    print("\n" + "="*80)
    print("🎉 Inference complete!")
    print(f"Results saved to: {args.output}")
    print("="*80)

if __name__ == '__main__':
    main()
