# 文件路径: inference_srcnn.py
# SRCNN专用推理器：修复SRCNN(tensor)vs SwinIR(dict)的输入差异

import os
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import rasterio
from tqdm import tqdm
import argparse

# 环境设置
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))
from utils.util_common import get_obj_from_str
from datapipe.datasets import create_dataset

# ==============================================================================
# 1. SRCNN专用推理预测器类
# ==============================================================================
class PredictorSRCNN:
    """
    SRCNN推理预测器：
    - 仅传lr_sequence张量给模型（不是字典）
    - 简化版本无时间维度处理
    - 兼容DDP模型权重加载
    """
    def __init__(self, configs, ckpt_path, in_chans):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.configs = configs
        
        # 设置SRCNN的in_channels
        self.configs.model.params.in_channels = in_chans
        if 'in_chans' in self.configs.model.params:
            del self.configs.model.params['in_chans']
        print(f"[PredictorSRCNN] Initializing SRCNN with in_channels={in_chans}")
        
        # 实例化模型
        try:
            self.model = get_obj_from_str(self.configs.model.target)(**self.configs.model.params).to(self.device)
        except Exception as e:
            print(f"[Error] Failed to instantiate model: {e}")
            raise
        
        # 加载权重
        print(f"[PredictorSRCNN] Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        
        # 移除DDP的module.前缀
        unwrapped_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            unwrapped_state_dict[name] = v
        
        self.model.load_state_dict(unwrapped_state_dict, strict=True)
        self.model.eval()
        print("[PredictorSRCNN] Model loaded successfully (eval mode).")

    @torch.no_grad()
    def _resolve_input_tensor(self, data_batch):
        """兼容不同数据集键名，返回SRCNN输入张量(B,C,H,W)。"""
        for key in ('s1', 'lr', 'lr_sequence'):
            if key in data_batch and isinstance(data_batch[key], torch.Tensor):
                x = data_batch[key].to(self.device)
                if x.dim() == 5:
                    x = x[:, -1, ...]
                if x.dim() != 4:
                    raise ValueError(f"Unsupported SRCNN input shape for key '{key}': {tuple(x.shape)}")
                return x
        raise KeyError(f"No SRCNN input key found. Available keys: {list(data_batch.keys())}")

    @torch.no_grad()
    def run_on_batch(self, data_batch):
        """
        【关键】单批次推理：仅传lr_sequence张量给SRCNN
        """
        # 数据移到设备
        srcnn_input = self._resolve_input_tensor(data_batch)
        
        # 混合精度推理
        with torch.cuda.amp.autocast(enabled=True):
            # 【重要】SRCNN只接受lr_sequence张量，不是字典
            predictions = self.model(srcnn_input)
        
        return predictions.cpu()

    def save_prediction(self, prediction_tensor, original_hr_path, output_path):
        """
        保存预测结果为GeoTIFF：
        - 8-bit拉伸（2%/98%分位点）
        - 保留原始地理元数据
        - 异常处理保证鲁棒性
        """
        try:
            # 1. 张量转numpy (B,C,H,W) -> (C,H,W)
            pred_numpy = prediction_tensor.squeeze(0).numpy()
            
            # 2. 8-bit对比度拉伸（2%/98%分位点）
            stretched_bands = []
            for band_idx in range(pred_numpy.shape[0]):
                band = pred_numpy[band_idx]
                lo, hi = np.percentile(band, (2, 98))
                
                if hi - lo < 1e-6:
                    stretched_band = np.zeros_like(band, dtype=np.uint8)
                else:
                    stretched_band = np.clip(band, lo, hi)
                    stretched_band = ((stretched_band - lo) / (hi - lo)) * 255
                    stretched_band = stretched_band.astype(np.uint8)
                stretched_bands.append(stretched_band)
            
            final_numpy = np.stack(stretched_bands)

            # 3. 读取原始HR文件的地理元数据
            try:
                with rasterio.open(original_hr_path) as src:
                    profile = src.profile
            except Exception as e:
                print(f"[Warning] Failed to read geo-metadata from {original_hr_path}: {e}")
                profile = {
                    'driver': 'GTiff',
                    'width': final_numpy.shape[2],
                    'height': final_numpy.shape[1],
                    'count': final_numpy.shape[0],
                    'dtype': final_numpy.dtype,
                    'crs': 'EPSG:32648',
                    'transform': rasterio.Affine.identity()
                }

            # 4. 更新配置并保存
            profile.update({
                'driver': 'GTiff',
                'count': final_numpy.shape[0],
                'dtype': final_numpy.dtype,
                'compress': 'lzw'
            })
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(final_numpy)
            
            print(f"[Save] Prediction saved to: {output_path}")
        
        except Exception as e:
            print(f"[Error] Failed to save prediction: {e}")

    @torch.no_grad()
    def run_inference(self, test_loader, out_dir):
        """批量推理并保存结果"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(test_loader, desc=f"Inference on {out_dir.name}")
        for data_batch in pbar:
            try:
                # 【关键】调用SRCNN专用的推理方法
                predictions = self.run_on_batch(data_batch)
                original_hr_path = data_batch['path'][0] if 'path' in data_batch else None
                output_filename = f"pred_{Path(original_hr_path).name}" if original_hr_path else "pred.tif"
                output_path = out_dir / output_filename
                self.save_prediction(predictions, original_hr_path, output_path)
            except Exception as e:
                print(f"\n⚠️ Failed to process batch: {e}")
                continue

        print(f"\n🎉 Inference complete! Results saved to: {out_dir}")


# ==============================================================================
# 2. 供main.py调用的推理函数
# ==============================================================================
def run_inference_srcnn(args, configs):
    """
    SRCNN专用推理主流程
    参数：
        args: main.py解析的命令行参数
        configs: 加载后的配置文件
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80 + f"\n🚀 Starting SRCNN Inference\n" + "="*80)

    # 步骤1：创建数据集並动态检测输入通道数
    try:
        print("[Main] Initializing dataset...")
        test_data_config = configs.data.test
        test_data_config.params.lr_dir = str(Path(args.input_dir) / 'LR')
        test_data_config.params.hr_dir = str(Path(args.input_dir) / 'HR')
        
        dataset = create_dataset(test_data_config, parent_configs=configs)
        
        if hasattr(dataset, 'in_chans'):
            dynamic_in_chans = dataset.in_chans
            print(f"[Main] ✅ Detected input channels: {dynamic_in_chans}")
        else:
            temp_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
            first_batch = next(iter(temp_loader))
            if 's1' in first_batch:
                probe = first_batch['s1']
            elif 'lr' in first_batch:
                probe = first_batch['lr']
            elif 'lr_sequence' in first_batch:
                probe = first_batch['lr_sequence']
            else:
                raise KeyError(f"No SRCNN input key found in batch. Keys: {list(first_batch.keys())}")

            if probe.dim() == 5:
                dynamic_in_chans = int(probe.shape[2])
            elif probe.dim() == 4:
                dynamic_in_chans = int(probe.shape[1])
            else:
                raise ValueError(f"Unsupported probe shape: {tuple(probe.shape)}")
            print(f"[Main] ✅ Detected input channels: {dynamic_in_chans}")
            del temp_loader, first_batch
        
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
    except Exception as e:
        print(f"❌ Failed to create data loader: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # 步骤2：初始化SRCNN推理器
    try:
        predictor = PredictorSRCNN(configs, args.ckpt_path, dynamic_in_chans)
    except Exception as e:
        print(f"❌ Failed to initialize SRCNN predictor: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # 步骤3：批量推理并保存
    pbar = tqdm(loader, desc="Running SRCNN inference")
    for data_batch in pbar:
        try:
            # 【关键】使用SRCNN专用推理
            predictions = predictor.run_on_batch(data_batch)
            
            original_hr_path = data_batch['path'][0] if 'path' in data_batch else None
            output_filename = f"pred_{Path(original_hr_path).name}" if original_hr_path else "pred.tif"
            output_path = output_dir / output_filename
            
            predictor.save_prediction(predictions, original_hr_path, output_path)
        
        except Exception as e:
            print(f"\n⚠️ Failed to process batch: {e}")
            continue
            
    print("\n" + "="*80 + f"\n🎉 SRCNN Inference complete!\nResults: {args.output_dir}\n" + "="*80)


# ==============================================================================
# 3. 独立运行入口
# ==============================================================================
def main_standalone():
    """独立运行SRCNN推理脚本"""
    parser = argparse.ArgumentParser(description="Standalone SRCNN Inference Script")
    parser.add_argument('--config', type=str, required=True, help="Path to config file (.yaml)")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to checkpoint file (.pth)")
    parser.add_argument('--input', type=str, required=True, help="Input data directory (contains LR/HR subdirs)")
    parser.add_argument('--output', type=str, required=True, help="Output directory to save predictions")
    args = parser.parse_args()
    
    configs = OmegaConf.load(args.config)
    
    fake_main_args = argparse.Namespace(
        ckpt_path=args.ckpt,
        input_dir=args.input,
        output_dir=args.output
    )
    
    run_inference_srcnn(fake_main_args, configs)


if __name__ == '__main__':
    main_standalone()
