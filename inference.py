# 文件路径: inference.py
# 最终整合版：动态通道检测 + 健壮的预测保存 + 兼容main.py调用/独立运行
import os
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import rasterio
from tqdm import tqdm
import argparse
import hydra  # 兼容配置实例化（也保留原get_obj_from_str）

# --- 环境设置 (保留原有路径配置) ---
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))
from utils.util_common import get_obj_from_str
from datapipe.datasets import create_dataset

# ==============================================================================
# 1. 推理器类 (Predictor) - 整合最优逻辑
# ==============================================================================
class Predictor:
    """
    封装模型加载/推理/结果保存的可重用类
    核心特性：
    - 动态适配输入通道数
    - 兼容DDP/单卡模型权重加载
    - 健壮的8-bit拉伸保存（提升可视化效果）
    - 明确指定模型输入为lr_sequence
    """
    def __init__(self, configs, ckpt_path, in_chans):
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.configs = configs
        
        # --- 核心：根据模型类型动态注入输入通道数 ---
        model_name = self.configs.model.target.split('.')[-1].lower()
        if 'srcnn' in model_name:
            self.configs.model.params.in_channels = in_chans
            if 'in_chans' in self.configs.model.params:
                del self.configs.model.params['in_chans']
            print(f"[Predictor] Initializing SRCNN model with in_channels={in_chans}")
        else:
            self.configs.model.params.in_chans = in_chans
            if 'in_channels' in self.configs.model.params:
                del self.configs.model.params['in_channels']
            print(f"[Predictor] Initializing model '{self.configs.model.target}' with in_chans={in_chans}")
        
        # --- 实例化模型（兼容hydra和原有工具函数） ---
        try:
            # 优先使用hydra实例化（更通用）
            self.model = hydra.utils.instantiate(self.configs.model).to(self.device)
        except:
            # 兜底使用原有工具函数
            self.model = get_obj_from_str(self.configs.model.target)(**self.configs.model.params).to(self.device)
        
        # --- 加载模型权重（兼容DDP的module.前缀） ---
        print(f"[Predictor] Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device,weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)  # 兼容带/不带state_dict键的ckpt
        
        # 移除DDP添加的module.前缀
        unwrapped_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            unwrapped_state_dict[name] = v
        
        self.model.load_state_dict(unwrapped_state_dict, strict=True)
        self.model.eval()  # 推理模式
        print("[Predictor] Model loaded successfully (eval mode enabled).")

    @torch.no_grad()
    def run_on_batch(self, data_batch):
        """
        单批次推理：明确使用lr_sequence作为模型输入
        """
        # 数据移到设备（仅保留张量类型）
        inputs = {k: v.to(self.device) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
        
        # 混合精度推理（加速）
        with torch.cuda.amp.autocast(enabled=True):
            # --- 核心：明确指定模型输入为lr_sequence ---
            predictions = self.model(inputs)
        
        return predictions.cpu()  # 返回CPU张量，避免显存占用

    def save_prediction(self, prediction_tensor, original_hr_path, output_path):
        """
        保存预测结果为GeoTIFF：
        - 8-bit拉伸（2%/98%分位点）提升可视化效果
        - 保留原始地理元数据
        - 异常处理保证鲁棒性
        """
        try:
            # 1. 张量转numpy (B,C,H,W) -> (C,H,W)
            pred_numpy = prediction_tensor.squeeze(0).numpy()
            
            # 2. 8-bit对比度拉伸（2%/98%分位点，避免极端值）
            stretched_bands = []
            for band_idx in range(pred_numpy.shape[0]):
                band = pred_numpy[band_idx]
                # 计算分位点
                lo, hi = np.percentile(band, (2, 98))
                # 处理极端情况（全0/全1）
                if hi - lo < 1e-6:
                    stretched_band = np.zeros_like(band, dtype=np.uint8)
                else:
                    # 裁剪+归一化+拉伸到0-255
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
                # 备用配置（无原始文件时）
                print(f"[Warning] Failed to read geo-metadata from {original_hr_path}: {e}")
                profile = {
                    'driver': 'GTiff',
                    'width': final_numpy.shape[2],
                    'height': final_numpy.shape[1],
                    'count': final_numpy.shape[0],
                    'dtype': final_numpy.dtype,
                    'crs': 'EPSG:32648',  # 可根据实际场景调整
                    'transform': rasterio.Affine.identity()
                }

            # 4. 更新配置并保存
            profile.update({
                'driver': 'GTiff',
                'count': final_numpy.shape[0],
                'dtype': final_numpy.dtype,
                'compress': 'lzw'  # 压缩保存，减少文件体积
            })
            
            # 创建输出目录（如有需要）
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(final_numpy)
            
            print(f"[Save] Prediction saved to: {output_path}")
        
        except Exception as e:
            print(f"[Error] Failed to save prediction: {e}")

    @torch.no_grad()
    def run_inference(self, test_loader, out_dir):
        """批量推理并保存结果（供 main.py 的 test 模式直接调用）。"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(test_loader, desc=f"Inference on {out_dir.name}")
        for data_batch in pbar:
            try:
                predictions = self.run_on_batch(data_batch)
                original_hr_path = data_batch['path'][0] if 'path' in data_batch else None
                output_filename = f"pred_{Path(original_hr_path).name}" if original_hr_path else "pred.tif"
                output_path = out_dir / output_filename
                self.save_prediction(predictions, original_hr_path, output_path)
            except Exception as e:
                print(f"\n⚠️ WARNING: Failed to process batch for {data_batch.get('path', ['N/A'])[0]}. Error: {e}")
                continue

        print(f"\n🎉 Inference complete! Results saved to: {out_dir}")

# ==============================================================================
# 2. 供main.py调用的推理函数 (保留兼容性)
# ==============================================================================
def run_inference(args, configs):
    """
    供main.py的test模式调用的推理主流程
    参数：
        args: main.py解析的命令行参数
        configs: 加载后的配置文件
    """
    # 初始化输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80, f"\n🚀 Starting Inference on input: {args.input_dir}\n", "="*80)

    # --- 步骤1：创建数据集并动态检测输入通道数 ---
    try:
        print("[Main] Initializing dataset to detect input channels...")
        # 覆盖数据集路径为命令行指定路径
        test_data_config = configs.data.test
        test_data_config.params.lr_dir = str(Path(args.input_dir) / 'LR')
        test_data_config.params.hr_dir = str(Path(args.input_dir) / 'HR')
        
        # 创建数据集（传递父配置，兼容特征开关）
        dataset = create_dataset(test_data_config, parent_configs=configs)
        
        # 动态检测通道数（优先dataset属性，备用加载样本）
        if hasattr(dataset, 'in_chans'):
            dynamic_in_chans = dataset.in_chans
            print(f"[Main] ✅ Detected input channels from dataset: {dynamic_in_chans}")
        else:
            # 备用方案：加载一个样本检测
            temp_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
            first_batch = next(iter(temp_loader))
            _,_, dynamic_in_chans, _, _ = first_batch['lr_sequence'].shape
            print(f"[Main] ✅ Detected input channels by loading sample: {dynamic_in_chans}")
            del temp_loader, first_batch
        
        # 创建推理数据加载器
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True  # 加速数据传输
        )
    except Exception as e:
        print(f"❌ CRITICAL: Failed to create data loader. Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # --- 步骤2：初始化推理器 ---
    try:
        predictor = Predictor(configs, args.ckpt_path, dynamic_in_chans)
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize predictor. Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # --- 步骤3：批量推理并保存 ---
    pbar = tqdm(loader, desc="Running inference on dataset")
    for data_batch in pbar:
        try:
            # 执行推理
            predictions = predictor.run_on_batch(data_batch)
            # import pdb;pdb.set_trace()
            
            # 构造输出路径（保留原文件名，添加pred前缀）
            original_hr_path = data_batch['path'][0] if 'path' in data_batch else None
            # if original_hr_path is None:
            #     print("[Warning] No path found in data batch, skip saving")
            #     continue
            
            output_filename = f"pred_{Path(original_hr_path).name}" if original_hr_path is not None else "pred.tif"
            output_path = output_dir / output_filename
            
            # 保存预测结果
            predictor.save_prediction(predictions, original_hr_path, output_path)
            
        except Exception as e:
            print(f"\n⚠️ WARNING: Failed to process batch: {e}")
            continue
            
    print("\n" + "="*80, "\n🎉 Inference complete!", f"\nResults saved to: {args.output_dir}", "\n" + "="*80)

# ==============================================================================
# 3. 独立运行入口 (适配直接调用)
# ==============================================================================
def main_standalone():
    """
    独立运行推理脚本的入口（直接执行 python inference.py）
    """
    parser = argparse.ArgumentParser(description="Standalone Inference Script for AnytimeSwinIR")
    # 核心参数（与main.py的test模式对齐）
    parser.add_argument('--config', type=str, required=True, help="Path to config file (.yaml)")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to checkpoint file (.pth)")
    parser.add_argument('--input', type=str, required=True, help="Input data directory (contains LR/HR subdirs)")
    parser.add_argument('--output', type=str, required=True, help="Output directory to save predictions")
    args = parser.parse_args()
    
    # 加载配置
    configs = OmegaConf.load(args.config)
    
    # 构造兼容run_inference的args对象
    fake_main_args = argparse.Namespace(
        ckpt_path=args.ckpt,
        input_dir=args.input,
        output_dir=args.output
    )
    
    # 调用统一的推理逻辑
    run_inference(fake_main_args, configs)

# ==============================================================================
# 4. 入口函数适配
# ==============================================================================
if __name__ == '__main__':
    # 直接运行时调用独立入口
    main_standalone()
