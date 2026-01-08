# 文件路径: datapipe/datasets.py
# (v_final_definitive_debug_ready - 最终版，已添加debug模式)

import os
import glob
import torch
import rasterio
import numpy as np
from pathlib import Path
from utils import util_common

def robust_per_image_normalize(image_numpy):
    """
    对单张图像进行健壮的、动态的 Min-Max 归一化。
    """
    image_float = image_numpy.astype(np.float32)
    min_val = np.min(image_float, axis=(1, 2), keepdims=True)
    max_val = np.max(image_float, axis=(1, 2), keepdims=True)
    range_val = max_val - min_val
    range_val[range_val < 1e-6] = 1.0
    image_norm_01 = (image_float - min_val) / range_val
    image_norm_neg1_1 = image_norm_01 * 2.0 - 1.0
    return image_norm_neg1_1

class PreprocessedTileDataset(torch.utils.data.Dataset):
    """
    最终版Dataset：
    1. 为LR和HR使用各自正确的归一化。
    2. 支持 debug 模式，用于加载单张图片。
    """
    def __init__(self, **params):
        # --- 【核心修改】添加 debug 模式支持 ---
        self.is_debug = params.get('is_debug', False)
        if self.is_debug:
            self.lr_files = [params['debug_lr_path']]
            # 自动推断HR路径，确保跨平台兼容
            hr_path = Path(params['debug_lr_path']).parent.parent / 'HR' / Path(params['debug_lr_path']).name
            self.hr_files = [str(hr_path)]
            print(f"[Dataset INFO] Initialized in DEBUG mode for 1 image.")
        else:
            # 您原始的、用于训练/验证/测试的目录扫描逻辑
            lr_dir = params['lr_dir']
            self.lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.tif')))
            self.hr_dir = params['hr_dir']
            
            sample_num = params.get('sample_num', None)
            if sample_num and sample_num > 0 and sample_num < len(self.lr_files):
                self.lr_files = self.lr_files[:sample_num]
            
            print(f"[Dataset INFO] Initialized in NORMAL mode. Found {len(self.lr_files)} images in '{lr_dir}'.")
        
        self.need_path = params.get('need_path', False)
        self.scale_factor = params.get('scale_factor', 3)


    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, index):
        lr_path = self.lr_files[index]
        
        # 在debug模式下，hr_files[0]就是对应的hr_path
        # 在正常模式下，需要拼接
        if self.is_debug:
            hr_path = self.hr_files[0]
        else:
            hr_path = os.path.join(self.hr_dir, os.path.basename(lr_path))
        
        try:
            with rasterio.open(lr_path) as src: lr_img = src.read()
        except rasterio.errors.RasterioIOError:
            # 提供更明确的错误信息
            raise FileNotFoundError(f"Dataset Error: Cannot open LR image at path: {lr_path}")

        try:
            with rasterio.open(hr_path) as src: hr_img = src.read()
        except rasterio.errors.RasterioIOError:
            # 兼容HR文件可能不存在的情况（例如，只对LR进行推理）
            # 创建一个与LR放大后尺寸相同的假HR
            print(f"Warning: HR image not found at {hr_path}. Creating a dummy HR tensor.")
            c, h, w = lr_img.shape
            hr_img = np.zeros((c, h * self.scale_factor, w * self.scale_factor), dtype=np.float32)

        # --- 归一化 (与您原版逻辑完全一致) ---
        lr_normalized = robust_per_image_normalize(lr_img)
        lr_tensor = torch.from_numpy(lr_normalized).float()
        
        hr_tensor = torch.from_numpy(hr_img.astype(np.float32)) / 127.5 - 1.0
        
        # --- 输入插值 (与您原版逻辑完全一致) ---
        input_tensor = torch.nn.functional.interpolate(
            lr_tensor.unsqueeze(0), 
            size=(hr_tensor.shape[1], hr_tensor.shape[2]), 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)
        
        # --- 组装样本 (使用 s1/gt 键名) ---
        sample = {
            's1': input_tensor.clamp(-1.0, 1.0), 
            'gt': hr_tensor.clamp(-1.0, 1.0),
        }
        if self.need_path:
            sample['path'] = str(lr_path)
            
        return sample

def create_dataset(configs):
    # (此函数与您原版完全一致，无需修改)
    target_class = util_common.get_obj_from_str(configs['target'])
    return target_class(**configs.get('params', {}))
