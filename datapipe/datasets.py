# 文件路径: datapipe/datasets.py
# (v_anytime - 实现了Anytime理念的全新数据集)

import os
import re
import torch
import rasterio
import numpy as np
from pathlib import Path
from utils import util_common # 假设您有这个工具文件

# ==============================================================================
# 1. 辅助函数 (新增时间戳编码 & 保留归一化)
# ==============================================================================

def get_timestamp_encoding(timestamps, encoding_dim=64):
    """
    为一批时间戳（例如，年内日）生成正弦/余弦位置编码。
    timestamps: 1D张量，包含需要编码的时间戳。
    encoding_dim: 编码向量的维度。
    """
    if encoding_dim % 2 != 0:
        raise ValueError(f"Encoding dimension must be an even number, but got {encoding_dim}")
    
    position = timestamps.unsqueeze(1)
    div_term = torch.exp(torch.arange(0, encoding_dim, 2, device=timestamps.device).float() * -(np.log(10000.0) / encoding_dim))
    
    pe = torch.zeros(len(timestamps), encoding_dim, device=timestamps.device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

def robust_per_image_normalize(img, lo=1, hi=99):
    """
    对每个通道独立使用其1%和99%分位点进行拉伸，然后归一化到[-1, 1]。
    img: 输入的Numpy数组，形状为 (C, H, W)。
    """
    img_float = img.astype(np.float32)
    normalized_bands = []
    for i in range(img_float.shape[0]):
        band = img_float[i]
        # 跳过全零的通道
        if np.all(band == 0):
            normalized_bands.append(band)
            continue
        
        lo_p, hi_p = np.percentile(band, (lo, hi))
        
        # 防止分母为零
        if hi_p - lo_p < 1e-6:
            normalized_band = np.zeros_like(band)
        else:
            clipped_band = np.clip(band, lo_p, hi_p)
            normalized_band = (clipped_band - lo_p) / (hi_p - lo_p)
        
        normalized_bands.append(normalized_band * 2 - 1) # 缩放到 [-1, 1]
        
    return np.stack(normalized_bands, axis=0)

# ==============================================================================
# 2. 全新的 AnytimeTemporalDataset
# ==============================================================================

class AnytimeTemporalDataset(torch.utils.data.Dataset):
    """
    一个支持不规则、异步多模态时间序列的数据集。
    - 动态发现每个瓦片可用的时相。
    - 引入时间戳编码。
    - 支持SAR数据融合。
    - 不在数据加载时进行上采样。
    """
    def __init__(self, **params):
        self.hr_dir = Path(params['hr_dir'])
        self.lr_dir = Path(params.get('lr_dir', None))
        self.sar_dir = Path(params.get('sar_dir', None))
        self.use_sar = params.get('use_sar', False)
        
        self.need_path = params.get('need_path', False)
        self.num_lr_bands = params.get('num_lr_bands', 9)
        self.num_sar_bands = params.get('num_sar_bands', 2)
        # 1. 扫描HR目录以确定所有目标瓦片ID
        self.tile_ids = sorted(list({re.search(r'tile_(\d+_\d+)\.tif', f.name).group(1) for f in self.hr_dir.glob('*_tile_*.tif')}))
        
        sample_num = params.get('sample_num', None)
        if sample_num and 0 < sample_num < len(self.tile_ids):
            self.tile_ids = self.tile_ids[:sample_num]
        print(f"[Dataset INFO] Initialized AnytimeTemporalDataset for {len(self.tile_ids)} tile locations.")
        if self.lr_dir: print(f"  - Optical LR source: {self.lr_dir}")
        if self.use_sar: print(f"  - SAR source: {self.sar_dir}")

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, index):
        tile_id = self.tile_ids[index]
        
        # 1. 动态查找该瓦片的所有时相文件
        lr_files = sorted(self.lr_dir.glob(f"*_tile_{tile_id}.tif"))
        if not lr_files:
            raise FileNotFoundError(f"No LR images found for tile_id {tile_id} in {self.lr_dir}")
            
        # 假设HR文件名与LR时相中的一个匹配（或有特定规则）
        # 这里简化为使用第一个LR文件的日期来找HR
        hr_fname_pattern = lr_files[0].name
        target_hr_path = self.hr_dir / hr_fname_pattern
        if not target_hr_path.exists():
            raise FileNotFoundError(f"HR file not found for tile_id {tile_id}: {target_hr_path}")

        # 2. 加载目标HR影像 (Ground Truth)
        with rasterio.open(target_hr_path) as src:
            hr_img = src.read()
        hr_tensor = torch.from_numpy(robust_per_image_normalize(hr_img)).float()

        # 3. 动态加载该瓦片所有可用的LR时相
        lr_tensors = []
        timestamps = []
        
        for lr_path in lr_files:
            with rasterio.open(lr_path) as src:
                lr_img = src.read() # 假设波段数已在prepare阶段统一
            
            lr_normalized = robust_per_image_normalize(lr_img)
            lr_tensors.append(torch.from_numpy(lr_normalized).float())
            
            # 从文件名解析日期并转换为年内日
            date_str = lr_path.name.split('_')[0]
            day_of_year = int(date_str[4:6]) * 30 + int(date_str[6:8]) # 简化版，建议用datetime
            timestamps.append(day_of_year)

        # (T, C, H, W)
        lr_sequence = torch.stack(lr_tensors, dim=0)
        time_tensor = torch.tensor(timestamps, dtype=torch.float32)
        
        # 4. (可选) 加载SAR数据
        # ... (此处省略SAR加载逻辑，可以仿照LR加载)

        # 5. 组装样本
        sample = {
            'lr_sequence': lr_sequence,        # (T_lr, C_lr, H, W)
            'timestamps': time_tensor,         # (T_lr,)
            'gt': hr_tensor                    # (C_hr, H_hr, W_hr)
        }
            
        if self.need_path:
            sample['path'] = str(target_hr_path)
            
        return sample

# ==============================================================================
# 3. 数据集创建函数 (保持不变，但现在会创建新版Dataset)
# ==============================================================================

def create_dataset(configs, parent_configs=None):
    target_class_str = configs['target']
    target_class = util_common.get_obj_from_str(target_class_str)
    params = dict(configs.get('params', {}))
    return target_class(**params)

