# 文件路径: datapipe/datasets.py
# (v_anytime - 实现了Anytime理念的全新数据集)

import os
import re
import torch
import rasterio
import numpy as np
from pathlib import Path
from utils import util_common # 假设您有这个工具文件
from datetime import datetime
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
# 2. 升级版的 AnytimeTemporalDataset
# ==============================================================================

class AnytimeTemporalDataset(torch.utils.data.Dataset):
    """
    一个支持不规则、异步时间序列的数据集。
    - 【新】根据config中的'features'开关，动态注入时间/掩膜特征。
    - 【改】将所有特征统一拼接成一个'lr_sequence'张量，以适配SwinIR。
    - 【保】保留了动态查找时相、按瓦片ID分组的核心逻辑。
    """
    def __init__(self, **params):
        self.hr_dir = Path(params['hr_dir'])
        self.lr_dir = Path(params.get('lr_dir', None))
        
        # --- 【核心改造】从总配置中读取特征开关 ---
        parent_configs = params.get('parent_configs', {})
        self.features_config = parent_configs.get('features', {})
        self.use_time_band = self.features_config.get('time_band', {}).get('enabled', False)
        self.use_mask_band = self.features_config.get('mask_band', {}).get('enabled', False)
        
        self.need_path = params.get('need_path', False)
        
        # 扫描HR目录以确定所有目标瓦片ID (保留您的逻辑)
        self.tile_ids = sorted(list({re.search(r'tile_(\d+_\d+)\.tif', f.name).group(1) 
                                     for f in self.hr_dir.glob('*_tile_*.tif')}))
        
        # (保留 sample_num 逻辑)
        sample_num = params.get('sample_num', None)
        if sample_num and 0 < sample_num < len(self.tile_ids):
            self.tile_ids = self.tile_ids[:sample_num]
        
        print(f"[Dataset INFO] Initialized for {len(self.tile_ids)} tile locations.")
        print(f"  - Time Feature Injection: {'Enabled' if self.use_time_band else 'Disabled'}")
        print(f"  - Mask Feature Injection: {'Enabled' if self.use_mask_band else 'Disabled'}")

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, index):
        tile_id = self.tile_ids[index]
        
        lr_files = sorted(self.lr_dir.glob(f"*_tile_{tile_id}.tif"))
        if not lr_files:
            raise FileNotFoundError(f"No LR images for tile_id {tile_id}")

        # 加载HR (保留您的逻辑)
        hr_fname_pattern = lr_files[0].name
        target_hr_path = self.hr_dir / hr_fname_pattern
        with rasterio.open(target_hr_path) as src:
            hr_img = src.read()
        hr_tensor = torch.from_numpy(robust_per_image_normalize(hr_img)).float()

        processed_lr_timesteps = []
        for lr_path in lr_files:
            with rasterio.open(lr_path) as src:
                reflectance_data = src.read() # (C, H, W)
                h, w = src.height, src.width
            
            # --- 【核心改造】根据开关，动态拼接特征 ---
            # 1. 基础特征：归一化后的反射率
            normalized_reflectance = robust_per_image_normalize(reflectance_data)
            features_to_concat = [normalized_reflectance]

            # 2. (可选) 添加“时间波段”
            if self.use_time_band:
                date_str = lr_path.name.split('_')[0]
                dt_object = datetime.strptime(date_str, '%Y%m%d')
                day_of_year = dt_object.timetuple().tm_yday
                normalized_doy = (day_of_year / 366.0) * 2.0 - 1.0
                time_band = np.full((1, h, w), normalized_doy, dtype=np.float32)
                features_to_concat.append(time_band)

            # 3. (可选) 添加“掩膜波段”
            if self.use_mask_band:
                # 您的下载脚本保证了云区像素值为0
                mask_band = (reflectance_data[0:1, :, :] > 0).astype(np.float32)
                mask_band = (mask_band * 2.0) - 1.0 # 映射到 [-1, 1]
                features_to_concat.append(mask_band)
            
            # 拼接成一个时相的完整数据
            full_timestep_data = np.concatenate(features_to_concat, axis=0)
            processed_lr_timesteps.append(torch.from_numpy(full_timestep_data).float())
        
        lr_sequence = torch.stack(processed_lr_timesteps, dim=0)  # (T, C, H, W)

        # 【改】保留时序维度 (T, C, H, W)，并添加timestamps
        timestamps = []
        for lr_path in lr_files:
            date_str = lr_path.name.split('_')[0]
            dt_object = datetime.strptime(date_str, '%Y%m%d')
            day_of_year = dt_object.timetuple().tm_yday
            timestamps.append(day_of_year)

        timestamps = torch.tensor(timestamps, dtype=torch.float32)  # (T,)

        # 返回模型期望的格式
        sample = {'lr_sequence': lr_sequence, 'timestamps': timestamps, 'gt': hr_tensor}
        
        if self.need_path:
            sample['path'] = str(target_hr_path)
            
        return sample

# ==============================================================================
# 3. 强化的数据集创建函数
# ==============================================================================

def create_dataset(configs, parent_configs=None):
    """
    (强化) 创建数据集实例，并确保将父配置传递下去。
    """
    target_class_str = configs['target']
    target_class = util_common.get_obj_from_str(target_class_str)
    params = dict(configs.get('params', {}))
    
    # 【核心】将总的configs对象作为parent_configs传递，以便Dataset能访问到features开关
    params['parent_configs'] = parent_configs
    
    return target_class(**params)
