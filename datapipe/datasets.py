# 文件路径: datapipe/datasets.py
# (v_final_definitive_debug_ready - 最终版，已添加debug模式)

import os
import glob
import torch
import rasterio
import numpy as np
from pathlib import Path
from utils import util_common
import re 

# ==============================================================================
# 1. 辅助函数：健壮的归一化
# ==============================================================================

def robust_per_image_normalize(img, lo=1, hi=99):
    """
    对每个通道独立使用其1%和99%分位点进行拉伸，然后归一化到[-1, 1]。
    img: 输入的Numpy数组，形状为 (C, H, W)。
    """
    # 计算每个通道的分位点
    img_flat = img.reshape(img.shape[0], -1)
    lo_p, hi_p = np.percentile(img_flat, (lo, hi), axis=1)
    
    # 防止出现分母为零的情况
    den = hi_p - lo_p
    den[den == 0] = 1e-6
    
    # 使用numpy的广播机制进行高效计算
    lo_p = lo_p[:, np.newaxis, np.newaxis]
    den = den[:, np.newaxis, np.newaxis]
    
    normalized = (img.astype(np.float32) - lo_p) / den
    return np.clip(normalized * 2 - 1, -1.0, 1.0)

# ==============================================================================
# 2. 最终版时序数据集
# ==============================================================================

class TemporalTileDataset(torch.utils.data.Dataset):
    """
    最终版时序数据集:
    1. 动态发现所有时相，并对缺失的影像进行零填充。
    2. 对每张LR影像独立进行健壮的归一化。
    3. 兼容Debug模式，用于单张瓦片的推理。
    """
    def __init__(self, **params):
        # 1. 初始化基本参数
        self.lr_dir = Path(params['lr_dir'])
        self.hr_dir = Path(params['hr_dir'])
        self.need_path = params.get('need_path', False)
        self.scale_factor = params.get('scale_factor', 3)
        self.num_lr_bands = params['num_lr_bands']
        self.is_debug = params.get('is_debug', False)
        
        # 2. 初始化 CCDC 配置（如果启用）
        self.ccdc_config = params.get('ccdc_config', None)
        self.use_ccdc = self.ccdc_config is not None and self.ccdc_config.get('enabled', False)
        if self.use_ccdc:
            # 确定 CCDC 特征目录（相对于 lr_dir）
            ccdc_base_dir = Path(self.ccdc_config.get('ccdc_dir', './data/ccdc_features'))
            # 从 lr_dir 推断是 train/val/test
            lr_dir_str = str(self.lr_dir)
            if 'train' in lr_dir_str:
                self.ccdc_dir = ccdc_base_dir / 'train' / 'LR'
            elif 'val' in lr_dir_str:
                self.ccdc_dir = ccdc_base_dir / 'val' / 'LR'
            elif 'test' in lr_dir_str:
                self.ccdc_dir = ccdc_base_dir / 'test' / 'LR'
            else:
                # 默认使用与 lr_dir 相同的相对路径
                self.ccdc_dir = ccdc_base_dir / Path(self.lr_dir).relative_to(Path(self.lr_dir).parents[2]) if len(Path(self.lr_dir).parts) > 2 else ccdc_base_dir
            
            # 计算 CCDC 特征通道数
            bands_to_process = self.ccdc_config.get('bands_to_process', {})
            coeffs_to_extract = self.ccdc_config.get('coeffs_to_extract', [])
            self.ccdc_num_channels = len(bands_to_process) * len(coeffs_to_extract)
            print(f"[Dataset INFO] CCDC features enabled. Channels: {self.ccdc_num_channels}, Directory: {self.ccdc_dir}")
        else:
            self.ccdc_num_channels = 0

        # 2. 扫描整个LR目录，发现所有唯一的日期（时相）
        # 无论在哪种模式下，这都是必需的，以确保通道数一致
        all_lr_files_scan = list(self.lr_dir.glob('*.tif'))
        all_dates = sorted(list({
            match.group(1) for f in all_lr_files_scan 
            if (match := re.search(r'^(\d{8})_', f.name))
        }))
        self.sorted_dates = all_dates
        
        # 3. 构建从 瓦片ID -> {日期: 文件路径} 的完整映射
        # 这也是必需的，因为 __getitem__ 需要查找一个瓦片的所有时序 "兄弟"
        self.tile_to_date_map = {}
        for f_path in all_lr_files_scan:
            if (match := re.search(r'tile_(\d+_\d+)\.tif', f_path.name)) and \
               (date_match := re.search(r'^(\d{8})_', f_path.name)):
                tile_id, date_str = match.group(1), date_match.group(1)
                if tile_id not in self.tile_to_date_map:
                    self.tile_to_date_map[tile_id] = {}
                self.tile_to_date_map[tile_id][date_str] = f_path

        # 4. 根据模式（Debug/Normal）确定要处理的目标文件列表
        if self.is_debug:
            # Debug模式：只处理指定的一张瓦片
            debug_lr_path = Path(params['debug_lr_path'])
            # 自动推断对应的HR路径
            debug_hr_path = self.hr_dir / debug_lr_path.name
            self.target_files = [debug_hr_path]
            print(f"[Dataset INFO] Initialized in DEBUG mode for 1 image.")
            print(f"      -> Target HR: {debug_hr_path.name}")
        else:
            # Normal模式：处理HR目录下的所有（或部分）文件
            self.target_files = sorted(list(self.hr_dir.glob('*.tif')))
            sample_num = params.get('sample_num', None)
            if sample_num and sample_num > 0 and sample_num < len(self.target_files):
                self.target_files = self.target_files[:sample_num]
            print(f"[Dataset INFO] Initialized in NORMAL mode. Found {len(self.target_files)} target HR images.")

        print(f"      -> Found {len(self.sorted_dates)} unique time steps (dates).")
        base_channels = len(self.sorted_dates) * self.num_lr_bands
        total_channels = base_channels + self.ccdc_num_channels
        print(f"      -> Model `in_chans` should be: {base_channels} (temporal) + {self.ccdc_num_channels} (CCDC) = {total_channels}")

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self, index):
        # 1. 获取目标HR路径并解析瓦片ID
        target_hr_path = self.target_files[index]
        fname = target_hr_path.name
        
        match = re.search(r'tile_(\d+_\d+)\.tif', fname)
        if not match: raise ValueError(f"Can't parse tile_id from {fname}")
        tile_id = match.group(1)

        # 2. 加载目标HR影像
        try:
            with rasterio.open(target_hr_path) as src:
                hr_img = src.read()
            hr_tensor = torch.from_numpy(hr_img.astype(np.float32)) / 127.5 - 1.0
        except Exception as e:
            raise IOError(f"Failed to read HR file {target_hr_path}: {e}")

        # 3. 构建LR时序堆栈 (带填充)
        lr_stack = []
        date_path_map = self.tile_to_date_map.get(tile_id, {})
        
        # 拿一个存在的LR影像获取尺寸，用于创建零填充张量
        if not date_path_map: raise ValueError(f"No LR images found for tile_id {tile_id}")
        with rasterio.open(next(iter(date_path_map.values()))) as src:
            h, w = src.height, src.width
            
        for date in self.sorted_dates:
            lr_path = date_path_map.get(date)
            if lr_path:
                with rasterio.open(lr_path) as src:
                    lr_img = src.read()
                
                # 裁剪或填充波段以确保一致
                if lr_img.shape[0] != self.num_lr_bands:
                    lr_img_new = np.zeros((self.num_lr_bands, lr_img.shape[1], lr_img.shape[2]), dtype=lr_img.dtype)
                    min_bands = min(self.num_lr_bands, lr_img.shape[0])
                    lr_img_new[:min_bands,:,:] = lr_img[:min_bands,:,:]
                    lr_img = lr_img_new

                lr_normalized = robust_per_image_normalize(lr_img)
                lr_tensor = torch.from_numpy(lr_normalized).float()
                lr_stack.append(lr_tensor)
            else:
                # 核心：如果当天影像不存在，则用零填充
                lr_stack.append(torch.zeros((self.num_lr_bands, h, w), dtype=torch.float32))

        # 4. 沿通道维度拼接
        stacked_lr_tensor = torch.cat(lr_stack, dim=0)

        # 5. 加载 CCDC 特征（如果启用）
        if self.use_ccdc:
            # 查找对应的 CCDC 特征文件
            # CCDC 文件名格式: {tile_id}_ccdc_features.tif (与 ccdc_class.py 中的保存格式一致)
            ccdc_filename = f"{tile_id}_ccdc_features.tif"
            ccdc_path = self.ccdc_dir / ccdc_filename
            
            if ccdc_path.exists():
                try:
                    with rasterio.open(ccdc_path) as src:
                        ccdc_features = src.read()  # (C, H, W)
                    # 转换为 torch tensor 并归一化（CCDC 特征通常在合理范围内，但需要归一化到 [-1, 1]）
                    ccdc_tensor = torch.from_numpy(ccdc_features.astype(np.float32))
                    # 对每个通道进行归一化（使用分位数归一化）
                    for c in range(ccdc_tensor.shape[0]):
                        channel_data = ccdc_tensor[c]
                        p1, p99 = torch.quantile(channel_data, torch.tensor([0.01, 0.99]))
                        if p99 > p1:
                            ccdc_tensor[c] = torch.clamp((channel_data - p1) / (p99 - p1) * 2 - 1, -1.0, 1.0)
                        else:
                            ccdc_tensor[c] = torch.zeros_like(channel_data)
                    
                    # 插值到 HR 尺寸
                    ccdc_tensor = torch.nn.functional.interpolate(
                        ccdc_tensor.unsqueeze(0),
                        size=(hr_tensor.shape[1], hr_tensor.shape[2]),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0)
                    
                    # 与原始时序数据拼接
                    stacked_lr_tensor = torch.cat([stacked_lr_tensor, ccdc_tensor], dim=0)
                except Exception as e:
                    print(f"[WARNING] Failed to load CCDC features from {ccdc_path}: {e}. Using zero padding.")
                    # 如果加载失败，使用零填充
                    ccdc_tensor = torch.zeros((self.ccdc_num_channels, h, w), dtype=torch.float32)
                    ccdc_tensor = torch.nn.functional.interpolate(
                        ccdc_tensor.unsqueeze(0),
                        size=(hr_tensor.shape[1], hr_tensor.shape[2]),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0)
                    stacked_lr_tensor = torch.cat([stacked_lr_tensor, ccdc_tensor], dim=0)
            else:
                # 如果 CCDC 文件不存在，使用零填充
                print(f"[WARNING] CCDC feature file not found: {ccdc_path}. Using zero padding.")
                ccdc_tensor = torch.zeros((self.ccdc_num_channels, h, w), dtype=torch.float32)
                ccdc_tensor = torch.nn.functional.interpolate(
                    ccdc_tensor.unsqueeze(0),
                    size=(hr_tensor.shape[1], hr_tensor.shape[2]),
                    mode='bicubic',
                    align_corners=False
                ).squeeze(0)
                stacked_lr_tensor = torch.cat([stacked_lr_tensor, ccdc_tensor], dim=0)

        # 6. 插值到HR尺寸
        input_tensor = torch.nn.functional.interpolate(
            stacked_lr_tensor.unsqueeze(0), 
            size=(hr_tensor.shape[1], hr_tensor.shape[2]), 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)
        
        # 7. 组装样本
        sample = {'s1': input_tensor, 'gt': hr_tensor.clamp(-1.0, 1.0)}
        if self.need_path:
            sample['path'] = str(target_hr_path)
            
        return sample

# ==============================================================================
# 3. 数据集创建函数
# ==============================================================================
def create_dataset(configs, parent_configs=None):
    """
    根据配置动态创建数据集实例。
    
    Args:
        configs: 数据集配置（包含 target 和 params）
        parent_configs: 父级配置对象（可选，用于获取 CCDC 等全局配置）
    """
    target_class_str = configs['target']
    # 动态地从字符串获取类定义
    # 假设 'datapipe.datasets.TemporalTileDataset'
    module_name, class_name = target_class_str.rsplit('.', 1)
    
    # 确保当前文件内的类可以被找到
    if module_name == __name__:
        target_class = globals()[class_name]
    else:
        # 如果在其他模块，使用 util_common
        target_class = util_common.get_obj_from_str(target_class_str)
    
    # 合并 params 和 CCDC 配置（如果存在）
    params = dict(configs.get('params', {}))
    if parent_configs and hasattr(parent_configs, 'ccdc') and parent_configs.ccdc.get('enabled', False):
        params['ccdc_config'] = dict(parent_configs.ccdc)
    
    return target_class(**params)


# class PreprocessedTileDataset(torch.utils.data.Dataset):
#     """
#     最终版Dataset：
#     1. 为LR和HR使用各自正确的归一化。
#     2. 支持 debug 模式，用于加载单张图片。
#     """
#     def __init__(self, **params):
#         # --- 【核心修改】添加 debug 模式支持 ---
#         self.is_debug = params.get('is_debug', False)
#         if self.is_debug:
#             self.lr_files = [params['debug_lr_path']]
#             # 自动推断HR路径，确保跨平台兼容
#             hr_path = Path(params['debug_lr_path']).parent.parent / 'HR' / Path(params['debug_lr_path']).name
#             self.hr_files = [str(hr_path)]
#             print(f"[Dataset INFO] Initialized in DEBUG mode for 1 image.")
#         else:
#             # 您原始的、用于训练/验证/测试的目录扫描逻辑
#             lr_dir = params['lr_dir']
#             self.lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.tif')))
#             self.hr_dir = params['hr_dir']
            
#             sample_num = params.get('sample_num', None)
#             if sample_num and sample_num > 0 and sample_num < len(self.lr_files):
#                 self.lr_files = self.lr_files[:sample_num]
            
#             print(f"[Dataset INFO] Initialized in NORMAL mode. Found {len(self.lr_files)} images in '{lr_dir}'.")
        
#         self.need_path = params.get('need_path', False)
#         self.scale_factor = params.get('scale_factor', 3)


#     def __len__(self):
#         return len(self.lr_files)

#     def __getitem__(self, index):
#         lr_path = self.lr_files[index]
        
#         # 在debug模式下，hr_files[0]就是对应的hr_path
#         # 在正常模式下，需要拼接
#         if self.is_debug:
#             hr_path = self.hr_files[0]
#         else:
#             hr_path = os.path.join(self.hr_dir, os.path.basename(lr_path))
        
#         try:
#             with rasterio.open(lr_path) as src: lr_img = src.read()
#         except rasterio.errors.RasterioIOError:
#             # 提供更明确的错误信息
#             raise FileNotFoundError(f"Dataset Error: Cannot open LR image at path: {lr_path}")

#         try:
#             with rasterio.open(hr_path) as src: hr_img = src.read()
#         except rasterio.errors.RasterioIOError:
#             # 兼容HR文件可能不存在的情况（例如，只对LR进行推理）
#             # 创建一个与LR放大后尺寸相同的假HR
#             print(f"Warning: HR image not found at {hr_path}. Creating a dummy HR tensor.")
#             c, h, w = lr_img.shape
#             hr_img = np.zeros((c, h * self.scale_factor, w * self.scale_factor), dtype=np.float32)

#         # --- 归一化 (与您原版逻辑完全一致) ---
#         lr_normalized = robust_per_image_normalize(lr_img)
#         lr_tensor = torch.from_numpy(lr_normalized).float()
        
#         hr_tensor = torch.from_numpy(hr_img.astype(np.float32)) / 127.5 - 1.0
        
#         # --- 输入插值 (与您原版逻辑完全一致) ---
#         input_tensor = torch.nn.functional.interpolate(
#             lr_tensor.unsqueeze(0), 
#             size=(hr_tensor.shape[1], hr_tensor.shape[2]), 
#             mode='bicubic', 
#             align_corners=False
#         ).squeeze(0)
        
#         # --- 组装样本 (使用 s1/gt 键名) ---
#         sample = {
#             's1': input_tensor.clamp(-1.0, 1.0), 
#             'gt': hr_tensor.clamp(-1.0, 1.0),
#         }
#         if self.need_path:
#             sample['path'] = str(lr_path)
            
#         return sample


# def robust_per_image_normalize(image_numpy):
#     """
#     对单张图像进行健壮的、动态的 Min-Max 归一化。
#     """
#     image_float = image_numpy.astype(np.float32)
#     min_val = np.min(image_float, axis=(1, 2), keepdims=True)
#     max_val = np.max(image_float, axis=(1, 2), keepdims=True)
#     range_val = max_val - min_val
#     range_val[range_val < 1e-6] = 1.0
#     image_norm_01 = (image_float - min_val) / range_val
#     image_norm_neg1_1 = image_norm_01 * 2.0 - 1.0
#     return image_norm_neg1_1

# def create_dataset(configs):
#     # (此函数与您原版完全一致，无需修改)
#     target_class = util_common.get_obj_from_str(configs['target'])
#     return target_class(**configs.get('params', {}))
