# === 4. 兼容性：PreprocessedTileDataset（用于旧配置字符串反射） ===
import glob
import torch
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

            self.enable_hr_zero_filter = bool(params.get('enable_hr_zero_filter', True))
            self.hr_zero_ratio_threshold = float(params.get('hr_zero_ratio_filter_threshold', 0.05))
            self.hr_zero_ratio_manifest = params.get('hr_zero_ratio_manifest', 'patch_stats.log')
            if self.enable_hr_zero_filter:
                zero_ratio_map = _load_hr_zero_ratio_manifest(self.hr_zero_ratio_manifest)
                self.lr_files, removed_count, matched_count = _filter_files_by_hr_zero_ratio(
                    self.lr_files,
                    zero_ratio_map,
                    self.hr_zero_ratio_threshold,
                )
                print(
                    f"[Dataset INFO] Pair-zero filter enabled: threshold>{self.hr_zero_ratio_threshold:.2f}, "
                    f"matched={matched_count}, removed={removed_count}, kept={len(self.lr_files)}"
                )
                if matched_count == 0:
                    print(
                        f"[Dataset WARN] No samples matched {self.hr_zero_ratio_manifest}; "
                        "pair-zero filtering was skipped for this dataset."
                    )
            
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
        
        # hr_tensor = torch.from_numpy(hr_img.astype(np.float32)) / 127.5 - 1.0
        hr_tensor = torch.from_numpy(robust_per_image_normalize(hr_img)).float()
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

        for k, v in list(sample.items()):
            if torch.is_tensor(v):
                sample[k] = v.contiguous().clone()
        return sample
# 文件路径: datapipe/datasets.py
# (v_anytime - 实现了Anytime理念的全新数据集)

import os
import re
import random
import time
import torch
import rasterio
import numpy as np
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path
from utils import util_common # 假设您有这个工具文件
from datetime import datetime
from utils.cloud_mask_processor import CloudMaskProcessor

_HR_ZERO_RATIO_CACHE = {}
_PATCH_STATS_LR_TILE_SIZE = 64


def _load_hr_zero_ratio_manifest(manifest_path):
    manifest = Path(manifest_path)
    cache_key = str(manifest.resolve()) if manifest.exists() else str(manifest)
    if cache_key in _HR_ZERO_RATIO_CACHE:
        return _HR_ZERO_RATIO_CACHE[cache_key]

    zero_ratio_map = {}
    if not manifest.exists():
        _HR_ZERO_RATIO_CACHE[cache_key] = zero_ratio_map
        return zero_ratio_map

    line_pattern = re.compile(
        r'^(?P<raw_path>.+?/L8_(?P<path_row>\d{5})_(?P<date>\d{8})_Masked\.tif),'
        r'(?P<r>\d+),(?P<c>\d+),.*?LR\[.*?zero=(?P<lr_zero>[0-9.]+)%\].*?HR\[.*?zero=(?P<hr_zero>[0-9.]+)%\]'
    )

    with manifest.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = line_pattern.search(line)
            if not match:
                continue
            date = match.group('date')
            r = int(match.group('r'))
            c = int(match.group('c'))
            zero_ratio = max(float(match.group('lr_zero')), float(match.group('hr_zero'))) / 100.0

            # 兼容两种文件命名：
            # 1) 原始窗口坐标：date_tile_<r>_<c>.tif
            # 2) 处理后瓦片索引：date_tile_<r//64>_<c//64>.tif
            candidate_keys = {
                f"{date}_tile_{r}_{c}.tif",
                f"{date}_tile_{r // _PATCH_STATS_LR_TILE_SIZE}_{c // _PATCH_STATS_LR_TILE_SIZE}.tif",
            }

            for key in candidate_keys:
                prev = zero_ratio_map.get(key)
                if prev is None or zero_ratio > prev:
                    zero_ratio_map[key] = zero_ratio

    _HR_ZERO_RATIO_CACHE[cache_key] = zero_ratio_map
    return zero_ratio_map


def _filter_files_by_hr_zero_ratio(lr_files, zero_ratio_map, threshold):
    if threshold is None:
        return list(lr_files), 0, 0

    kept_files = []
    removed_count = 0
    matched_count = 0
    for lr_path in lr_files:
        key = Path(lr_path).name
        zero_ratio = zero_ratio_map.get(key)
        if zero_ratio is None:
            kept_files.append(lr_path)
            continue

        matched_count += 1
        if zero_ratio > threshold:
            removed_count += 1
            continue
        kept_files.append(lr_path)

    return kept_files, removed_count, matched_count
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


def _safe_resize_mask(mask_hw, target_h, target_w):
    """将掩膜 resize 到目标大小，保持 [0,1] 范围。"""
    if mask_hw.shape[0] == target_h and mask_hw.shape[1] == target_w:
        return np.clip(mask_hw.astype(np.float32), 0.0, 1.0)

    tensor = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False,
    )
    return np.clip(resized.squeeze(0).squeeze(0).numpy().astype(np.float32), 0.0, 1.0)


def _normalize_softmask_values(mask_raw):
    """软掩膜值统一到 [0,1]。

    支持常见口径：
    1. 已经是 [0,1]
    2. 四档概率等级 [0,1,2,3] -> [0,0.33,0.67,1]
    3. 其他情况按最大值归一化
    """
    m = mask_raw.astype(np.float32)
    if np.isnan(m).any():
        m = np.nan_to_num(m, nan=0.0)

    m_min = float(np.min(m))
    m_max = float(np.max(m))
    if m_max <= 1.0 and m_min >= 0.0:
        return np.clip(m, 0.0, 1.0)

    # EarthEngine 常见置信度四档 0/1/2/3
    if m_min >= 0.0 and m_max <= 3.0:
        return np.clip(m / 3.0, 0.0, 1.0)

    if m_max - m_min < 1e-6:
        return np.zeros_like(m, dtype=np.float32)

    m = (m - m_min) / (m_max - m_min)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def _extract_softmask_from_lr_if_possible(reflectance_data):
    """尝试从 LR 文件末波段提取软掩膜。

    若末波段满足掩膜特征（范围接近 0~1 或 0~3，且离散层级较少），返回归一化掩膜；
    否则返回 None。
    """
    if reflectance_data.shape[0] < 1:
        return None

    candidate = reflectance_data[-1].astype(np.float32)
    c_min = float(np.min(candidate))
    c_max = float(np.max(candidate))
    if c_max < 1e-6:
        return None

    # 允许少量噪声，限制独特值数量，避免把反射率错当掩膜
    uniq = np.unique(np.round(candidate, 2))
    if uniq.size > 32:
        return None

    # 仅接受看起来像掩膜的范围
    if not (0.0 <= c_min <= 3.0 and 0.0 < c_max <= 3.0):
        return None

    return _normalize_softmask_values(candidate)


def _pad_or_truncate_temporal_sample(sample, fixed_temporal_len):
    if fixed_temporal_len is None:
        return sample

    lr_sequence = sample.get('lr_sequence')
    if not torch.is_tensor(lr_sequence):
        return sample

    current_len = int(lr_sequence.shape[0])
    if current_len == fixed_temporal_len:
        return sample

    def _adjust_time_tensor(tensor, pad_value=0.0):
        if not torch.is_tensor(tensor):
            return tensor
        if tensor.shape[0] >= fixed_temporal_len:
            return tensor[:fixed_temporal_len].contiguous()

        pad_shape = (fixed_temporal_len - tensor.shape[0],) + tuple(tensor.shape[1:])
        pad_tensor = tensor.new_full(pad_shape, pad_value)
        return torch.cat([tensor, pad_tensor], dim=0).contiguous()

    padded = dict(sample)
    for key in ('lr_sequence', 'timestamps', 'mask', 'mask_prob', 'indicating_mask'):
        if key in padded and torch.is_tensor(padded[key]):
            padded[key] = _adjust_time_tensor(padded[key], pad_value=0.0)

    return padded

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
        print("[Dataset INFO] AnytimeTemporalDataset __init__: begin")
        self.hr_dir = Path(params['hr_dir'])
        self.lr_dir = Path(params.get('lr_dir', None))
        
        # --- 【核心改造】从总配置中读取特征开关 ---
        parent_configs = params.get('parent_configs', {})
        self.features_config = parent_configs.get('features', {})
        self.use_time_band = self.features_config.get('time_band', {}).get('enabled', False)
        self.use_mask_band = self.features_config.get('mask_band', {}).get('enabled', False)

        self.mask_config = self.features_config.get('mask_band', {})
        self.mask_processor_config = self.mask_config.get('processor', {})
        self.mask_type = str(self.mask_processor_config.get('mask_type', 'hard')).lower()
        self.use_advanced_processor = bool(self.mask_config.get('use_advanced_processor', False))
        self.cloud_mask_processor = None

        if self.use_mask_band and self.use_advanced_processor:
            self.cloud_mask_processor = CloudMaskProcessor(
                mask_type=self.mask_type,
                soft_mask_sigma=float(self.mask_processor_config.get('soft_mask_sigma', 2.0)),
                use_spatial_smoothing=bool(self.mask_processor_config.get('use_spatial_smoothing', False)),
                smoothing_sigma=float(self.mask_processor_config.get('smoothing_sigma', 1.0)),
                valid_pixel_ratio_threshold=float(self.mask_processor_config.get('valid_pixel_ratio_threshold', 0.05)),
            )

        # 可选外部 soft mask 目录（若存在则优先使用）
        self.cloud_mask_dir = params.get('cloud_mask_dir', self.mask_config.get('cloud_mask_dir', ''))
        self.cloud_mask_pattern = params.get(
            'cloud_mask_pattern',
            self.mask_config.get('cloud_mask_pattern', '{date}_tile_{tile_id}_SoftMask.tif')
        )
        self.cloud_mask_dir = Path(self.cloud_mask_dir) if self.cloud_mask_dir else None
        
        self.need_path = params.get('need_path', False)
        self.fixed_temporal_len = int(params.get('fixed_temporal_len', 20))
        self.cache_dir = Path(params['cache_dir']) if params.get('cache_dir') else None
        self.use_tile_cache = bool(params.get('use_tile_cache', self.cache_dir is not None))
        self.tile_cache_mem_size = int(params.get('tile_cache_mem_size', 64))
        self.slow_sample_warn_sec = float(params.get('slow_sample_warn_sec', 1.5))
        self.max_timesteps_per_sample = int(params.get('max_timesteps_per_sample', 0))
        self._tile_cache_mem = OrderedDict()

        self.enable_hr_zero_filter = bool(params.get('enable_hr_zero_filter', True))
        self.hr_zero_ratio_threshold = float(params.get('hr_zero_ratio_filter_threshold', 0.05))
        self.hr_zero_ratio_manifest = params.get('hr_zero_ratio_manifest', 'patch_stats.log')
        self.pre_filter_sample_num = params.get('pre_filter_sample_num', None)
        self.pre_filter_seed = int(params.get('pre_filter_seed', 0))
        self._pair_zero_ratio_map = _load_hr_zero_ratio_manifest(self.hr_zero_ratio_manifest) if self.enable_hr_zero_filter else {}
        self._tile_to_lr_files = defaultdict(list)

        # 关键性能优化：仅扫描一次 LR 目录并建立 tile -> files 索引。
        print("[Dataset INFO] Building LR file index (single pass)...")
        for lr_path in self.lr_dir.glob('*_tile_*.tif'):
            m = re.search(r'tile_(\d+_\d+)\.tif', lr_path.name)
            if m is None:
                continue
            self._tile_to_lr_files[m.group(1)].append(lr_path)
        for k in list(self._tile_to_lr_files.keys()):
            self._tile_to_lr_files[k].sort()
        print(f"[Dataset INFO] LR index done: {len(self._tile_to_lr_files)} tiles indexed")
        
        # 扫描HR目录以确定所有目标瓦片ID (保留您的逻辑)
        self.tile_ids = sorted(list({re.search(r'tile_(\d+_\d+)\.tif', f.name).group(1) 
                                     for f in self.hr_dir.glob('*_tile_*.tif')}))

        if self.enable_hr_zero_filter:
            if self.pre_filter_sample_num and 0 < self.pre_filter_sample_num < len(self.tile_ids):
                rng = random.Random(self.pre_filter_seed)
                sampled_tile_ids = list(self.tile_ids)
                rng.shuffle(sampled_tile_ids)
                self.tile_ids = sampled_tile_ids[:self.pre_filter_sample_num]
                print(
                    f"[Dataset INFO] Pre-filter random sample enabled: seed={self.pre_filter_seed}, "
                    f"subset_tiles={len(self.tile_ids)}"
                )

            kept_tile_ids = []
            removed_tile_count = 0
            matched_count = 0
            removed_count = 0
            for tile_id in self.tile_ids:
                all_lr_files = self._tile_to_lr_files.get(tile_id, [])
                filtered_lr_files, removed_one, matched_one = _filter_files_by_hr_zero_ratio(
                    all_lr_files,
                    self._pair_zero_ratio_map,
                    self.hr_zero_ratio_threshold,
                )
                matched_count += matched_one
                removed_count += removed_one
                if filtered_lr_files:
                    kept_tile_ids.append(tile_id)
                else:
                    removed_tile_count += 1

            self.tile_ids = kept_tile_ids
            print(
                f"[Dataset INFO] Pair-zero filter enabled: threshold>{self.hr_zero_ratio_threshold:.2f}, "
                f"matched={matched_count}, removed={removed_count}, removed_tiles={removed_tile_count}, kept_tiles={len(self.tile_ids)}"
            )
            if matched_count == 0:
                print(
                    f"[Dataset WARN] No samples matched {self.hr_zero_ratio_manifest}; "
                    "pair-zero filtering was skipped for this dataset."
                )
        
        # (保留 sample_num 逻辑)
        sample_num = params.get('sample_num', None)
        if sample_num and 0 < sample_num < len(self.tile_ids):
            self.tile_ids = self.tile_ids[:sample_num]
        
        print(f"[Dataset INFO] Initialized for {len(self.tile_ids)} tile locations.")
        print(f"  - Time Feature Injection: {'Enabled' if self.use_time_band else 'Disabled'}")
        print(f"  - Mask Feature Injection: {'Enabled' if self.use_mask_band else 'Disabled'}")
        if self.use_mask_band:
            print(f"  - Mask Type: {self.mask_type}")
            print(f"  - Use Advanced Processor: {self.use_advanced_processor}")
            if self.cloud_mask_dir:
                print(f"  - External Cloud Mask Dir: {self.cloud_mask_dir}")
        if self.use_tile_cache:
            print(f"  - Tile Cache: Enabled ({self.cache_dir})")
        if self.max_timesteps_per_sample > 0:
            print(f"  - Max Timesteps Per Sample: {self.max_timesteps_per_sample} (latest)")
        print("[Dataset INFO] AnytimeTemporalDataset __init__: done")

    def __len__(self):
        return len(self.tile_ids)

    def _load_tile_cache(self, tile_id):
        if not self.use_tile_cache or self.cache_dir is None:
            return None
        mem_payload = self._tile_cache_mem.get(tile_id)
        if mem_payload is not None:
            # LRU promote
            self._tile_cache_mem.move_to_end(tile_id)
            return mem_payload
        refl_path = self.cache_dir / f"tile_{tile_id}_reflectance.npy"
        mask_path = self.cache_dir / f"tile_{tile_id}_hard_valid_mask.npy"
        doy_path = self.cache_dir / f"tile_{tile_id}_day_of_year.npy"
        names_path = self.cache_dir / f"tile_{tile_id}_file_names.txt"

        has_npy_cache = refl_path.exists() and mask_path.exists() and doy_path.exists() and names_path.exists()
        try:
            if has_npy_cache:
                with names_path.open("r", encoding="utf-8") as f:
                    file_names = [line.strip() for line in f if line.strip()]
                reflectance_norm = np.load(refl_path, allow_pickle=False).astype(np.float32)
                hard_valid_mask = np.load(mask_path, allow_pickle=False).astype(np.float32)
                day_of_year = np.load(doy_path, allow_pickle=False).astype(np.float32)
            else:
                # Backward compatibility with old .npz cache.
                cache_path = self.cache_dir / f"tile_{tile_id}.npz"
                if not cache_path.exists():
                    return None
                cached = np.load(cache_path, allow_pickle=True)
                file_names = [str(x) for x in cached['file_names'].tolist()]
                reflectance_norm = cached['reflectance_norm'].astype(np.float32)
                hard_valid_mask = cached['hard_valid_mask'].astype(np.float32)
                day_of_year = cached['day_of_year'].astype(np.float32)
            payload = (file_names, reflectance_norm, hard_valid_mask, day_of_year)
            if self.tile_cache_mem_size > 0:
                self._tile_cache_mem[tile_id] = payload
                if len(self._tile_cache_mem) > self.tile_cache_mem_size:
                    self._tile_cache_mem.popitem(last=False)
            return payload
        except Exception:
            return None

    def __getitem__(self, index):
        sample_t0 = time.time()
        tile_id = self.tile_ids[index]
        can_use_cache = (
            self.use_tile_cache
            and self.cache_dir is not None
            and self.mask_type == 'hard'
            and (not self.use_advanced_processor)
            and (self.cloud_mask_dir is None)
        )

        cache_payload = self._load_tile_cache(tile_id) if can_use_cache else None
        lr_files = []
        cached_file_names = None
        cached_reflectance = None
        cached_hard_mask = None
        cached_doy = None

        if cache_payload is not None:
            cached_file_names, cached_reflectance, cached_hard_mask, cached_doy = cache_payload
            selected_indices = []
            for i, fname in enumerate(cached_file_names):
                if self.enable_hr_zero_filter:
                    zr = self._pair_zero_ratio_map.get(fname)
                    if zr is not None and zr > self.hr_zero_ratio_threshold:
                        continue
                selected_indices.append(i)
            if not selected_indices:
                raise FileNotFoundError(f"No cached timesteps remain after filtering for tile_id {tile_id}")

            if self.max_timesteps_per_sample > 0 and len(selected_indices) > self.max_timesteps_per_sample:
                # Keep latest K timesteps by filename/date order.
                selected_indices = selected_indices[-self.max_timesteps_per_sample:]

            cached_reflectance = cached_reflectance[selected_indices]
            cached_hard_mask = cached_hard_mask[selected_indices]
            cached_doy = cached_doy[selected_indices]
            cached_file_names = [cached_file_names[i] for i in selected_indices]
            lr_files = [self.lr_dir / n for n in cached_file_names]
        else:
            lr_files = self._tile_to_lr_files.get(tile_id, [])
            if self.enable_hr_zero_filter:
                lr_files, _, _ = _filter_files_by_hr_zero_ratio(
                    lr_files,
                    self._pair_zero_ratio_map,
                    self.hr_zero_ratio_threshold,
                )
            if not lr_files:
                raise FileNotFoundError(f"No LR images for tile_id {tile_id}")
            if self.max_timesteps_per_sample > 0 and len(lr_files) > self.max_timesteps_per_sample:
                lr_files = lr_files[-self.max_timesteps_per_sample:]

        # 加载HR (保留您的逻辑)
        hr_fname_pattern = lr_files[0].name
        target_hr_path = self.hr_dir / hr_fname_pattern
        with rasterio.open(target_hr_path) as src:
            hr_img = src.read()
        hr_tensor = torch.from_numpy(robust_per_image_normalize(hr_img)).float()

        processed_lr_timesteps = []
        timestamps = []
        temporal_validity_mask = []
        mask_prob_list = []
        indicating_mask_list = []
        if cache_payload is not None:
            for i in range(len(cached_file_names)):
                normalized_reflectance = cached_reflectance[i]
                hard_valid_mask = cached_hard_mask[i]
                day_of_year = float(cached_doy[i])
                h, w = hard_valid_mask.shape
                timestamps.append(day_of_year)
                mask_prob = hard_valid_mask.copy()
                time_is_valid = bool(float(hard_valid_mask.mean()) > 1e-6)
                features_to_concat = [normalized_reflectance]

                if self.use_time_band:
                    normalized_doy = (day_of_year / 366.0) * 2.0 - 1.0
                    time_band = np.full((1, h, w), normalized_doy, dtype=np.float32)
                    features_to_concat.append(time_band)

                if self.use_mask_band:
                    mask_band = mask_prob[np.newaxis, :, :].astype(np.float32)
                    mask_band = (mask_band * 2.0) - 1.0
                    features_to_concat.append(mask_band)
                    mask_prob_list.append(mask_prob.astype(np.float32))
                    indicating_mask_list.append(mask_prob.astype(np.float32))

                temporal_validity_mask.append(1.0 if time_is_valid else 0.0)
                full_timestep_data = np.concatenate(features_to_concat, axis=0)
                processed_lr_timesteps.append(torch.from_numpy(full_timestep_data).float())
        else:
            for lr_path in lr_files:
                with rasterio.open(lr_path) as src:
                    reflectance_data = src.read() # (C, H, W)
                    h, w = src.height, src.width

                date_str = lr_path.name.split('_')[0]
                dt_object = datetime.strptime(date_str, '%Y%m%d')
                day_of_year = dt_object.timetuple().tm_yday
                timestamps.append(day_of_year)

                # 基础有效性掩膜：所有反射率通道都大于0视作有效
                hard_valid_mask = (np.all(reflectance_data > 0, axis=0)).astype(np.float32)

                # 默认 mask_prob 就是硬掩膜
                mask_prob = hard_valid_mask.copy()
                soft_mask_raw = None

                # 若启用软掩膜，优先顺序：外部文件 -> LR末波段 -> 回退硬掩膜
                if self.use_mask_band and self.mask_type == 'soft':
                    loaded = None

                    if self.cloud_mask_dir is not None and self.cloud_mask_dir.exists():
                        soft_name = self.cloud_mask_pattern.format(date=date_str, tile_id=tile_id)
                        soft_path = self.cloud_mask_dir / soft_name
                        if soft_path.exists():
                            with rasterio.open(soft_path) as msrc:
                                loaded = msrc.read(1)

                    if loaded is None:
                        loaded = _extract_softmask_from_lr_if_possible(reflectance_data)

                    if loaded is not None:
                        soft_mask_raw = loaded
                        loaded = _normalize_softmask_values(loaded)
                        mask_prob = _safe_resize_mask(loaded, h, w)
                    else:
                        # 回退：无软掩膜来源时使用硬掩膜
                        mask_prob = hard_valid_mask.copy()

                # 高级处理器：根据配置对掩膜进行平滑/归一化/有效性判定
                time_is_valid = bool(float(hard_valid_mask.mean()) > 1e-6)
                if self.use_advanced_processor and self.cloud_mask_processor is not None:
                    processor_input = soft_mask_raw if soft_mask_raw is not None else mask_prob
                    mask_prob, processor_meta = self.cloud_mask_processor.process_pixel_mask(
                        processor_input,
                        reflectance_data=reflectance_data,
                    )
                    time_is_valid = bool(processor_meta.get('is_valid', time_is_valid))
                
                # --- 【核心改造】根据开关，动态拼接特征 ---
                # 1. 基础特征：归一化后的反射率
                normalized_reflectance = robust_per_image_normalize(reflectance_data)
                features_to_concat = [normalized_reflectance]

                # 2. (可选) 添加“时间波段”
                if self.use_time_band:
                    normalized_doy = (day_of_year / 366.0) * 2.0 - 1.0
                    time_band = np.full((1, h, w), normalized_doy, dtype=np.float32)
                    features_to_concat.append(time_band)

                # 3. (可选) 添加“掩膜波段”
                if self.use_mask_band:
                    mask_band = mask_prob[np.newaxis, :, :].astype(np.float32)
                    mask_band = (mask_band * 2.0) - 1.0 # 映射到 [-1, 1]
                    features_to_concat.append(mask_band)

                    mask_prob_list.append(mask_prob.astype(np.float32))
                    # 4c trainer 允许连续权重，直接用同口径的像素权重
                    indicating_mask_list.append(mask_prob.astype(np.float32))

                temporal_validity_mask.append(1.0 if time_is_valid else 0.0)
                
                # 拼接成一个时相的完整数据
                full_timestep_data = np.concatenate(features_to_concat, axis=0)
                processed_lr_timesteps.append(torch.from_numpy(full_timestep_data).float())
        
        lr_sequence = torch.stack(processed_lr_timesteps, dim=0)  # (T, C, H, W)

        timestamps = torch.tensor(timestamps, dtype=torch.float32)  # (T,)
        temporal_validity_mask = torch.tensor(temporal_validity_mask, dtype=torch.float32)  # (T,)

        # 返回模型期望的格式
        sample = {
            'lr_sequence': lr_sequence,
            'timestamps': timestamps,
            'gt': hr_tensor,
            # 供 temporal_fusion_mean 使用
            'mask': temporal_validity_mask,
        }

        if self.use_mask_band:
            # (T,H,W) -> (T,1,H,W)
            sample['mask_prob'] = torch.from_numpy(np.stack(mask_prob_list, axis=0)).unsqueeze(1).float()
            # (T,H,W)
            sample['indicating_mask'] = torch.from_numpy(np.stack(indicating_mask_list, axis=0)).float()
        
        if self.need_path:
            sample['path'] = str(target_hr_path)

        for k, v in list(sample.items()):
            if torch.is_tensor(v):
                sample[k] = v.contiguous().clone()
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
