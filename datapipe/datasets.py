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
            target_n = sample_num if (sample_num and sample_num > 0) else None

            self.enable_hr_zero_filter = bool(params.get('enable_hr_zero_filter', True))
            self.hr_zero_ratio_threshold = float(params.get('hr_zero_ratio_filter_threshold', 0.05))
            self.hr_zero_ratio_manifest = params.get('hr_zero_ratio_manifest', 'patch_stats.log')
            if self.enable_hr_zero_filter:
                zero_ratio_map = _load_hr_zero_ratio_manifest(self.hr_zero_ratio_manifest)
                self.lr_files, removed_count, matched_count = _filter_files_by_hr_zero_ratio(
                    self.lr_files,
                    zero_ratio_map,
                    self.hr_zero_ratio_threshold,
                    target_sample_num=target_n,
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
            else:
                if target_n is not None and target_n < len(self.lr_files):
                    self.lr_files = self.lr_files[:target_n]

            print(f"[Dataset INFO] Initialized in NORMAL mode. Found {len(self.lr_files)} images in '{lr_dir}'.")
        
        self.need_path = params.get('need_path', False)
        self.scale_factor = params.get('scale_factor', 3)
        self.output_band_count = int(params.get('output_band_count', 0) or 0)
        self.output_band_indices = params.get('output_band_indices', None)

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
        hr_tensor = _select_output_bands_tensor(
            hr_tensor,
            output_band_count=self.output_band_count,
            output_band_indices=self.output_band_indices,
        )
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
import json
import hashlib
import random
import time
import bisect
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


def _filter_files_by_hr_zero_ratio(lr_files, zero_ratio_map, threshold, target_sample_num=None):
    if threshold is None:
        out = list(lr_files)
        if target_sample_num is not None and target_sample_num > 0:
            out = out[:target_sample_num]
        return out, 0, 0

    kept_files = []
    removed_count = 0
    matched_count = 0
    for lr_path in lr_files:
        if target_sample_num is not None and target_sample_num > 0 and len(kept_files) >= target_sample_num:
            break
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
    含 NaN/Inf 的波段使用 nanpercentile，避免训练中出现全 NaN 张量。
    """
    img_float = img.astype(np.float32)
    normalized_bands = []
    for i in range(img_float.shape[0]):
        band = img_float[i]
        # 跳过全零的通道
        if np.all(band == 0):
            normalized_bands.append(band)
            continue

        if not np.isfinite(band).any():
            normalized_bands.append(np.zeros_like(band))
            continue

        pct = np.nanpercentile if np.any(~np.isfinite(band)) else np.percentile
        lo_p, hi_p = pct(band, (lo, hi))

        if not np.isfinite(lo_p) or not np.isfinite(hi_p) or hi_p - lo_p < 1e-6:
            normalized_bands.append(np.zeros_like(band))
            continue

        clipped_band = np.clip(np.nan_to_num(band, nan=lo_p, posinf=hi_p, neginf=lo_p), lo_p, hi_p)
        normalized_band = (clipped_band - lo_p) / (hi_p - lo_p)
        normalized_bands.append(normalized_band * 2 - 1)  # 缩放到 [-1, 1]

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


def _infer_valid_mask_band_count(reflectance_data):
    """Infer how many leading LR bands should contribute to reflectance validity."""
    band_count = int(reflectance_data.shape[0])
    if band_count <= 1:
        return band_count

    candidate = reflectance_data[-1].astype(np.float32)
    finite = candidate[np.isfinite(candidate)]
    if finite.size == 0:
        return band_count - 1

    c_min = float(np.min(finite))
    c_max = float(np.max(finite))
    unique_count = int(np.unique(np.round(finite, 3)).size)

    # Many processed LR tiles carry a trailing QA/cloud-mask-like band. It can be
    # all-zero after min-max normalization, so including it in np.all(arr > 0)
    # makes an otherwise valid timestep look completely invalid.
    if 0.0 <= c_min <= 3.0 and 0.0 <= c_max <= 3.0 and unique_count <= 8:
        return band_count - 1

    return band_count


def _compute_hard_valid_mask(reflectance_data, valid_mask_band_count=0):
    if valid_mask_band_count and int(valid_mask_band_count) > 0:
        band_count = min(int(valid_mask_band_count), int(reflectance_data.shape[0]))
    else:
        band_count = _infer_valid_mask_band_count(reflectance_data)

    if band_count <= 0:
        return np.zeros(reflectance_data.shape[-2:], dtype=np.float32)

    reflectance_bands = reflectance_data[:band_count]
    valid = np.all(np.isfinite(reflectance_bands) & (reflectance_bands > 0), axis=0)
    return valid.astype(np.float32)


def _select_reflectance_bands(reflectance_data, reflectance_band_count=0):
    """Return the bands that should be treated as physical LR reflectance."""
    band_count = int(reflectance_band_count or 0)
    if band_count <= 0:
        return reflectance_data
    if reflectance_data.ndim == 4:
        band_count = min(band_count, int(reflectance_data.shape[1]))
        return reflectance_data[:, :band_count, ...]
    band_count = min(band_count, int(reflectance_data.shape[0]))
    return reflectance_data[:band_count, ...]


def _as_index_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = [v.strip() for v in value.split(',') if v.strip()]
    return [int(v) for v in list(value)]


def _select_output_bands_tensor(hr_tensor, output_band_count=0, output_band_indices=None):
    """Select output/target AlphaEarth bands while keeping 64-band behavior by default."""
    indices = _as_index_list(output_band_indices)
    if indices:
        max_idx = int(hr_tensor.shape[0]) - 1
        if min(indices) < 0 or max(indices) > max_idx:
            raise ValueError(
                f"output_band_indices={indices} out of range for HR tensor with {hr_tensor.shape[0]} bands"
            )
        index_tensor = torch.tensor(indices, dtype=torch.long, device=hr_tensor.device)
        return hr_tensor.index_select(0, index_tensor).contiguous()

    band_count = int(output_band_count or 0)
    if band_count <= 0:
        return hr_tensor
    if band_count > int(hr_tensor.shape[0]):
        raise ValueError(
            f"output_band_count={band_count} exceeds HR tensor band count {hr_tensor.shape[0]}"
        )
    return hr_tensor[:band_count].contiguous()


def _stable_file_list_hash(file_names):
    blob = "\n".join(str(name) for name in file_names)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:10]


def _atomic_save_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.{time.time_ns()}.tmp.npy")
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


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
        mask_cache_payload = {
            'version': 2,
            'mask_type': self.mask_type,
            'use_advanced_processor': self.use_advanced_processor,
            'processor': dict(self.mask_processor_config),
            'valid_mask_band_count': int(params.get('valid_mask_band_count', 0)),
        }
        mask_cache_blob = json.dumps(mask_cache_payload, sort_keys=True, default=str)
        self.mask_cache_key = hashlib.md5(mask_cache_blob.encode('utf-8')).hexdigest()[:10]
        
        self.need_path = params.get('need_path', False)
        fixed_temporal_len = params.get('fixed_temporal_len', 20)
        self.fixed_temporal_len = (
            None if fixed_temporal_len is None or int(fixed_temporal_len) <= 0
            else int(fixed_temporal_len)
        )
        self.cache_dir = Path(params['cache_dir']) if params.get('cache_dir') else None
        self.use_tile_cache = bool(params.get('use_tile_cache', self.cache_dir is not None))
        mask_cache_dir = params.get('mask_cache_dir', None)
        self.mask_cache_dir = Path(mask_cache_dir) if mask_cache_dir else self.cache_dir
        hard_mask_cache_dir = params.get('hard_mask_cache_dir', mask_cache_dir)
        self.hard_mask_cache_dir = Path(hard_mask_cache_dir) if hard_mask_cache_dir else None
        hr_cache_dir = params.get('hr_cache_dir', mask_cache_dir)
        self.hr_cache_dir = Path(hr_cache_dir) if hr_cache_dir else None
        self.valid_mask_band_count = int(params.get('valid_mask_band_count', 0))
        self.reflectance_band_count = int(params.get('reflectance_band_count', 0))
        self.output_band_count = int(params.get('output_band_count', 0) or 0)
        self.output_band_indices = params.get('output_band_indices', None)
        self.use_hard_mask_cache = bool(params.get('use_hard_mask_cache', self.hard_mask_cache_dir is not None))
        self.use_hr_cache = bool(params.get('use_hr_cache', self.hr_cache_dir is not None))
        self.build_hard_mask_cache_on_miss = bool(params.get('build_hard_mask_cache_on_miss', True))
        self.build_mask_cache_on_miss = bool(params.get('build_mask_cache_on_miss', True))
        self.require_tile_cache = bool(params.get('require_tile_cache', False))
        self.filter_to_tile_cache = bool(params.get('filter_to_tile_cache', self.require_tile_cache))
        self.require_hr_cache = bool(params.get('require_hr_cache', False))
        self.filter_to_hr_cache = bool(params.get('filter_to_hr_cache', self.require_hr_cache))
        hard_cache_payload = {
            'version': 2,
            'valid_mask_band_count': self.valid_mask_band_count,
        }
        self.hard_mask_cache_key = hashlib.md5(
            json.dumps(hard_cache_payload, sort_keys=True).encode('utf-8')
        ).hexdigest()[:10]
        self.hr_cache_key = hashlib.md5(b"hr_robust_per_image_normalize_v1_lo1_hi99").hexdigest()[:10]
        self.tile_cache_mem_size = int(params.get('tile_cache_mem_size', 64))
        self.slow_sample_warn_sec = float(params.get('slow_sample_warn_sec', 1.5))
        self.slow_sample_warn_max = int(params.get('slow_sample_warn_max', 5))
        self._slow_sample_warn_count = 0
        self.max_timesteps_per_sample = int(params.get('max_timesteps_per_sample', 0))
        self._tile_cache_mem = OrderedDict()

        self.enable_hr_zero_filter = bool(params.get('enable_hr_zero_filter', True))
        self.hr_zero_ratio_threshold = float(params.get('hr_zero_ratio_filter_threshold', 0.05))
        self.hr_zero_ratio_manifest = params.get('hr_zero_ratio_manifest', 'patch_stats.log')
        self.pre_filter_sample_num = params.get('pre_filter_sample_num', None)
        self.pre_filter_seed = int(params.get('pre_filter_seed', 0))
        sample_num = params.get('sample_num', None)
        self._target_sample_num = sample_num if (sample_num and sample_num > 0) else None
        default_sample_cache_size = (
            int(self._target_sample_num)
            if self._target_sample_num is not None and int(self._target_sample_num) <= 256
            else 0
        )
        self.sample_cache_mem_size = int(params.get('sample_cache_mem_size', default_sample_cache_size))
        self.use_sample_cache = bool(params.get('use_sample_cache', self.sample_cache_mem_size > 0))
        self._sample_cache_mem = OrderedDict()
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
                if self._target_sample_num is not None and len(kept_tile_ids) >= self._target_sample_num:
                    break
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
        
        # sample_num：未启用 pair-zero 过滤时仍截取前 N；启用时已在上方按过滤结果凑满 N 个 tile（若不足则全保留）
        if not self.enable_hr_zero_filter and self._target_sample_num is not None:
            if self._target_sample_num < len(self.tile_ids):
                self.tile_ids = self.tile_ids[: self._target_sample_num]

        if self.filter_to_tile_cache:
            before_cache_filter = len(self.tile_ids)
            self.tile_ids = [tile_id for tile_id in self.tile_ids if self._tile_cache_files_exist(tile_id)]
            removed_cache_miss = before_cache_filter - len(self.tile_ids)
            print(
                f"[Dataset INFO] Tile-cache filter enabled: kept={len(self.tile_ids)} "
                f"removed_cache_miss={removed_cache_miss}"
            )

        if self.filter_to_hr_cache:
            before_hr_filter = len(self.tile_ids)
            self.tile_ids = [tile_id for tile_id in self.tile_ids if self._hr_cache_file_exists(tile_id)]
            removed_hr_miss = before_hr_filter - len(self.tile_ids)
            print(
                f"[Dataset INFO] HR-cache filter enabled: kept={len(self.tile_ids)} "
                f"removed_cache_miss={removed_hr_miss}"
            )
        
        print(f"[Dataset INFO] Initialized for {len(self.tile_ids)} tile locations.")
        if self.tile_ids:
            preview_ids = self.tile_ids[: min(8, len(self.tile_ids))]
            print(
                f"  - Tile IDs Preview: first={preview_ids} "
                f"last={self.tile_ids[-1]} total={len(self.tile_ids)}"
            )
        else:
            print("  - Tile IDs Preview: EMPTY")
        print(f"  - Time Feature Injection: {'Enabled' if self.use_time_band else 'Disabled'}")
        print(f"  - Mask Feature Injection: {'Enabled' if self.use_mask_band else 'Disabled'}")
        if self.use_mask_band:
            print(f"  - Mask Type: {self.mask_type}")
            print(f"  - Use Advanced Processor: {self.use_advanced_processor}")
            if self.cloud_mask_dir:
                print(f"  - External Cloud Mask Dir: {self.cloud_mask_dir}")
        if self.reflectance_band_count > 0:
            print(f"  - Reflectance Input Bands: first {self.reflectance_band_count}")
        if self.output_band_indices is not None:
            print(f"  - Output Target Bands: indices {_as_index_list(self.output_band_indices)}")
        elif self.output_band_count > 0:
            print(f"  - Output Target Bands: first {self.output_band_count}")
        self.can_use_tile_cache = (
            self.use_tile_cache
            and self.cache_dir is not None
            and (self.cloud_mask_dir is None)
        )
        if self.use_tile_cache:
            if self.can_use_tile_cache:
                print(f"  - Tile Cache: Active ({self.cache_dir})")
                if self.use_hard_mask_cache and self.hard_mask_cache_dir is not None:
                    print(
                        f"  - Hard Mask Sidecar Cache: Enabled "
                        f"({self.hard_mask_cache_dir}, key={self.hard_mask_cache_key})"
                    )
                if self.use_mask_band and (self.mask_type == 'soft' or self.use_advanced_processor):
                    print(f"  - Mask Sidecar Cache: Enabled ({self.mask_cache_dir}, key={self.mask_cache_key})")
            else:
                print(
                    "  - Tile Cache: Bypassed "
                    f"(cache_dir={self.cache_dir}, cloud_mask_dir={self.cloud_mask_dir})"
                )
        if self.use_hr_cache and self.hr_cache_dir is not None:
            print(f"  - HR Cache: Enabled ({self.hr_cache_dir}, key={self.hr_cache_key})")
        if self.use_sample_cache and self.sample_cache_mem_size > 0:
            print(f"  - Processed Sample Cache: Enabled (LRU size={self.sample_cache_mem_size})")
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
            corrected_hard_mask = self._load_or_build_hard_mask_cache(tile_id, file_names)
            if corrected_hard_mask is not None and corrected_hard_mask.shape == hard_valid_mask.shape:
                hard_valid_mask = corrected_hard_mask
            if self.reflectance_band_count > 0:
                reflectance_norm = _select_reflectance_bands(
                    reflectance_norm,
                    reflectance_band_count=self.reflectance_band_count,
                )
            payload = (file_names, reflectance_norm, hard_valid_mask, day_of_year)
            if self.tile_cache_mem_size > 0:
                self._tile_cache_mem[tile_id] = payload
                if len(self._tile_cache_mem) > self.tile_cache_mem_size:
                    self._tile_cache_mem.popitem(last=False)
            return payload
        except Exception:
            return None

    def _tile_cache_files_exist(self, tile_id):
        if self.cache_dir is None:
            return False
        base = self.cache_dir / f"tile_{tile_id}"
        has_split_npy = (
            Path(str(base) + "_reflectance.npy").is_file()
            and Path(str(base) + "_hard_valid_mask.npy").is_file()
            and Path(str(base) + "_day_of_year.npy").is_file()
            and (self.cache_dir / f"tile_{tile_id}_file_names.txt").is_file()
        )
        return has_split_npy or (self.cache_dir / f"tile_{tile_id}.npz").is_file()

    def _first_cached_or_indexed_file_name(self, tile_id):
        if self.cache_dir is not None:
            names_path = self.cache_dir / f"tile_{tile_id}_file_names.txt"
            if names_path.is_file():
                try:
                    with names_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            name = line.strip()
                            if name:
                                return name
                except Exception:
                    pass

        lr_files = self._tile_to_lr_files.get(tile_id, [])
        if lr_files:
            return lr_files[0].name
        return None

    def _hr_cache_file_exists(self, tile_id):
        if not self.use_hr_cache or self.hr_cache_dir is None:
            return False
        first_name = self._first_cached_or_indexed_file_name(tile_id)
        if not first_name:
            return False
        target_hr_path = self.hr_dir / first_name
        return self._hr_cache_path(target_hr_path).is_file()

    def _hard_mask_cache_paths(self, tile_id):
        base = self.hard_mask_cache_dir / f"tile_{tile_id}_hardmask_{self.hard_mask_cache_key}"
        return Path(str(base) + "_valid.npy"), Path(str(base) + "_names.txt")

    def _load_or_build_hard_mask_cache(self, tile_id, file_names):
        if (
            not self.use_hard_mask_cache
            or self.hard_mask_cache_dir is None
            or self.cloud_mask_dir is not None
        ):
            return None

        mask_path, names_path = self._hard_mask_cache_paths(tile_id)
        if mask_path.exists() and names_path.exists():
            try:
                with names_path.open("r", encoding="utf-8") as f:
                    cached_names = [line.strip() for line in f if line.strip()]
                hard_mask = np.load(mask_path, allow_pickle=False).astype(np.float32)
                if cached_names == list(file_names) and hard_mask.shape[0] == len(file_names):
                    return hard_mask
            except Exception:
                pass
        if not self.build_hard_mask_cache_on_miss:
            return None

        hard_mask_list = []
        try:
            for fname in file_names:
                raw_path = self.lr_dir / fname
                with rasterio.open(raw_path) as src:
                    reflectance_data = src.read()
                hard_mask_list.append(
                    _compute_hard_valid_mask(
                        reflectance_data,
                        valid_mask_band_count=self.valid_mask_band_count,
                    ).astype(np.float32)
                )

            hard_mask_arr = np.stack(hard_mask_list, axis=0).astype(np.float16)
            try:
                self.hard_mask_cache_dir.mkdir(parents=True, exist_ok=True)
                _atomic_save_npy(mask_path, hard_mask_arr)
                tmp_names = names_path.with_name(
                    f".{names_path.stem}.{os.getpid()}.{time.time_ns()}.tmp"
                )
                try:
                    with tmp_names.open("w", encoding="utf-8") as f:
                        for name in file_names:
                            f.write(str(name) + "\n")
                    os.replace(tmp_names, names_path)
                finally:
                    if tmp_names.exists():
                        tmp_names.unlink()
            except Exception as e:
                print(f"[Dataset WARN] Failed to write hard mask sidecar cache for tile={tile_id}: {e}")
            return hard_mask_arr.astype(np.float32)
        except Exception as e:
            print(f"[Dataset WARN] Failed to build hard mask sidecar cache for tile={tile_id}: {e}")
            return None

    def _mask_cache_paths(self, tile_id, file_names):
        file_key = _stable_file_list_hash(file_names)
        base = self.mask_cache_dir / f"tile_{tile_id}_mask_{self.mask_cache_key}_{file_key}"
        return (
            Path(str(base) + "_prob.npy"),
            Path(str(base) + "_valid.npy"),
        )

    def _load_or_build_mask_cache(self, tile_id, file_names, hard_valid_mask):
        if (
            self.mask_cache_dir is None
            or not self.use_mask_band
            or not (self.mask_type == 'soft' or self.use_advanced_processor)
            or self.cloud_mask_dir is not None
        ):
            return None, None

        mask_path, valid_path = self._mask_cache_paths(tile_id, file_names)
        if mask_path.exists() and valid_path.exists():
            try:
                mask_prob = np.load(mask_path, allow_pickle=False).astype(np.float32)
                time_valid = np.load(valid_path, allow_pickle=False).astype(np.float32)
                if mask_prob.shape[0] == len(file_names) and time_valid.shape[0] == len(file_names):
                    return mask_prob, time_valid
            except Exception:
                pass
        if not self.build_mask_cache_on_miss:
            return None, None

        mask_prob_list = []
        time_valid_list = []
        try:
            for idx, fname in enumerate(file_names):
                raw_path = self.lr_dir / fname
                with rasterio.open(raw_path) as src:
                    reflectance_data = src.read()
                    h, w = src.height, src.width

                hard_valid = hard_valid_mask[idx].astype(np.float32)
                mask_prob = hard_valid.copy()
                soft_mask_raw = None
                has_soft_mask_source = False

                if self.mask_type == 'soft':
                    loaded = _extract_softmask_from_lr_if_possible(reflectance_data)
                    if loaded is not None:
                        soft_mask_raw = loaded
                        has_soft_mask_source = True
                        loaded = _normalize_softmask_values(loaded)
                        mask_prob = _safe_resize_mask(loaded, h, w)

                time_is_valid = bool(float(hard_valid.mean()) > 1e-6)
                if (
                    self.use_advanced_processor
                    and self.cloud_mask_processor is not None
                    and has_soft_mask_source
                ):
                    processor_input = soft_mask_raw
                    mask_prob, processor_meta = self.cloud_mask_processor.process_pixel_mask(
                        processor_input,
                        reflectance_data=reflectance_data,
                    )
                    time_is_valid = bool(processor_meta.get('is_valid', time_is_valid))

                mask_prob_list.append(mask_prob.astype(np.float32))
                time_valid_list.append(1.0 if time_is_valid else 0.0)

            mask_prob_arr = np.stack(mask_prob_list, axis=0).astype(np.float16)
            time_valid_arr = np.asarray(time_valid_list, dtype=np.float16)
            try:
                _atomic_save_npy(mask_path, mask_prob_arr)
                _atomic_save_npy(valid_path, time_valid_arr)
            except Exception as e:
                print(f"[Dataset WARN] Failed to write mask sidecar cache for tile={tile_id}: {e}")
            return mask_prob_arr.astype(np.float32), time_valid_arr.astype(np.float32)
        except Exception as e:
            print(f"[Dataset WARN] Failed to build mask sidecar cache for tile={tile_id}: {e}")
            return None, None

    def _hr_cache_path(self, target_hr_path):
        return self.hr_cache_dir / f"{target_hr_path.stem}_hr_{self.hr_cache_key}.npy"

    def _load_or_build_hr_tensor(self, target_hr_path):
        if self.use_hr_cache and self.hr_cache_dir is not None:
            hr_cache_path = self._hr_cache_path(target_hr_path)
            if hr_cache_path.exists():
                try:
                    cached_hr = np.load(hr_cache_path, allow_pickle=False).astype(np.float32)
                    return torch.from_numpy(cached_hr).float()
                except Exception:
                    pass
            if self.require_hr_cache:
                raise RuntimeError(
                    f"HR cache required but missing/unreadable for {target_hr_path.name}, "
                    f"cache_path={hr_cache_path}"
                )

        with rasterio.open(target_hr_path) as src:
            hr_img = src.read()
        hr_norm = robust_per_image_normalize(hr_img).astype(np.float32)

        if self.use_hr_cache and self.hr_cache_dir is not None:
            try:
                _atomic_save_npy(self._hr_cache_path(target_hr_path), hr_norm.astype(np.float16))
            except Exception as e:
                print(f"[Dataset WARN] Failed to write HR cache for {target_hr_path.name}: {e}")

        return torch.from_numpy(hr_norm).float()

    @staticmethod
    def _clone_sample(sample):
        return {
            k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in sample.items()
        }

    def _get_processed_sample_cache(self, tile_id):
        if not self.use_sample_cache or self.sample_cache_mem_size <= 0:
            return None
        sample = self._sample_cache_mem.get(tile_id)
        if sample is None:
            return None
        self._sample_cache_mem.move_to_end(tile_id)
        return self._clone_sample(sample)

    def _put_processed_sample_cache(self, tile_id, sample):
        if not self.use_sample_cache or self.sample_cache_mem_size <= 0:
            return
        self._sample_cache_mem[tile_id] = self._clone_sample(sample)
        self._sample_cache_mem.move_to_end(tile_id)
        while len(self._sample_cache_mem) > self.sample_cache_mem_size:
            self._sample_cache_mem.popitem(last=False)

    def __getitem__(self, index):
        sample_t0 = time.time()
        tile_id = self.tile_ids[index]
        cached_sample = self._get_processed_sample_cache(tile_id)
        if cached_sample is not None:
            return cached_sample

        cache_payload = self._load_tile_cache(tile_id) if self.can_use_tile_cache else None
        if cache_payload is None and self.require_tile_cache:
            raise RuntimeError(
                f"Tile cache required but missing/bypassed for tile_id={tile_id}, cache_dir={self.cache_dir}"
            )
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
            cached_mask_prob, cached_time_valid = self._load_or_build_mask_cache(
                tile_id,
                cached_file_names,
                cached_hard_mask,
            )
        else:
            cached_mask_prob = None
            cached_time_valid = None
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
        hr_tensor = self._load_or_build_hr_tensor(target_hr_path)
        hr_tensor = _select_output_bands_tensor(
            hr_tensor,
            output_band_count=self.output_band_count,
            output_band_indices=self.output_band_indices,
        )

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
                if cached_mask_prob is not None and i < cached_mask_prob.shape[0]:
                    mask_prob = cached_mask_prob[i].astype(np.float32)
                    if cached_time_valid is not None and i < cached_time_valid.shape[0]:
                        time_is_valid = bool(float(cached_time_valid[i]) > 0.5)
                    else:
                        time_is_valid = bool(float(mask_prob.mean()) > 1e-6)
                else:
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

                # 基础有效性掩膜：只检查真实反射率通道，避免尾部 QA/mask 通道污染有效性。
                hard_valid_mask = _compute_hard_valid_mask(
                    reflectance_data,
                    valid_mask_band_count=self.valid_mask_band_count,
                )

                # 默认 mask_prob 就是硬掩膜
                mask_prob = hard_valid_mask.copy()
                soft_mask_raw = None
                has_soft_mask_source = False

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
                        has_soft_mask_source = True
                        loaded = _normalize_softmask_values(loaded)
                        mask_prob = _safe_resize_mask(loaded, h, w)
                    else:
                        # 回退：无软掩膜来源时使用硬掩膜
                        mask_prob = hard_valid_mask.copy()

                # 高级处理器：根据配置对掩膜进行平滑/归一化/有效性判定
                time_is_valid = bool(float(hard_valid_mask.mean()) > 1e-6)
                if (
                    self.use_advanced_processor
                    and self.cloud_mask_processor is not None
                    and has_soft_mask_source
                ):
                    processor_input = soft_mask_raw
                    mask_prob, processor_meta = self.cloud_mask_processor.process_pixel_mask(
                        processor_input,
                        reflectance_data=reflectance_data,
                    )
                    time_is_valid = bool(processor_meta.get('is_valid', time_is_valid))
                
                # --- 【核心改造】根据开关，动态拼接特征 ---
                # 1. 基础特征：归一化后的反射率
                reflectance_input = _select_reflectance_bands(
                    reflectance_data,
                    reflectance_band_count=self.reflectance_band_count,
                )
                normalized_reflectance = robust_per_image_normalize(reflectance_input)
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

        sample = _pad_or_truncate_temporal_sample(sample, self.fixed_temporal_len)

        for k, v in list(sample.items()):
            if torch.is_tensor(v):
                sample[k] = v.contiguous().clone()
        self._put_processed_sample_cache(tile_id, sample)
        elapsed = time.time() - sample_t0
        if (
            self.slow_sample_warn_sec > 0
            and elapsed > self.slow_sample_warn_sec
            and self._slow_sample_warn_count < self.slow_sample_warn_max
        ):
            self._slow_sample_warn_count += 1
            print(
                f"[Dataset WARN] Slow sample tile={tile_id} took {elapsed:.2f}s "
                f"(tile_cache={'hit' if cache_payload is not None else 'miss/bypassed'}, "
                f"processed_cache=miss)."
            )
        return sample

class MultiPathAnytimeTemporalDataset(torch.utils.data.Dataset):
    """
    Concatenate multiple processed-year AnytimeTemporalDataset roots.

    Each root remains an independent child dataset so identical tile ids from
    different years are not merged into one longer temporal sequence.
    """
    def __init__(self, **params):
        parent_configs = params.get('parent_configs', {})
        inner_target = params.get('inner_target', 'datapipe.datasets.AnytimeTemporalDataset')
        inner_class = util_common.get_obj_from_str(str(inner_target))
        if inner_class is MultiPathAnytimeTemporalDataset:
            raise ValueError("MultiPathAnytimeTemporalDataset cannot wrap itself")

        roots = self._as_list(
            params.get('dataset_roots', params.get('roots', params.get('data_roots', None)))
        )
        split = params.get('split', None)
        if roots:
            if not split:
                raise ValueError("Multi-path dataset with dataset_roots requires params.split")
            lr_dirs = [Path(root) / str(split) / 'LR' for root in roots]
            hr_dirs = [Path(root) / str(split) / 'HR' for root in roots]
        else:
            lr_dirs = [Path(p) for p in self._as_list(params.get('lr_dirs', params.get('lr_dir', None)))]
            hr_dirs = [Path(p) for p in self._as_list(params.get('hr_dirs', params.get('hr_dir', None)))]
            if not lr_dirs or not hr_dirs:
                raise ValueError("Multi-path dataset requires dataset_roots or lr_dirs/hr_dirs")

        if len(lr_dirs) != len(hr_dirs):
            raise ValueError(f"lr_dirs/hr_dirs length mismatch: {len(lr_dirs)} vs {len(hr_dirs)}")

        dataset_names = self._as_list(params.get('dataset_names', params.get('names', None)))
        if dataset_names and len(dataset_names) != len(lr_dirs):
            raise ValueError(
                f"dataset_names length mismatch: {len(dataset_names)} vs {len(lr_dirs)}"
            )
        if not dataset_names:
            dataset_names = [
                self._default_dataset_name(root if roots else lr_dirs[i], i)
                for i, root in enumerate(roots or lr_dirs)
            ]
        dataset_names = [str(x) for x in dataset_names]

        reserved = {
            'parent_configs', 'inner_target',
            'dataset_roots', 'roots', 'data_roots', 'split',
            'lr_dir', 'hr_dir', 'lr_dirs', 'hr_dirs',
            'dataset_names', 'names',
            'cache_dir', 'mask_cache_dir', 'hard_mask_cache_dir', 'hr_cache_dir',
            'cache_dirs', 'mask_cache_dirs', 'hard_mask_cache_dirs', 'hr_cache_dirs',
            'cache_root', 'mask_cache_root', 'hard_mask_cache_root', 'hr_cache_root',
        }
        shared_params = {k: v for k, v in params.items() if k not in reserved}

        self.datasets = []
        self.dataset_names = dataset_names
        self.lr_dirs = [Path(p) for p in lr_dirs]
        self.hr_dirs = [Path(p) for p in hr_dirs]
        self.lr_dir = self.lr_dirs
        self.hr_dir = self.hr_dirs
        self._cum_lengths = []

        print("[Dataset INFO] MultiPathAnytimeTemporalDataset __init__: begin")
        print(f"  - split: {split if split else 'explicit lr_dirs/hr_dirs'}")
        print(f"  - roots: {len(self.lr_dirs)}")

        total = 0
        for idx, (name, lr_dir, hr_dir) in enumerate(zip(dataset_names, self.lr_dirs, self.hr_dirs)):
            if not lr_dir.is_dir() or not hr_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing LR/HR directory for dataset '{name}': "
                    f"lr_dir={lr_dir} exists={lr_dir.is_dir()} "
                    f"hr_dir={hr_dir} exists={hr_dir.is_dir()}"
                )

            child_params = dict(shared_params)
            child_params['lr_dir'] = str(lr_dir)
            child_params['hr_dir'] = str(hr_dir)
            child_params['parent_configs'] = parent_configs
            self._assign_cache_param(child_params, params, 'cache_dir', 'cache_dirs', 'cache_root', name, idx)
            self._assign_cache_param(child_params, params, 'mask_cache_dir', 'mask_cache_dirs', 'mask_cache_root', name, idx)
            self._assign_cache_param(child_params, params, 'hard_mask_cache_dir', 'hard_mask_cache_dirs', 'hard_mask_cache_root', name, idx)
            self._assign_cache_param(child_params, params, 'hr_cache_dir', 'hr_cache_dirs', 'hr_cache_root', name, idx)

            print(f"[Dataset INFO] Multi-path child[{idx}] name={name} lr={lr_dir} hr={hr_dir}")
            child = inner_class(**child_params)
            self.datasets.append(child)
            total += len(child)
            self._cum_lengths.append(total)
            print(f"[Dataset INFO] Multi-path child[{idx}] name={name} len={len(child)}")

        if not self.datasets:
            raise ValueError("Multi-path dataset built zero child datasets")
        print(f"[Dataset INFO] MultiPathAnytimeTemporalDataset __init__: done total_len={total}")

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
        except Exception:
            pass
        if isinstance(value, (str, Path)):
            return [value]
        return list(value)

    @staticmethod
    def _default_dataset_name(root_or_lr_dir, idx):
        path = Path(root_or_lr_dir)
        if path.name in {'LR', 'HR'}:
            path = path.parent.parent
        if path.name.startswith('processed_data') and path.parent.name:
            return path.parent.name
        return path.name or f"dataset_{idx}"

    @classmethod
    def _assign_cache_param(cls, child_params, params, singular_key, plural_key, root_key, name, idx):
        values = cls._as_list(params.get(plural_key, None))
        if values:
            if idx >= len(values):
                raise ValueError(f"{plural_key} has {len(values)} entries, but dataset index {idx} is required")
            child_params[singular_key] = str(values[idx])
            return

        root_value = params.get(root_key, None)
        scalar_value = params.get(singular_key, None)
        base = root_value if root_value is not None else scalar_value
        if base:
            child_params[singular_key] = str(Path(str(base)) / str(name))

    def __len__(self):
        return self._cum_lengths[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_idx = bisect.bisect_right(self._cum_lengths, index)
        prev = self._cum_lengths[dataset_idx - 1] if dataset_idx > 0 else 0
        return self.datasets[dataset_idx][index - prev]

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
