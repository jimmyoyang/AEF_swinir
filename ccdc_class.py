import os
import re
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pyccd
from osgeo import gdal
from tqdm import tqdm

# ==============================================================================
# 1. 核心处理函数 (保持为顶层函数，以便多进程调用)
# ==============================================================================

def _process_pixel_task_with_coords(args):
    """
    对单个像素的时序数据运行CCDC并返回特征向量和坐标。
    这是一个独立的函数，以便multiprocessing可以轻松地序列化(pickle)它。
    """
    (r, c), pixel_ts_data, observation_dates, band_map, coeffs_to_extract = args
    result = _process_pixel_task((pixel_ts_data, observation_dates, band_map, coeffs_to_extract))
    return (r, c), result

def _process_pixel_task(args):
    """
    对单个像素的时序数据运行CCDC并返回特征向量。
    这是一个独立的函数，以便multiprocessing可以轻松地序列化(pickle)它。
    """
    pixel_ts_data, observation_dates, band_map, coeffs_to_extract = args
    
    # pyccd 需要 (波段, 时间) 格式
    pixel_ts_transposed = pixel_ts_data.T
    
    ccd_params = {'dates': observation_dates}
    for band_name, band_idx in band_map.items():
        # pyccd 的参数名是复数形式, e.g., 'blues', 'greens'
        ccd_params[f'{band_name}s'] = pixel_ts_transposed[band_idx]
        
    try:
        results = pyccd.detect(**ccd_params)
        
        if results and len(results['change_models']) > 0:
            model = results['change_models'][0]
            feature_vector = []
            # 'a0'是截距, a/b是谐波系数, c1是斜率
            coeff_indices = {'a0': 0, 'c1': 1, 'a1': 2, 'b1': 3, 'a2': 4, 'b2': 5, 'a3': 6, 'b3': 7}

            # 按字母顺序遍历波段和系数，以确保输出通道的顺序始终一致
            for band_name in sorted(band_map.keys()):
                coeffs = model[band_name]['coefficients']
                for coeff_name in sorted(coeffs_to_extract):
                    feature_vector.append(coeffs[coeff_indices[coeff_name]])
            
            return np.array(feature_vector, dtype=np.float32)
            
    except Exception:
        # 如果CCDC因任何原因失败（如数据不足），则忽略该像素
        pass
    
    # 默认返回一个零向量，其维度与成功时相同
    num_output_coeffs = len(band_map) * len(coeffs_to_extract)
    return np.zeros(num_output_coeffs, dtype=np.float32)


# ==============================================================================
# 2. CCDC 特征提取器 类
# ==============================================================================

class CCDCFeatureExtractor:
    """
    一个用于从时序栅格数据中提取CCDC特征并生成特征图像的类。
    它封装了数据加载、并行处理和结果重塑的核心逻辑。
    """
    def __init__(self, bands_to_process, coeffs_to_extract, use_multiprocessing=True, num_workers=None):
        """
        初始化提取器。

        Args:
            bands_to_process (dict): 告知处理哪些波段及其索引, e.g., {'blue': 0, 'green': 1}.
            coeffs_to_extract (list): 告知提取哪些谐波系数, e.g., ['a0', 'a1', 'b1'].
            use_multiprocessing (bool): 是否启用并行处理。
            num_workers (int, optional): 使用的CPU核心数. None表示全部.
        """
        self.band_map = bands_to_process
        self.coeffs_to_extract = coeffs_to_extract
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        self.num_output_channels = len(self.band_map) * len(self.coeffs_to_extract)

    @staticmethod
    def _filename_to_julian(filename):
        """从文件名中解析日期并转换为儒略日。"""
        try:
            # 假设文件名格式为 '..._YYYYMMDD.tif'
            date_str = os.path.splitext(filename)[0].split('_')[-1]
            dt = datetime.strptime(date_str, '%Y%m%d')
            return dt.toordinal()
        except:
            return None

    def process_tile(self, time_series_files, verbose=True):
        """
        处理一个瓦片(tile)的所有时序文件，并返回CCDC特征图像。

        Args:
            time_series_files (list): 包含单个瓦片所有时相文件路径的列表。
            verbose (bool): 是否打印详细日志和进度条。

        Returns:
            tuple: (ccdc_feature_image, reference_ds)
                   ccdc_feature_image 是一个 [H, W, C] 的numpy数组。
                   reference_ds 是一个GDAL数据集对象，用于地理参考。
        """
        if not time_series_files:
            raise ValueError("文件列表不能为空。")

        # 1. 加载数据
        if verbose: print(f"[INFO] 正在加载 {len(time_series_files)} 个时序影像...")
        time_series_stack, observation_dates, reference_ds = [], [], None
        
        for f_path in sorted(time_series_files):
            julian_date = self._filename_to_julian(os.path.basename(f_path))
            if julian_date is None: continue

            ds = gdal.Open(f_path)
            if ds is None: continue
            
            if reference_ds is None: reference_ds = ds
            
            observation_dates.append(julian_date)
            time_series_stack.append(ds.ReadAsArray())

        if not time_series_stack:
            raise ValueError("没有成功加载任何有效的时序影像。")
        
        # (时间, 波段, H, W) -> (H, W, 时间, 波段)
        time_series_data = np.stack(time_series_stack, axis=0).transpose(2, 3, 0, 1)
        observation_dates = np.array(observation_dates)
        height, width, _, _ = time_series_data.shape
        if verbose: print(f"[INFO] 数据加载完成. 维度: (H={height}, W={width}, T={len(observation_dates)}).")

        # 2. 准备并行任务（包含像素坐标以确保顺序正确）
        tasks = [
            ((r, c), time_series_data[r, c, :, :], observation_dates, self.band_map, self.coeffs_to_extract)
            for r in range(height) for c in range(width)
        ]

        # 3. 执行CCDC计算
        desc = f"CCDC on {height}x{width} pixels"
        if self.use_multiprocessing:
            if verbose: print(f"[INFO] 使用 {self.num_workers} 个CPU核心进行并行计算...")
            with Pool(processes=self.num_workers) as pool:
                # 使用 imap 而不是 imap_unordered 以保持顺序
                results_dict = {}
                for (r, c), result in tqdm(pool.imap(_process_pixel_task_with_coords, tasks), total=len(tasks), desc=desc):
                    results_dict[(r, c)] = result
        else:
            if verbose: print("[INFO] 使用单核心进行计算...")
            results_dict = {}
            for task in tqdm(tasks, desc=desc):
                (r, c), result = _process_pixel_task_with_coords(task)
                results_dict[(r, c)] = result
        
        # 4. 将结果重塑为图像（按坐标顺序）
        ccdc_feature_image = np.zeros((height, width, self.num_output_channels), dtype=np.float32)
        for (r, c), result in results_dict.items():
            ccdc_feature_image[r, c, :] = result
        
        if verbose: print("[INFO] CCDC特征提取完成。")
        return ccdc_feature_image, reference_ds


# ==============================================================================
# 3. 主工作流函数
# ==============================================================================

def run_ccdc_workflow(raw_data_dir, output_dir, bands_config, coeffs_config, use_multiprocessing=True):
    """
    一个完整的工作流，用于遍历目录、处理所有瓦片并保存结果。
    """
    start_time = time.time()
    print("="*60)
    print("CCDC 特征生成工作流启动")
    print("="*60)
    print(f"[CONFIG] 输入目录: {raw_data_dir}")
    print(f"[CONFIG] 输出目录: {output_dir}")
    print(f"[CONFIG] 并行计算: {'启用' if use_multiprocessing else '禁用'}")

    # 1. 初始化 CCDC 特征提取器
    extractor = CCDCFeatureExtractor(
        bands_to_process=bands_config,
        coeffs_to_extract=coeffs_config,
        use_multiprocessing=use_multiprocessing
    )

    # 2. 准备文件列表，按瓦片ID分组
    try:
        os.makedirs(output_dir, exist_ok=True)
        all_files = [f for f in os.listdir(raw_data_dir) if f.endswith(('.tif', '.tiff', '.img'))]
        if not all_files:
            print(f"[ERROR] 在目录 {raw_data_dir} 中未找到任何影像文件。请检查路径。")
            return
            
        tiles = {}
        for f in all_files:
            # 提取瓦片ID，格式应该与数据集中的格式一致: tile_(\d+_\d+)
            # 例如: 20230101_tile_001_002.tif -> tile_id = "001_002"
            match = re.search(r'tile_(\d+_\d+)\.tif', f)
            if match:
                tile_id = match.group(1)
                if tile_id not in tiles: tiles[tile_id] = []
                tiles[tile_id].append(os.path.join(raw_data_dir, f))
    except FileNotFoundError:
        print(f"[ERROR] 原始数据目录不存在: {raw_data_dir}")
        return

    print(f"[INFO] 发现 {len(tiles)} 个独立的瓦片需要处理: {list(tiles.keys())}")

    # 3. 遍历并处理每个瓦片
    for i, (tile_id, file_paths) in enumerate(tiles.items()):
        tile_start_time = time.time()
        print(f"\n--- [{i+1}/{len(tiles)}] 开始处理瓦片: {tile_id} ---")
        try:
            # 核心调用
            feature_image, ref_ds = extractor.process_tile(file_paths)
            
            # 保存结果
            output_filepath = os.path.join(output_dir, f"{tile_id}_ccdc_features.tif")
            
            # 辅助函数：保存GeoTIFF
            driver = gdal.GetDriverByName('GTiff')
            h, w, c = feature_image.shape
            out_ds = driver.Create(output_filepath, w, h, c, gdal.GDT_Float32)
            if ref_ds:
                out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
                out_ds.SetProjection(ref_ds.GetProjection())
            for band_idx in range(c):
                out_ds.GetRasterBand(band_idx + 1).WriteArray(feature_image[:, :, band_idx])
            out_ds.FlushCache()
            del out_ds
            
            tile_time = time.time() - tile_start_time
            print(f"[SUCCESS] 瓦片 {tile_id} 处理完毕，结果已保存。耗时: {tile_time:.2f} 秒。")

        except Exception as e:
            print(f"[ERROR] 处理瓦片 {tile_id} 时发生严重错误: {e}")

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"所有任务完成！总耗时: {total_time:.2f} 秒。")
    print("="*60)



