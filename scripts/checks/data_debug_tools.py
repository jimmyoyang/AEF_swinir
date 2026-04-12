#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据/实验流程人工核查与可视化工具
1. 随机抽查 val/test 下 LR/HR 配对，输出文件名和可视化
2. 输出部分模型输入、输出和GT的可视化
3. 检查数据分割脚本 prepare_data_local.py 的分割逻辑
4. 检查指标实现是否用模型输出和GT
"""
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

def random_tile_pair_check(lr_dir, hr_dir, num=5, seed=42):
    """随机抽查 LR/HR 配对，输出文件名和可视化"""
    random.seed(seed)
    lr_files = sorted(list(Path(lr_dir).glob('*.tif')))
    hr_files = sorted(list(Path(hr_dir).glob('*.tif')))
    print(f"[INFO] LR files: {len(lr_files)}, HR files: {len(hr_files)}")
    pairs = [(lr, Path(hr_dir)/lr.name) for lr in lr_files if (Path(hr_dir)/lr.name).exists()]
    print(f"[INFO] Matched pairs: {len(pairs)}")
    if len(pairs) == 0:
        print("[ERROR] No matched LR/HR pairs found!")
        return
    sample = random.sample(pairs, min(num, len(pairs)))
    for i, (lr_path, hr_path) in enumerate(sample):
        print(f"Pair {i+1}: {lr_path.name}")
        with rasterio.open(lr_path) as lr_ds, rasterio.open(hr_path) as hr_ds:
            lr_img = lr_ds.read([1,2,3]) if lr_ds.count>=3 else lr_ds.read(1)
            hr_img = hr_ds.read([1,2,3]) if hr_ds.count>=3 else hr_ds.read(1)
            fig, axs = plt.subplots(1,2,figsize=(8,4))
            axs[0].imshow(np.moveaxis(lr_img,0,-1) if lr_img.ndim==3 else lr_img, cmap='gray')
            axs[0].set_title('LR')
            axs[1].imshow(np.moveaxis(hr_img,0,-1) if hr_img.ndim==3 else hr_img, cmap='gray')
            axs[1].set_title('HR')
            plt.suptitle(lr_path.name)
            plt.show()

def check_prepare_data_local_split():
    """检查 prepare_data_local.py 的分割逻辑（输出分割比例和tile分布）"""
    # 自动化检查train/val/test tile_id无重叠
    def get_tile_ids(tile_dir):
        tile_ids = set()
        for f in Path(tile_dir).glob('*.tif'):
            # 假设文件名格式为 *_tile_行_列.tif
            parts = f.stem.split('_tile_')
            if len(parts) == 2:
                tile_ids.add(parts[1])
        return tile_ids

    base = Path('./data/processed_data')
    sets = ['train', 'val', 'test']
    id_map = {}
    for s in sets:
        lr_dir = base / s / 'LR'
        id_map[s] = get_tile_ids(lr_dir)
        print(f"[{s}] tile_id count: {len(id_map[s])}")
    # 检查两两交集
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            inter = id_map[sets[i]] & id_map[sets[j]]
            if inter:
                print(f"[ERROR] {sets[i]} & {sets[j]} overlap tile_ids: {sorted(list(inter))[:10]} ... (total {len(inter)})")
            else:
                print(f"[OK] {sets[i]} & {sets[j]} no overlap.")


def check_metric_code(metric_py_path):
    """检查指标实现是否用模型输出和GT"""
    with open(metric_py_path, 'r', encoding='utf-8') as f:
        code = f.read()
    if 'pred' in code and 'gt' in code:
        print("[OK] Metric implementation contains both 'pred' and 'gt' variables. Please manually confirm pred=model output, gt=ground truth.")
    else:
        print("[WARN] 'pred'/'gt' variables not detected in metric code. Please manually review the metric implementation!")

if __name__ == '__main__':
    # 1. 随机抽查 val/test 下 LR/HR 配对
    print("==== Random check: val/LR, val/HR pairs ====")
    random_tile_pair_check('./data/processed_data/val/LR', './data/processed_data/val/HR', num=5)
    print("==== Random check: test/LR, test/HR pairs ====")
    random_tile_pair_check('./data/processed_data/test/LR', './data/processed_data/test/HR', num=5)
    # 2. Check data split script
    check_prepare_data_local_split()
    # 3. Check metric implementation
    check_metric_code('scripts/compare_recovered_vs_current.py')
    check_metric_code('scripts/compare_training_log_metrics.py')
    # 4. Visualization of model input/output/GT (extend as needed)
    print("[TODO] For model input/output/GT visualization, please add model inference and output path, then call random_tile_pair_check.")
