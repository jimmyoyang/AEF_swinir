# 文件路径: check_prepared_data.py
# (一个用于验证 prepare_data_local.py 输出的专用调试脚本)

import os
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import random

# --- 环境设置 ---
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))
from datapipe.datasets import create_dataset

def check_data_structure(base_dir):
    """检查物理文件结构是否符合预期。"""
    print("\n--- [Check 1/3] Verifying Directory Structure ---")
    base_path = Path(base_dir)
    splits = ['train', 'val', 'test']
    is_ok = True
    
    if not base_path.exists():
        print(f"❌ FAILED: Base processed data directory not found: {base_path}")
        return False

    for split in splits:
        split_path = base_path / split
        if not split_path.exists():
            print(f"⚠️ WARNING: Split directory not found: {split_path}. This might be okay if you intended to skip it.")
            continue
        
        lr_path = split_path / 'LR'
        hr_path = split_path / 'HR'
        if not lr_path.exists() or not hr_path.exists():
            print(f"❌ FAILED: LR or HR subdirectory missing in: {split_path}")
            is_ok = False
        
        print(f"✅ Found '{split}' directory with LR and HR subdirectories.")
    
    if is_ok:
        print("✅ Directory structure appears correct.")
    return is_ok

def check_dataset_loading(configs):
    """检查 AnytimeTemporalDataset 是否能成功加载数据并返回正确格式。"""
    print("\n--- [Check 2/3] Verifying Dataset Loading and Sample Format ---")
    try:
        # 我们只检查训练集，因为它通常是数据量最大的
        train_conf = configs.data.train
        dataset = create_dataset(train_conf, parent_configs=configs)
        
        if len(dataset) == 0:
            print("❌ FAILED: Training dataset is empty. Check your data paths in the config and the output of prepare_data_local.py.")
            return False, None
            
        # 随机选择一个样本进行深度检查
        random_idx = random.randint(0, len(dataset) - 1)
        print(f"Inspecting random sample at index: {random_idx}")
        sample = dataset[random_idx]
        
        # 检查样本结构
        assert isinstance(sample, dict), "Sample must be a dictionary."
        required_keys = ['lr_sequence', 'timestamps', 'gt']
        assert all(key in sample for key in required_keys), f"Sample is missing required keys. Found: {list(sample.keys())}"
        
        # 检查张量类型和维度
        lr_seq = sample['lr_sequence']
        ts = sample['timestamps']
        gt = sample['gt']
        
        assert isinstance(lr_seq, torch.Tensor) and lr_seq.dim() == 4, "lr_sequence should be a 4D Tensor (T, C, H, W)."
        assert isinstance(ts, torch.Tensor) and ts.dim() == 1, "timestamps should be a 1D Tensor (T,)."
        assert isinstance(gt, torch.Tensor) and gt.dim() == 3, "gt should be a 3D Tensor (C, H, W)."
        assert lr_seq.shape[0] == ts.shape[0], "Length of lr_sequence and timestamps must match."
        
        print("✅ Sample format and tensor dimensions are correct.")
        print(f"   - LR Sequence Shape: {lr_seq.shape}")
        print(f"   - Timestamps Shape: {ts.shape}")
        print(f"   - GT Shape: {gt.shape}")
        
        return True, sample
    except Exception as e:
        print(f"❌ FAILED: An error occurred during dataset loading check. Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def visualize_sample(sample, save_path, rgb_channels=[0,1,2]):
    """可视化一个加载的样本。"""
    print("\n--- [Check 3/3] Visualizing a Sample ---")
    try:
        lr_img = sample['lr_sequence'][0].numpy() # 取第一个时相
        gt_img = sample['gt'].numpy()
        
        def norm_for_vis(img):
            # 假设输入是 [-1, 1]，先转到 [0, 1]
            img = (img + 1) / 2.0
            img = img.transpose(1, 2, 0) # C,H,W -> H,W,C
            return np.clip(img, 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Data Preparation & Loading Check (Normalized)")
        
        # 使用配置的RGB通道进行可视化
        axes[0].imshow(norm_for_vis(lr_img[rgb_channels,:,:])); axes[0].set_title(f"LR Sample (1st Timestep)\nShape: {lr_img.shape}")
        axes[1].imshow(norm_for_vis(gt_img[rgb_channels,:,:])); axes[1].set_title(f"GT Sample\nShape: {gt_img.shape}")
        
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Visualization saved to: {save_path}")
    except Exception as e:
        print(f"❌ FAILED: Could not visualize the sample. Error: {e}")

def main():
    """主执行函数。"""
    # 使用您的主配置文件来获取数据路径
    config_path = 'configs/config_swinir.yaml'
    print("="*80)
    print(f"🚀 Starting Data Preparation Verification using '{config_path}'...")
    print("="*80)

    try:
        configs = OmegaConf.load(config_path)
        processed_data_dir = Path(configs.data.train.params.hr_dir).parent.parent
    except Exception as e:
        print(f"❌ CRITICAL: Could not load config or parse data directory. Error: {e}")
        return

    # 依次执行检查
    if not check_data_structure(processed_data_dir):
        sys.exit(1)
        
    is_ok, sample = check_dataset_loading(configs)
    if not is_ok:
        sys.exit(1)
        
    visualize_sample(sample, processed_data_dir / "data_check_visualization.png", configs.train.get('rgb_chn', [0,1,2]))

    print("\n" + "="*80)
    print("🎉🎉🎉 Data Preparation Verification Passed! 🎉🎉🎉")
    print("Your prepared data structure is correct and can be loaded by the dataset.")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
