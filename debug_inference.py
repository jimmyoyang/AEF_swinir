# 文件路径: debug_inference.py
# (v_anytime - 适配AnytimeTemporalDataset，用于单瓦片调试)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') # 确保在不同环境下能弹出绘图窗口

# ==============================================================================
# 1. 环境与路径设置
# ==============================================================================
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

from utils.util_common import get_obj_from_str
from datapipe.datasets import create_dataset # 复用 create_dataset

# ==============================================================================
# 2. 核心调试函数
# ==============================================================================
def debug_single_inference(config_path, ckpt_path, hr_image_path):
    print("\n" + "="*80 + "\n===== MODE: DEBUG SINGLE ANYTIME INFERENCE =====\n" + "="*80 + "\n")
    
    # 1. 加载配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}"); sys.exit(1)
    
    # 2. 设备与模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = get_obj_from_str(configs['model']['target'])(**configs['model']['params']).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ 模型加载成功: {ckpt_path}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}"); sys.exit(1)
        
    # 3. 【核心修改】使用 Dataset 加载与该HR图像关联的完整时序数据
    print(f"✅ 使用 AnytimeTemporalDataset 加载与HR图像关联的完整时序: {Path(hr_image_path).name}")
    try:
        # 伪造一个 dataset config 来实例化，并强制只加载这一个样本
        debug_data_config = {
            'target': 'datapipe.datasets.AnytimeTemporalDataset',
            'params': {
                'hr_dir': str(Path(hr_image_path).parent),
                'lr_dir': str(Path(hr_image_path).parent.parent / 'LR'), # 推断LR目录
                'sample_num': 1 # 确保只加载这一个
            }
        }
        # 找到这个特定文件在文件列表中的索引
        dataset_full = create_dataset(debug_data_config, parent_configs=configs)
        target_idx = -1
        for i, sample in enumerate(dataset_full.target_files):
            if Path(sample).name == Path(hr_image_path).name:
                target_idx = i
                break
        if target_idx == -1:
            raise FileNotFoundError(f"指定的HR图像 {hr_image_path} 未在数据集中找到。")

        data = dataset_full[target_idx] # 获取经过与训练时完全相同处理的数据
        
        # 添加 batch 维度并移动到设备
        inputs = {k: v.unsqueeze(0).to(device) for k, v in data.items() if isinstance(v, torch.Tensor)}
        print(f"✅ 数据加载与预处理成功。")

    except Exception as e:
        print(f"❌ 使用Dataset加载数据失败: {e}"); sys.exit(1)

    # 4. 推理
    with torch.no_grad():
        pred_tensor = model(inputs)
    
    # 5. 打印张量统计信息
    print("\n--- Tensor Stats (Range [-1, 1]) ---")
    print(f"  - Input LR Seq: Min={inputs['lr_sequence'].min():.3f}, Max={inputs['lr_sequence'].max():.3f}, Mean={inputs['lr_sequence'].mean():.3f}")
    print(f"  - GT:           Min={inputs['gt'].min():.3f}, Max={inputs['gt'].max():.3f}, Mean={inputs['gt'].mean():.3f}")
    print(f"  - Prediction:   Min={pred_tensor.min():.3f}, Max={pred_tensor.max():.3f}, Mean={pred_tensor.mean():.3f}")

    # 6. 可视化
    print("\n✅ 生成可视化图表...")
    def norm_display(img):
        # 归一化到 [0, 1] 用于显示
        return (img - img.min()) / (img.max() - img.min()) if (img.max() - img.min()) > 1e-6 else img

    rgb_chn = configs.get('train', {}).get('rgb_chn', [2, 1, 0]) # 假设是 BGR -> RGB
    
    # 我们只可视化第一个LR时相作为代表
    lr_vis = norm_display(inputs['lr_sequence'][0, 0, :, :, :].cpu().permute(1, 2, 0).numpy())
    sr_vis = norm_display(pred_tensor[0].cpu().permute(1, 2, 0).numpy())
    hr_vis = norm_display(inputs['gt'][0].cpu().permute(1, 2, 0).numpy())
    error_map = np.abs(sr_vis - hr_vis).mean(axis=-1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f'Debug Inference for: {Path(hr_image_path).name}', fontsize=16)
    
    axes[0,0].imshow(lr_vis[:, :, rgb_chn]); axes[0,0].set_title('Input LR (First Timestep)')
    axes[0,1].imshow(sr_vis[:, :, rgb_chn]); axes[0,1].set_title('Prediction (SR)')
    axes[1,0].imshow(hr_vis[:, :, rgb_chn]); axes[1,0].set_title('Ground Truth (HR)')
    im = axes[1,1].imshow(error_map, cmap='hot'); axes[1,1].set_title('Absolute Error Map'); fig.colorbar(im, ax=axes[1,1])
        
    for ax in axes.flatten(): ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

# ==============================================================================
# 3. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Debug a single inference for AnytimeSwinIR.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file (e.g., configs/config_swinir.yaml).")
    parser.add_argument('--image', type=str, required=True, help="Path to the single HIGH-RESOLUTION ground truth image file for debugging.")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to a specific model checkpoint (.pth).")
    args = parser.parse_args()
    
    debug_single_inference(args.config, args.ckpt, args.image)
