# 文件路径: debug_inference.py
# (v10 - 最终融合版：采纳v8健壮性 + 统一数据处理流程 + 统一 s1/gt 键名)

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
matplotlib.use('TkAgg')

# ==============================================================================
# 1. 环境与路径设置
# ==============================================================================
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

from utils.util_common import get_obj_from_str
from datapipe.datasets import create_dataset # 直接复用 create_dataset

# ==============================================================================
# 2. 核心调试函数
# ==============================================================================
def debug_single_inference(config_path, ckpt_path, lr_image_path):
    print("\n" + "="*80 + "\n===== MODE: UNIFIED DEBUG SINGLE INFERENCE =====\n" + "="*80 + "\n")
    
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
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ 模型加载成功: {ckpt_path}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}"); sys.exit(1)
        
    # 3. 【核心】使用 Dataset 进行统一的数据加载和预处理
    print(f"✅ 使用 PreprocessedTileDataset 加载图像: {Path(lr_image_path).name}")
    try:
        # 伪造一个 dataset config 来实例化，并传入 debug 模式所需的路径
        debug_data_config = {
            'target': 'datapipe.datasets.PreprocessedTileDataset',
            'params': {
                'is_debug': True,
                'debug_lr_path': str(lr_image_path),
            }
        }
        debug_dataset = create_dataset(debug_data_config)
        data = debug_dataset[0] # 获取经过与训练时完全相同处理的数据
        
        # 使用 's1' 和 'gt' 键名
        inputs = data['s1'].unsqueeze(0).to(device)
        gt_tensor = data.get('gt', None)
        if gt_tensor is not None:
            gt_tensor = gt_tensor.unsqueeze(0).to(device)
        print(f"✅ 数据加载与预处理成功。")

    except Exception as e:
        print(f"❌ 使用Dataset加载数据失败: {e}"); sys.exit(1)

    # 4. 推理
    with torch.no_grad():
        pred_tensor = model(inputs)
    
    # 5. 打印张量统计信息
    print("\n--- Tensor Stats (Range [-1, 1]) ---")
    print(f"  - Input (s1): Min={inputs.min():.3f}, Max={inputs.max():.3f}, Mean={inputs.mean():.3f}")
    if gt_tensor is not None:
        print(f"  - GT:         Min={gt_tensor.min():.3f}, Max={gt_tensor.max():.3f}, Mean={gt_tensor.mean():.3f}")
    print(f"  - Prediction: Min={pred_tensor.min():.3f}, Max={pred_tensor.max():.3f}, Mean={pred_tensor.mean():.3f}")

    # 6. 可视化
    print("\n✅ 生成可视化图表...")
    def norm_display(img):
        min_val, max_val = img.min(), img.max()
        return (img - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-6 else img

    rgb_chn = configs.get('train', {}).get('rgb_chn', [0, 1, 2])
    
    lr_vis = norm_display(((inputs.clamp(-1, 1) + 1) / 2)[0].cpu().permute(1, 2, 0).numpy())
    sr_vis = norm_display(((pred_tensor.clamp(-1, 1) + 1) / 2)[0].cpu().permute(1, 2, 0).numpy())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f'Debug Inference for: {Path(lr_image_path).name}', fontsize=16)
    
    axes[0,0].imshow(lr_vis[:, :, rgb_chn]); axes[0,0].set_title('Input (s1)')
    axes[0,1].imshow(sr_vis[:, :, rgb_chn]); axes[0,1].set_title('Prediction')
    
    if gt_tensor is not None:
        hr_vis = norm_display(((gt_tensor.clamp(-1, 1) + 1) / 2)[0].cpu().permute(1, 2, 0).numpy())
        error_map = np.abs(sr_vis - hr_vis).mean(axis=-1)
        axes[1,0].imshow(hr_vis[:, :, rgb_chn]); axes[1,0].set_title('Ground Truth (gt)')
        im = axes[1,1].imshow(error_map, cmap='hot'); axes[1,1].set_title('Absolute Error Map'); fig.colorbar(im, ax=axes[1,1])
    else:
        axes[1,0].axis('off'); axes[1,0].text(0.5, 0.5, 'Ground Truth N/A', ha='center', va='center')
        axes[1,1].axis('off'); axes[1,1].text(0.5, 0.5, 'Error Map N/A', ha='center', va='center')
        
    for ax in axes.flatten(): ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

# ==============================================================================
# 3. 主程序入口 (采纳您v8的健壮设计)
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Debug a single inference.")
    parser.add_argument('--config', type=str, help="Path to the config file (e.g., configs/config_swinir.yaml).")
    parser.add_argument('--image', type=str, help="Path to the single LOW-RESOLUTION image file for debugging.")
    parser.add_argument('--ckpt', type=str, default=None, help="Optional: Path to a specific model checkpoint.")
    args = parser.parse_args()
    
    if not args.config or not args.image:
        print("❌ Error: Both --config and --image arguments are required.")
        parser.print_help()
        sys.exit(1)
        
    config_path_abs = Path(args.config).resolve()
    image_path_abs = Path(args.image).resolve()
    ckpt_path_abs = None
    
    if args.ckpt:
        ckpt_path_abs = Path(args.ckpt).resolve()
    else:
        print(f"ℹ️ 未指定--ckpt，将根据config文件中的save_dir自动搜索...")
        try:
            with open(config_path_abs, 'r', encoding='utf-8') as f:
                run_log_dir = Path(yaml.safe_load(f)['train']['save_dir'])
            if not run_log_dir.is_absolute():
                run_log_dir = project_root / run_log_dir

            latest_run_dir = max([d for d in run_log_dir.iterdir() if d.is_dir()], key=os.path.getmtime)
            print(f"ℹ️ 搜索最新的训练目录: {latest_run_dir.name}")
            
            ckpt_dir = latest_run_dir / 'ckpts'
            best_ckpt = ckpt_dir / 'model_best.pth'
            
            if best_ckpt.exists():
                ckpt_path_abs = best_ckpt
                print(f"✅ 找到最佳检查点: model_best.pth")
            else:
                ckpt_files = list(ckpt_dir.glob('model_*.pth'))
                if not ckpt_files: raise FileNotFoundError
                latest_ckpt_file = max(ckpt_files, key=lambda p: int(p.stem.split('_')[-1]))
                ckpt_path_abs = latest_ckpt_file
                print(f"⚠️ 未找到model_best.pth，使用最新检查点: {latest_ckpt_file.name}")
        except Exception as e:
            print(f"❌ 自动查找检查点失败: {e}"); sys.exit(1)
    # 校验所有路径有效性
    path_check = {
        "Config": config_path_abs,
        "Checkpoint": ckpt_path_abs,
        "Image": image_path_abs
    }
    invalid_paths = [name for name, path in path_check.items() if not path.exists()]
    
    if invalid_paths:
        print(f"\n❌ 以下路径无效: {', '.join(invalid_paths)}")
        for name, path in path_check.items():
            print(f"  - {name}: '{path}' (存在: {path.exists()})")
        sys.exit(1)
    else:
        # 执行推理调试
        debug_single_inference(str(config_path_abs), str(ckpt_path_abs), str(image_path_abs))