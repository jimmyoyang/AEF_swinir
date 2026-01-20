# 文件路径: main.py
# (v_final - 统一的、与新架构完全兼容的程序入口)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf

# --- 环境设置 ---
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

from utils.util_common import get_obj_from_str
from datapipe.datasets import create_dataset
from trainer import TrainerAlphaSR # 假设您的Trainer类在trainer.py中
from inference import Predictor      # 假设您的Predictor类在inference.py中
import torch

def main():
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Unified Main Runner for Anytime Spatio-Temporal Fusion Project")
    parser.add_argument('--cfg_path', type=str, default='configs/config_swinir.yaml', help='Path to the configuration file.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help="Execution mode: 'train' or 'test'.")
    parser.add_argument('--save_dir', type=str, default=None, help='Override the save directory specified in the config file.')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from the latest checkpoint in the save_dir.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Specify a direct path to a checkpoint for resuming or testing.')
    
    # 推理专用参数
    parser.add_argument('--input_dir', type=str, default=None, help="[Test Mode] Path to the input data directory for inference.")
    parser.add_argument('--output_dir', type=str, default=None, help="[Test Mode] Path to save inference results.")

    args = parser.parse_args()

    # --- 2. 加载和合并配置 ---
    configs = OmegaConf.load(args.cfg_path)
    if args.save_dir:
        configs.train.save_dir = args.save_dir

    # --- 3. 根据模式执行不同逻辑 ---
    if args.mode == 'train':
        run_training(args, configs)
    elif args.mode == 'test':
        run_testing(args, configs)

def run_training(args, configs):
    """执行训练流程"""
    print("\n" + "="*80 + "\n🚀 Starting Training Mode...\n" + "="*80)
    
    # --- 智能处理恢复逻辑 ---
    exp_dir = Path(configs.train.save_dir)
    resume_path = None
    if args.ckpt_path: # 最高优先级：直接指定了ckpt路径
        resume_path = args.ckpt_path
        print(f"[Main] Resuming from user-specified checkpoint: {resume_path}")
    elif args.resume: # 第二优先级：用户要求恢复，自动查找
        if not exp_dir.exists() or not any(exp_dir.iterdir()):
             print(f"[Main] '--resume' flag is set, but experiment directory {exp_dir} is empty or not found. Starting from scratch.")
        else:
            try:
                latest_run_dir = max([d for d in exp_dir.iterdir() if d.is_dir()], key=os.path.getmtime)
                ckpt_dir = latest_run_dir / 'ckpts'
                if ckpt_dir.exists():
                    ckpt_files = list(ckpt_dir.glob('model_*.pth'))
                    if ckpt_files:
                        latest_ckpt = max(ckpt_files, key=lambda p: int(p.stem.split('_')[-1]))
                        resume_path = str(latest_ckpt)
                        print(f"[Main] '--resume' flag is set. Automatically resuming from latest checkpoint: {resume_path}")
                    else: print(f"[Main] '--resume' flag is set, but no checkpoints found in {ckpt_dir}. Starting from scratch.")
                else: print(f"[Main] '--resume' flag is set, but ckpt directory not found in {latest_run_dir}. Starting from scratch.")
            except ValueError:
                print(f"[Main] '--resume' flag is set, but no subdirectories found in {exp_dir}. Starting from scratch.")

    configs.resume = resume_path 

    # --- 实例化并启动训练器 ---
    trainer_cls = get_obj_from_str(configs.trainer.target)
    trainer = trainer_cls(configs)
    trainer.train()

def run_testing(args, configs):
    """执行测试/推理流程"""
    print("\n" + "="*80 + "\n🚀 Starting Test (Inference) Mode...\n" + "="*80)
    
    if not args.ckpt_path:
        print("❌ CRITICAL: For 'test' mode, a checkpoint path must be specified via '--ckpt_path'.")
        sys.exit(1)
    if not args.input_dir or not args.output_dir:
        print("❌ CRITICAL: For 'test' mode, '--input_dir' and '--output_dir' must be specified.")
        sys.exit(1)

    # --- 实例化 Predictor ---
    predictor = Predictor(args.cfg_path, args.ckpt_path)
    
    # --- 创建 Test DataLoader ---
    # 使用与训练时相同的配置来创建数据集
    data_config = {
        'target': 'datapipe.datasets.AnytimeTemporalDataset',
        'params': {
            'lr_dir': str(Path(args.input_dir) / 'LR'),
            'hr_dir': str(Path(args.input_dir) / 'HR'),
            'need_path': True # 确保返回路径以便保存
        }
    }
    test_dataset = create_dataset(data_config, parent_configs=predictor.configs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # --- 执行推理 ---
    predictor.run_inference(test_loader, Path(args.output_dir))

if __name__ == '__main__':
    main()
