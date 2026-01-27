# 文件路径: main.py
# 最终整合版：兼容新架构的统一程序入口（保留完整的训练恢复逻辑+新推理流程）
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf
import torch

# --- 环境设置 (整合所有必要依赖) ---
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

from utils.util_common import get_obj_from_str
from datapipe.datasets import create_dataset  # 新架构数据集创建工具
from trainer import TrainerAlphaSR            # 训练器类
from inference import Predictor               # 新架构推理预测器类

def main():
    # --- 1. 解析命令行参数 (保持所有原有参数不变) ---
    parser = argparse.ArgumentParser(description="Unified Main Runner for Anytime Spatio-Temporal Fusion Project")
    parser.add_argument('--cfg_path', type=str, default='configs/config_swinir.yaml', help='Path to the configuration file.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help="Execution mode: 'train' or 'test'.")
    parser.add_argument('--save_dir', type=str, default=None, help='Override the save directory specified in the config file.')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from the latest checkpoint in the save_dir.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Specify a direct path to a checkpoint for resuming or testing.')
    
    # 推理专用参数（保留不变）
    parser.add_argument('--input_dir', type=str, default=None, help="[Test Mode] Path to the input data directory for inference.")
    parser.add_argument('--output_dir', type=str, default=None, help="[Test Mode] Path to save inference results.")

    args = parser.parse_args()

    # --- 2. 加载和合并配置 (保持原有逻辑) ---
    configs = OmegaConf.load(args.cfg_path)
    if args.save_dir:
        configs.train.save_dir = args.save_dir

    # --- 3. 根据模式执行不同逻辑 (保持分支结构不变) ---
    if args.mode == 'train':
        run_training(args, configs)
    elif args.mode == 'test':
        run_testing(args, configs)

def run_training(args, configs):
    """执行训练流程（整合完整的恢复训练逻辑）"""
    print("\n" + "="*80 + "\n🚀 Starting Training Mode...\n" + "="*80)
    
    # --- 智能处理恢复逻辑 (保留v_final的完整逻辑，带详细打印) ---
    exp_dir = Path(configs.train.save_dir)
    resume_path = None
    
    # 最高优先级：用户直接指定了ckpt路径
    if args.ckpt_path:
        resume_path = args.ckpt_path
        print(f"[Main] Resuming from user-specified checkpoint: {resume_path}")
    # 第二优先级：用户要求恢复，自动查找最新ckpt
    elif args.resume:
        if not exp_dir.exists() or not any(exp_dir.iterdir()):
            print(f"[Main] '--resume' flag is set, but experiment directory {exp_dir} is empty or not found. Starting from scratch.")
        else:
            try:
                # 找到最新的运行目录
                latest_run_dir = max([d for d in exp_dir.iterdir() if d.is_dir()], key=os.path.getmtime)
                ckpt_dir = latest_run_dir / 'ckpts'
                if ckpt_dir.exists():
                    # 查找所有模型ckpt文件
                    ckpt_files = list(ckpt_dir.glob('model_*.pth'))
                    if ckpt_files:
                        # 按文件名中的数字找到最新的ckpt
                        latest_ckpt = max(ckpt_files, key=lambda p: int(p.stem.split('_')[-1]))
                        resume_path = str(latest_ckpt)
                        print(f"[Main] '--resume' flag is set. Automatically resuming from latest checkpoint: {resume_path}")
                    else:
                        print(f"[Main] '--resume' flag is set, but no checkpoints found in {ckpt_dir}. Starting from scratch.")
                else:
                    print(f"[Main] '--resume' flag is set, but ckpt directory not found in {latest_run_dir}. Starting from scratch.")
            except ValueError:
                print(f"[Main] '--resume' flag is set, but no subdirectories found in {exp_dir}. Starting from scratch.")

    # 将恢复路径写入配置
    configs.resume = resume_path 

    # --- 实例化并启动训练器 (保持原有逻辑) ---
    trainer_cls = get_obj_from_str(configs.trainer.target)
    trainer = trainer_cls(configs)
    trainer.train()

def run_testing(args, configs):
    """执行测试/推理流程（整合新架构的Predictor逻辑）"""
    print("\n" + "="*80 + "\n🚀 Starting Test (Inference) Mode...\n" + "="*80)
    
    # --- 前置检查 (保留原有校验逻辑) ---
    if not args.ckpt_path:
        print("❌ CRITICAL: For 'test' mode, a checkpoint path must be specified via '--ckpt_path'.")
        sys.exit(1)
    if not args.input_dir or not args.output_dir:
        print("❌ CRITICAL: For 'test' mode, '--input_dir' and '--output_dir' must be specified.")
        sys.exit(1)

    # --- 新架构核心逻辑：实例化Predictor ---
    predictor = Predictor(args.cfg_path, args.ckpt_path)
    
    # --- 创建Test DataLoader (适配新数据管道) ---
    # 使用与训练一致的配置创建数据集
    data_config = {
        'target': 'datapipe.datasets.AnytimeTemporalDataset',
        'params': {
            'lr_dir': str(Path(args.input_dir) / 'LR'),
            'hr_dir': str(Path(args.input_dir) / 'HR'),
            'need_path': True  # 确保返回路径以便按原路径保存结果
        }
    }
    test_dataset = create_dataset(data_config, parent_configs=predictor.configs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4  # 适配批量推理，可根据硬件调整
    )
    
    # --- 执行推理 (调用Predictor的推理方法) ---
    predictor.run_inference(test_loader, Path(args.output_dir))

if __name__ == '__main__':
    main()
