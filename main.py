# 文件路径: main.py
# (v2 - 最终的、统一的、独立的程序入口)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse, os, sys, yaml, torch
from omegaconf import OmegaConf
from pathlib import Path

# --- 环境设置 ---
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))
from utils.util_common import get_obj_from_str
from datapipe.datasets import create_dataset
# 假设 predictor.py 中有 Predictor 类
from predictor import Predictor

def main():
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Unified Main Runner for SR Project")
    parser.add_argument('--cfg_path', type=str, default='configs/local_config.yaml', help='Path to the configuration file.')
    parser.add_argument('--save_dir', type=str, default=None, help='Override the save directory specified in the config file.')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from the latest checkpoint in the save_dir.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Specify a direct path to a checkpoint to resume from (highest priority).')
    parser.add_argument('--run_test', action='store_true', help='Run inference on the test set after training is complete.')
    args = parser.parse_args()

    # --- 2. 加载和合并配置 ---
    configs = OmegaConf.load(args.cfg_path)
    if args.save_dir:
        # 命令行指定的 save_dir 优先级最高
        configs.train.save_dir = args.save_dir

    # --- 3. 智能处理恢复逻辑 ---
    exp_dir = Path(configs.train.save_dir)
    resume_path = None
    if args.ckpt_path: # 最高优先级：直接指定了ckpt路径
        resume_path = args.ckpt_path
        print(f"[Main] Resuming from user-specified checkpoint: {resume_path}")
    elif args.resume: # 第二优先级：用户要求恢复，自动查找
        # 自动查找最新的run目录
        if not exp_dir.exists() or not any(exp_dir.iterdir()):
             print(f"[Main] '--resume' flag is set, but experiment directory {exp_dir} is empty or not found. Starting from scratch.")
        else:
            try:
                # 找到最新的一个run
                latest_run_dir = max([d for d in exp_dir.iterdir() if d.is_dir()], key=os.path.getmtime)
                ckpt_dir = latest_run_dir / 'ckpts'
                if ckpt_dir.exists():
                    ckpt_files = list(ckpt_dir.glob('model_*.pth'))
                    if ckpt_files:
                        latest_ckpt = max(ckpt_files, key=lambda p: int(p.stem.split('_')[-1]))
                        resume_path = str(latest_ckpt)
                        print(f"[Main] '--resume' flag is set. Automatically resuming from latest checkpoint: {resume_path}")
                    else:
                        print(f"[Main] '--resume' flag is set, but no checkpoints found in {ckpt_dir}. Starting from scratch.")
                else:
                    print(f"[Main] '--resume' flag is set, but ckpt directory not found in {latest_run_dir}. Starting from scratch.")
            except ValueError:
                print(f"[Main] '--resume' flag is set, but no subdirectories found in {exp_dir}. Starting from scratch.")

    # 将最终确定的恢复路径写入配置，以便 trainer 使用
    configs.resume = resume_path 

    # --- 4. 实例化并启动训练器 ---
    trainer_cls = get_obj_from_str(configs.trainer.target)
    trainer = trainer_cls(configs)
    trainer.train()

    # --- 5. (可选) 训练后在测试集上运行推理 ---
    if args.run_test:
        print("\n" + "="*80 + "\n[Main] Training finished. Running inference on the TEST set...\n" + "="*80)
        
        # 训练完成后，save_dir 已经是具体的 run 目录
        run_dir = trainer.save_dir
        best_ckpt_path = run_dir / 'model_best.pth'
        if not best_ckpt_path.exists():
             print(f"[Main] WARNING: 'model_best.pth' not found in {run_dir}. Cannot run test inference.")
             return

        # 实例化 Predictor
        predictor = Predictor(configs, str(best_ckpt_path))
        
        # 创建 Test DataLoader
        test_dataset = create_dataset(configs.data.test)
        # 使用验证集的batch size，或者在config中定义test的batch size
        test_batch_size = configs.train.batch[1] if len(configs.train.batch) > 1 else 1
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)
        
        # 定义输出目录并执行
        test_out_dir = run_dir / "inference_results_test"
        predictor.run_inference(test_loader, test_out_dir)

if __name__ == '__main__':
    main()
