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
from inference import Predictor               # 新架构推理预测器类
from inference_srcnn import PredictorSRCNN, run_inference_srcnn  # SRCNN专用推理

SWINIR_MODEL_SIZE_PRESETS = {
    'default': {
        'embed_dim': 180,
        'depths': [6, 6, 6, 6],
        'num_heads': [6, 6, 6, 6],
        'head_dim': 6,
    },
    'base': {
        'embed_dim': 180,
        'depths': [6, 6, 6, 6],
        'num_heads': [6, 6, 6, 6],
        'head_dim': 6,
    },
    'large': {
        'embed_dim': 240,
        'depths': [6, 6, 6, 6, 6, 6, 6, 6, 6],
        'num_heads': [8, 8, 8, 8, 8, 8, 8, 8, 8],
        'head_dim': 8,
    },
}

def detect_model_type(configs):
    """检测模型类型（SRCNN或其他）"""
    model_target = configs.model.target.lower()
    if 'srcnn' in model_target:
        return 'srcnn'
    return 'default'

def apply_model_size_override(configs, model_size):
    """Apply an optional SwinIR model-size preset without changing config-only runs."""
    if not model_size:
        return configs

    normalized_size = model_size.strip().lower()
    if normalized_size in ('config', 'none'):
        return configs

    if normalized_size not in SWINIR_MODEL_SIZE_PRESETS:
        valid_sizes = ', '.join(sorted(SWINIR_MODEL_SIZE_PRESETS.keys()))
        raise ValueError(f"Unknown --model_size '{model_size}'. Valid options: {valid_sizes}, config, none")

    model_target = configs.model.target.lower()
    if 'swinir' not in model_target:
        raise ValueError(f"--model_size is only supported for SwinIR targets, got: {configs.model.target}")

    preset = SWINIR_MODEL_SIZE_PRESETS[normalized_size]
    params = configs.model.params
    params.embed_dim = preset['embed_dim']
    params.depths = list(preset['depths'])
    params.num_heads = list(preset['num_heads'])

    if 'cross_num_heads' in params:
        params.cross_num_heads = preset['head_dim']
    if 'pos_emb_dim' in params and int(params.get('pos_emb_dim', 0) or 0) > 0:
        params.pos_emb_dim = preset['embed_dim']
    if 'temporal_attention_params' in params and params.temporal_attention_params is not None:
        params.temporal_attention_params.num_heads = preset['head_dim']

    print(
        f"[Main] Applied SwinIR model_size={normalized_size}: "
        f"embed_dim={params.embed_dim}, layers={len(params.depths)}, heads={list(params.num_heads)}"
    )
    return configs

def main():
    # --- 1. 解析命令行参数 (保持所有原有参数不变) ---
    parser = argparse.ArgumentParser(description="Unified Main Runner for Anytime Spatio-Temporal Fusion Project")
    parser.add_argument('--cfg_path', type=str, default='configs/config_swinir.yaml', help='Path to the configuration file.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help="Execution mode: 'train' or 'test'.")
    parser.add_argument('--save_dir', type=str, default=None, help='Override the save directory specified in the config file.')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from the latest checkpoint in the save_dir.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Specify a direct path to a checkpoint for resuming or testing.')
    parser.add_argument('--model_size', type=str, default=None, help="Optional SwinIR size preset: default/base/large. Omit to use the YAML as-is.")
    
    # 推理专用参数（保留不变）
    parser.add_argument('--input_dir', type=str, default=None, help="[Test Mode] Path to the input data directory for inference.")
    parser.add_argument('--output_dir', type=str, default=None, help="[Test Mode] Path to save inference results.")

    args = parser.parse_args()

    # --- 2. 加载和合并配置 (保持原有逻辑) ---
    configs = OmegaConf.load(args.cfg_path)
    configs = apply_model_size_override(configs, args.model_size)
    if args.save_dir:
        configs.train.save_dir = args.save_dir

    # --- 3. 根据模式执行不同逻辑 (保持分支结构不变) ---
    if args.mode == 'train':
        run_training(args, configs)
    elif args.mode == 'test':
        run_testing(args, configs)

def run_training(args, configs):
    """执行训练流程（兼容SRCNN和其他模类）"""
    print("\n" + "="*80 + "\n🚀 Starting Training Mode...\n" + "="*80)
    
    # 检测模型类型
    model_type = detect_model_type(configs)
    print(f"[Main] Model type detected: {model_type}")
    
    # --- 智能处理恢复逻辑 ---
    exp_dir = Path(configs.train.save_dir)
    resume_path = None
    
    if args.ckpt_path:
        resume_path = args.ckpt_path
        print(f"[Main] Resuming from user-specified checkpoint: {resume_path}")
    elif args.resume:
        if not exp_dir.exists() or not any(exp_dir.iterdir()):
            print(f"[Main] '--resume' flag is set, but experiment directory {exp_dir} is empty or not found. Starting from scratch.")
        else:
            try:
                # 找到最新的运行目录
                latest_run_dir = max([d for d in exp_dir.iterdir() if d.is_dir()], key=os.path.getmtime)
                ckpt_dir = latest_run_dir / 'ckpts'
                if ckpt_dir.exists():
                    # 查找所有带数字迭代号的模型ckpt文件，避免 model_best.pth 干扰自动恢复。
                    ckpt_files = [
                        p for p in ckpt_dir.glob('model_*.pth')
                        if p.stem.split('_')[-1].isdigit()
                    ]
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

    configs.resume = resume_path 

    # --- 根据配置选择训练器 ---
    if hasattr(configs, 'trainer') and 'target' in configs.trainer:
        trainer_target = configs.trainer.target
    elif model_type == 'srcnn':
        trainer_target = 'trainer_srcnn.TrainerSRCNN'
    else:
        trainer_target = 'trainer.TrainerAlphaSR'

    print(f"[Main] Using trainer: {trainer_target}")
    trainer_cls = get_obj_from_str(trainer_target)
    trainer = trainer_cls(configs)
    
    trainer.train()

def run_testing(args, configs):
    """执行测试/推理流程（兼容SRCNN和其他模型）"""
    print("\n" + "="*80 + "\n🚀 Starting Test (Inference) Mode...\n" + "="*80)
    
    # 检测模型类型
    model_type = detect_model_type(configs)
    print(f"[Main] Model type detected: {model_type}")
    
    # --- 前置检查 ---
    if not args.ckpt_path:
        print("❌ CRITICAL: For 'test' mode, a checkpoint path must be specified via '--ckpt_path'.")
        sys.exit(1)
    if not args.input_dir or not args.output_dir:
        print("❌ CRITICAL: For 'test' mode, '--input_dir' and '--output_dir' must be specified.")
        sys.exit(1)

    # --- 创建Test DataLoader ---
    if model_type == 'srcnn':
        data_config = {
            'target': 'datapipe.datasets.PreprocessedTileDataset',
            'params': {
                'lr_dir': str(Path(args.input_dir) / 'LR'),
                'hr_dir': str(Path(args.input_dir) / 'HR'),
                'need_path': True
            }
        }
    else:
        if not hasattr(configs.data, 'test'):
            print("❌ CRITICAL: Non-SRCNN test mode requires configs.data.test.")
            sys.exit(1)
        data_config = OmegaConf.to_container(configs.data.test, resolve=True)
        data_config.setdefault('params', {})
        data_config['params']['lr_dir'] = str(Path(args.input_dir) / 'LR')
        data_config['params']['hr_dir'] = str(Path(args.input_dir) / 'HR')
        data_config['params']['need_path'] = True

    test_dataset = create_dataset(data_config, parent_configs=configs)

    # 动态检测输入通道数
    try:
        sample = test_dataset[0]
        if 's1' in sample:
            probe = sample['s1']
        elif 'lr' in sample:
            probe = sample['lr']
        elif 'lr_sequence' in sample:
            probe = sample['lr_sequence']
        else:
            raise KeyError(f"No valid input key in test sample. Keys: {list(sample.keys())}")

        if probe.dim() == 5:
            dynamic_in_chans = int(probe.shape[2])
        elif probe.dim() == 4:
            dynamic_in_chans = int(probe.shape[1])
        elif probe.dim() == 3:
            dynamic_in_chans = int(probe.shape[0])
        else:
            raise ValueError(f"Unsupported test sample shape: {tuple(probe.shape)}")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to infer in_chans from test dataset. Error: {e}")
        sys.exit(1)

    # --- 根据模型类型选择推理器 ---
    if model_type == 'srcnn':
        print("[Main] Using SRCNN predictor...")
        run_inference_srcnn(args, configs)
    else:
        print("[Main] Using default (SwinIR) predictor...")
        predictor = Predictor(configs, args.ckpt_path, dynamic_in_chans)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=4
        )
        predictor.run_inference(test_loader, Path(args.output_dir))

if __name__ == '__main__':
    main()
