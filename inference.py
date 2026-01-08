# 文件路径: inference.py
# (最终融合版)

import argparse
from pathlib import Path
import yaml
from torch.utils.data import DataLoader

# 【融合】导入我们自己的模块，而不是原版的
from utils.util_common import get_obj_from_str
from predictor import PredictorSRCNN 

def get_parser(**parser_kwargs):
    """
    【借鉴】从原版借鉴的、专业的命令行参数解析器。
    我们根据自己的需求，简化并调整了参数。
    """
    parser = argparse.ArgumentParser(description="SRCNN Local Inference Script", **parser_kwargs)
    parser.add_argument(
        "-i", "--in_path", 
        type=str, 
        required=True, 
        help="Path to the input LR image folder."
    )
    parser.add_argument(
        "-o", "--out_path", 
        type=str, 
        required=True, 
        help="Path to save the output results and metrics."
    )
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True, 
        help="Path to the model's configuration file (.yaml)."
    )
    # 【轻量化】移除了原版中不必要的 task, seed, bs 等参数，因为它们可以在配置文件中定义。
    args = parser.parse_args()
    return args

def get_configs(args):
    """
    【借鉴】从原版借鉴的、用于加载和准备配置的函数。
    """
    # 1. 加载 YAML 配置文件
    with open(args.config_path, 'r', encoding='utf-8') as f:
        # 使用 argparse.Namespace 可以让我们用 configs.model 这样的点号来访问
        configs = argparse.Namespace(**yaml.safe_load(f))
    
    # 2. 确保输出目录存在
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    
    return configs

def main():
    """
    【借鉴】从原版借鉴的、结构清晰的主执行函数。
    """
    # --- 步骤 1: 获取参数和配置 ---
    args = get_parser()
    configs = get_configs(args)
    print(f"[Inference INFO] Configuration loaded from: {args.config_path}")

    # --- 步骤 2: 构建数据集和加载器 (我们的定制逻辑) ---
    print("[Inference INFO] Building data loader...")
    # 从配置文件中获取 'test' 部分的参数
    data_params = configs.data['test']['params']
    data_params['lr_dir'] = args.in_path
    
    # 自动检查是否存在对应的 HR 文件夹用于评估
    hr_path = Path(args.in_path).parent / 'HR'
    if hr_path.exists():
        data_params['hr_dir'] = str(hr_path)
        print(f"  - Found corresponding HR directory for evaluation: {hr_path}")
    else:
        # 如果没有HR文件夹，则将 hr_dir 设为 None，Dataset内部会处理这种情况
        data_params['hr_dir'] = None 
        print(f"  - No HR directory found. Evaluation will be skipped.")

    # 使用我们自己的 PreprocessedTileDataset
    dataset = get_obj_from_str(configs.data['test']['target'])(**data_params)
    test_loader = DataLoader(
        dataset, 
        batch_size=1, # 推理时 batch size 通常为 1
        shuffle=False, 
        num_workers=configs.train.get('num_workers', 0)
    )

    # --- 步骤 3: 【核心修改】创建并运行我们自己的 PredictorSRCNN ---
    print("[Inference INFO] Initializing predictor...")
    predictor = PredictorSRCNN(configs, args.ckpt_path)
    
    # 启动推理流程
    predictor.run_inference(test_loader, args.out_path)

if __name__ == '__main__':
    main()
