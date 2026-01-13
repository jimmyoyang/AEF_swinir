#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 CCDC 特征文件的脚本

使用方法:
    python generate_ccdc_features.py [config_path]
    
    如果不指定 config_path，默认使用 configs/config_swinir.yaml
"""
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from ccdc_class import run_ccdc_workflow


def generate_ccdc_features(config_path='configs/config_swinir.yaml'):
    """
    根据配置文件生成 CCDC 特征文件
    
    Args:
        config_path: 配置文件路径
    """
    print("="*60)
    print("CCDC 特征生成脚本")
    print("="*60)
    
    # 加载配置
    if not Path(config_path).exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        return
    
    configs = OmegaConf.load(config_path)
    
    # 检查 CCDC 配置
    if not hasattr(configs, 'ccdc'):
        print("[WARNING] 配置文件中没有 CCDC 配置，使用默认配置")
        ccdc_config = OmegaConf.create({
            'enabled': True,
            'ccdc_dir': './data/ccdc_features',
            'bands_to_process': {
                'b2': 0, 'b3': 1, 'b4': 2, 'b5': 3, 'b6': 4,
                'b7': 5, 'b8': 6, 'b10': 7, 'b11': 8
            },
            'coeffs_to_extract': ['a0', 'a1', 'b1']
        })
    else:
        ccdc_config = configs.ccdc
    
    if not ccdc_config.get('enabled', False):
        print("[INFO] CCDC 未启用，跳过特征生成")
        print("提示: 在配置文件中设置 ccdc.enabled: true 来启用 CCDC")
        return
    
    # 获取配置参数
    ccdc_base_dir = Path(ccdc_config.ccdc_dir)
    bands_config = dict(ccdc_config.bands_to_process)
    coeffs_config = list(ccdc_config.coeffs_to_extract)
    
    print(f"\n[配置信息]")
    print(f"  CCDC 目录: {ccdc_base_dir}")
    print(f"  处理的波段数: {len(bands_config)}")
    print(f"  提取的系数: {coeffs_config}")
    print(f"  特征通道数: {len(bands_config) * len(coeffs_config)}")
    
    # 确定数据目录结构
    data_splits = ['train', 'val', 'test']
    
    # 从配置中获取基础数据目录
    if hasattr(configs, 'data') and hasattr(configs.data, 'train'):
        base_lr_dir = Path(configs.data.train.params.lr_dir).parent
    else:
        # 默认路径
        base_lr_dir = Path('./data/processed_data')
    
    print(f"\n[数据目录]")
    print(f"  基础 LR 目录: {base_lr_dir}")
    
    # 处理每个数据集
    for split in data_splits:
        lr_dir = base_lr_dir / split / 'LR'
        output_dir = ccdc_base_dir / split / 'LR'
        
        if not lr_dir.exists():
            print(f"\n[WARNING] LR 目录不存在，跳过 {split}: {lr_dir}")
            continue
        
        # 检查目录中是否有文件
        lr_files = list(lr_dir.glob('*.tif'))
        if not lr_files:
            print(f"\n[WARNING] LR 目录中没有 .tif 文件，跳过 {split}: {lr_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理 {split.upper()} 数据集")
        print(f"{'='*60}")
        print(f"  输入目录: {lr_dir}")
        print(f"  输出目录: {output_dir}")
        print(f"  找到 {len(lr_files)} 个影像文件")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 运行 CCDC 工作流
            run_ccdc_workflow(
                raw_data_dir=str(lr_dir),
                output_dir=str(output_dir),
                bands_config=bands_config,
                coeffs_config=coeffs_config,
                use_multiprocessing=True
            )
            print(f"\n[SUCCESS] {split.upper()} 数据集处理完成")
        except Exception as e:
            print(f"\n[ERROR] 处理 {split.upper()} 数据集时发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("所有 CCDC 特征文件生成完成！")
    print("="*60)
    print(f"\n输出目录: {ccdc_base_dir}")
    print("提示: 在配置文件中设置 ccdc.enabled: true 来使用 CCDC 特征")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成 CCDC 特征文件')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_swinir.yaml',
        help='配置文件路径 (默认: configs/config_swinir.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        generate_ccdc_features(args.config)
    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断，退出...")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
