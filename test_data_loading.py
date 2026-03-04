#!/usr/bin/env python
"""
测试数据加载是否正常工作
"""
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf
from datapipe.datasets import AnytimeTemporalDataset

def test_data_loading():
    """测试数据加载"""
    print("="*80)
    print("测试数据加载")
    print("="*80)
    
    # 加载配置
    config_path = "configs/config_baseline_swinir.yaml"
    print(f"\n📄 加载配置: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # 检查数据路径
    lr_dir = Path(cfg.data.train.params.lr_dir)
    hr_dir = Path(cfg.data.train.params.hr_dir)
    print(f"\n📁 数据路径:")
    print(f"  LR: {lr_dir} (存在: {lr_dir.exists()})")
    print(f"  HR: {hr_dir} (存在: {hr_dir.exists()})")
    
    if not lr_dir.exists() or not hr_dir.exists():
        print("❌ 数据路径不存在！")
        return False
    
    # 创建数据集
    print(f"\n🔧 创建数据集...")
    try:
        dataset = AnytimeTemporalDataset(
            lr_dir=str(lr_dir),
            hr_dir=str(hr_dir),
            sample_num=cfg.data.train.params.sample_num,
            parent_configs=dict(cfg)
        )
        print(f"✅ 数据集创建成功，大小: {len(dataset)}")
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试加载第一个样本
    print(f"\n📦 测试加载第一个样本...")
    try:
        sample = dataset[0]
        print(f"✅ 样本加载成功")
        print(f"  样本键: {list(sample.keys())}")
        print(f"  lr_sequence shape: {sample['lr_sequence'].shape}")
        print(f"  timestamps shape: {sample['timestamps'].shape}")
        print(f"  gt shape: {sample['gt'].shape}")
        print(f"  mask shape: {sample['mask'].shape}")
        
        # 检查维度匹配
        T = sample['lr_sequence'].shape[0]
        if sample['timestamps'].shape[0] != T:
            print(f"❌ timestamps 长度不匹配: {sample['timestamps'].shape[0]} != {T}")
            return False
        if sample['mask'].shape[0] != T:
            print(f"❌ mask 长度不匹配: {sample['mask'].shape[0]} != {T}")
            return False
        
        print(f"✅ 所有维度匹配")
        
        # 检查是否有 None 值
        for key, value in sample.items():
            if value is None:
                print(f"❌ {key} 是 None！")
                return False
        
        print(f"✅ 没有 None 值")
        
    except Exception as e:
        print(f"❌ 样本加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 DataLoader
    print(f"\n🔄 测试 DataLoader...")
    try:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0  # 使用单进程避免问题
        )
        
        batch = next(iter(dataloader))
        print(f"✅ DataLoader 工作正常")
        print(f"  batch keys: {list(batch.keys())}")
        print(f"  lr_sequence shape: {batch['lr_sequence'].shape}")
        print(f"  timestamps shape: {batch['timestamps'].shape}")
        print(f"  gt shape: {batch['gt'].shape}")
        print(f"  mask shape: {batch['mask'].shape}")
        
    except Exception as e:
        print(f"❌ DataLoader 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n✅ 所有测试通过！数据加载正常。")
    return True

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
