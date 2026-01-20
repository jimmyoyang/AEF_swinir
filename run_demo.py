# 文件路径: run_demo.py
# (已修改) - 这是一个用于快速验证【完整数据加载管道】的脚本。

import torch
from omegaconf import OmegaConf
from datapipe.datasets import create_dataset # 确保 create_dataset 函数可以被导入
from utils import util_common
import time

# ==============================================================================
#                      !!! 核心修改部分 !!!
# ==============================================================================
# [修改] 在这里配置您想要测试的样本数量
NUM_SAMPLES_TO_TEST = 10
# ==============================================================================


def run_pipeline_test():
    """
    执行一次“数据管道测试”：
    1.  加载配置。
    2.  使用训练集(或验证集)配置，但通过 'sample_num' 参数强制只加载少量样本。
    3.  创建模型。
    4.  将这些样本逐一送入模型，检查端到端流程是否会报错。
    """
    print("\n" + "="*80)
    print(f"🚀 Starting Data Pipeline Test for {NUM_SAMPLES_TO_TEST} samples...")
    print("="*80 + "\n")

    # 1. 加载主配置文件 (保持不变)
    try:
        # [注意] 请确保这个配置文件路径是正确的
        configs = OmegaConf.load('./configs/config_swinir.yaml')
        print("✅ Successfully loaded 'config_swinir.yaml'.")
    except FileNotFoundError:
        print("❌ CRITICAL: './configs/config_swinir.yaml' not found. Please check the path.")
        return

    # 2. [修改] 准备数据集（加载 N 个样本）
    print(f"\n--- [Step 1/3] Preparing Dataset ({NUM_SAMPLES_TO_TEST} samples) ---")
    
    try:
        # [修改] 我们将目标从 'val' 改为 'train'，以更真实地验证训练管道
        target_configs = configs.data.val  # 如果您想测试验证集，请取消注释这一行
        # target_configs = configs.data.train # 当前使用训练集配置

        # [修改] 动态设置要加载的样本数量
        target_configs.params.sample_num = NUM_SAMPLES_TO_TEST
        print(f"🔹 Set 'sample_num' to {NUM_SAMPLES_TO_TEST} for quick loading.")
        
        # 使用您项目中的 create_dataset 函数 (保持不变)
        demo_dataset = create_dataset(target_configs)
        # batch_size=1, 逐一样本测试
        demo_loader = torch.utils.data.DataLoader(demo_dataset, batch_size=1)
        
        print(f"✅ Successfully created dataset and loader for the '{'train' if 'train' in target_configs else 'val'}' set.")
        if len(demo_loader) != NUM_SAMPLES_TO_TEST:
             print(f"⚠️ WARNING: Loader contains {len(demo_loader)} samples, but {NUM_SAMPLES_TO_TEST} were requested. This can happen if the dataset has fewer samples than requested.")


    except Exception as e:
        print(f"❌ CRITICAL: Failed during data loading. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 准备模型 (保持不变)
    print("\n--- [Step 2/3] Preparing Model ---")
    try:
        model = util_common.instantiate_from_config(configs.model).cuda()
        model.eval() # 设置为评估模式
        print("✅ Successfully instantiated the model and moved it to GPU.")
    except Exception as e:
        print(f"❌ CRITICAL: Failed during model instantiation. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. [重构] 循环执行前向传播 (Forward Pass)
    print(f"\n--- [Step 3/3] Performing {len(demo_loader)} forward passes ---")
    
    total_time = 0
    try:
        # [修改] 使用 for 循环遍历所有加载的样本
        for i, sample in enumerate(demo_loader):
            print(f"\n--- Testing Sample {i+1}/{len(demo_loader)} ---")
            
            # 将字典中所有的张量都移动到GPU
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key] = value.cuda()
            
            with torch.no_grad():
                start_time = time.time()
                # 将整个样本字典传递给模型
                output = model(sample)
                end_time = time.time()
                total_time += (end_time - start_time)

            print(f"🔹 Input 'lr_sequence' shape: {sample['lr_sequence'].shape}")
            print(f"🔹 Output tensor shape: {output.shape}")
            print(f"✅ Forward pass for sample {i+1} completed in {end_time - start_time:.4f} seconds.")

    except Exception as e:
        print(f"❌ CRITICAL: An error occurred during the model's forward pass on sample {i+1}. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80)
    print("🎉 Pipeline Test Passed! Your data loading and model are compatible.")
    print(f"   Successfully processed {len(demo_loader)} samples in a total of {total_time:.4f} seconds.")
    print("   You can now confidently run 'python main.py' for the full training.")
    print("="*80 + "\n")

if __name__ == '__main__':
    run_pipeline_test()
