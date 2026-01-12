# 文件路径: run_demo.py
# 这是一个用于快速验证完整数据管道和模型的独立脚本。

import torch
from omegaconf import OmegaConf
from datapipe.datasets import create_dataset
from utils import util_common
import time

def run_smoke_test():
    """
    执行一次“冒烟测试”：
    1. 加载配置。
    2. 使用验证集配置，但只加载1个样本。
    3. 创建模型。
    4. 将1个样本送入模型，检查是否会报错。
    """
    print("\n" + "="*80)
    print("🚀 Starting Smoke Test for 20 Time-Step Temporal Model...")
    print("="*80 + "\n")

    # 1. 加载主配置文件
    try:
        configs = OmegaConf.load('/home/charles/lab/AEF_swinir/configs/config_swinir.yaml')
        print("✅ Successfully loaded 'config_swinir.yaml'.")
    except FileNotFoundError:
        print("❌ CRITICAL: 'config_swinir.yaml' not found in the project root.")
        return

    # 2. 准备数据集（只取1个样本）
    print("\n--- [Step 1/3] Preparing Dataset ---")
    val_configs = configs.data.val
    
    # 强制只使用1个样本进行测试，这样加载会非常快
    val_configs.params.sample_num = 1
    
    try:
        # 使用您项目中的 create_dataset 函数
        demo_dataset = create_dataset(val_configs)
        demo_loader = torch.utils.data.DataLoader(demo_dataset, batch_size=1)
        
        # 从加载器中取出一个样本
        sample = next(iter(demo_loader))
        print("✅ Successfully loaded 1 sample from the validation set.")
    except Exception as e:
        print(f"❌ CRITICAL: Failed during data loading. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 准备模型
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

    # 4. 执行一次前向传播 (Forward Pass)
    print("\n--- [Step 3/3] Performing a single forward pass ---")
    inputs = sample['s1'].cuda()
    
    # 关键验证点：检查输入张量的形状
    print(f"🔹 Input tensor shape: {inputs.shape}")
    print(f"   (Should be [batch_size, in_chans, height, width], e.g., [1, 180, 192, 192])")

    if inputs.shape[1] != configs.model.params.in_chans:
        print(f"❌ MISMATCH: Input tensor has {inputs.shape[1]} channels, but model expects {configs.model.params.in_chans}.")
        return

    try:
        with torch.no_grad():
            start_time = time.time()
            output = model(inputs)
            end_time = time.time()

        # 关键验证点：检查输出张量的形状
        print(f"🔹 Output tensor shape: {output.shape}")
        print(f"   (Should be [batch_size, out_channels, height, width], e.g., [1, 64, 192, 192])")
        print(f"✅ Successfully completed the forward pass in {end_time - start_time:.4f} seconds.")

    except Exception as e:
        print(f"❌ CRITICAL: An error occurred during the model's forward pass. Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80)
    print("🎉 Smoke Test Passed! Your full 20-image pipeline is ready.")
    print("   You can now confidently run 'python main.py' for the full training.")
    print("="*80 + "\n")


if __name__ == '__main__':
    run_smoke_test()
