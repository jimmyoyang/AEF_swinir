# 文件路径: run_demo.py
# 这是一个用于快速验证完整数据管道和模型的独立脚本。

import torch
from omegaconf import OmegaConf
from datapipe.datasets import create_dataset
from utils import util_common
import time
NUM_SAMPLE_TO_TEST =10

def run_pipeline_test():
    """ 
    执行一次“冒烟测试”：
    1. 加载配置
    2. 使用验证集配置，但通过‘sample_num’参数强制只加载少量样本
    3. 创建模型。
    4. 将这些样本逐一送入模型，检查是否会报错。
    """
    print("\n" + "="*80)
    print("🚀 Starting Data Pipeline Test for {NUM_SAMPLE_TO_TEST} samples...")
    print("="*80 + "\n")

    # 1. 加载主配置文件
    try:
        configs = OmegaConf.load('/mnt/lm_data_afs/wangzining/charles/AEF_swinir/configs/config_swinir.yaml')
        print("✅ Successfully loaded 'config_swinir.yaml'.")
    except FileNotFoundError:
        print("❌ CRITICAL: 'config_swinir.yaml' not found in the project root.")
        return

    # 2. 准备数据集（加载N个样本）
    print("\n--- [Step 1/3] Preparing Dataset {NUM_SAMPLE_TO_TEST} samples---")
    target_configs = configs.data.val
    
    # 强制只使用1个样本进行测试，这样加载会非常快
    target_configs.params.sample_num = NUM_SAMPLE_TO_TEST
    
    try:
        # 使用您项目中的 create_dataset 函数
        
        demo_dataset = create_dataset(target_configs)
        #batch_size=1 逐样本测试
        demo_loader = torch.utils.data.DataLoader(demo_dataset, batch_size=1)
        
        # 从加载器中取出一个样本
        # sample = next(iter(demo_loader))

        print("✅ Successfully loaded 1 sample from the validation set.")
        if len(demo_loader)!=NUM_SAMPLE_TO_TEST:
            print(f"WARNING: Loader contains {len(demo_loader)} samples, but {NUM_SAMPLE_TO_TEST} were requested. This can happen if the dataset has fewer samoles than requested.")

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

    # 4. [重构]循环执行前向传播 (Forward Pass)
    print(f"\n--- [Step 3/3] Performing {len(demo_loader)} forward pass ---")
    total_time = 0
    try:
        for i,samoke in enumerate(demo_loader):
            print(f"\n--- Testing Sample {i+1}/{len(demo_loader)}---")
            #将字典中的所有张量都移动到GPU
            for key,value in sample.items():
                if isinstance(value,torch.Tensor):
                    sample[key] = value.cuda()
            with torch.no_grad():
                start_time=time.time()
                #将整个样本字典传递给模型
                output= model(sample)
                end_time=time.time()
                total_time+=(end_time-start_time)
            print(f"🔹 Input 'lr_sequence' shape:{sample['lr_sequence'].shape}")
            print(f"🔹 {output.shape}")
            print(f"✅ Forward pass for sample {i+1} completed in {end_time - start_time:.4f}seconds.")
    except Exception as e:
        print(f"❌ CRITICAL: An error occurred during the model's forward pass on sample {i+1} . Error: {e}")
        import traceback
        traceback.print_exc()
        return


    print("\n" + "="*80)
    print("🎉 Pipeline Test Passed! Your data loading and model are compatible.")
    print(f"✅ Successfully processed {len(demo_loader)} samples in a total of {total_time:.4f}seconds.")
    print("   You can now confidently run 'python main.py' for the full training.")
    print("="*80 + "\n")


if __name__ == '__main__':
    run_pipeline_test()
