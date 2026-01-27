# 文件路径: run_integration_test.py
# 一个用于快速验证端到端流程的集成测试脚本

import os
import sys
import subprocess
from pathlib import Path
from omegaconf import OmegaConf
import shutil

# --- 环境设置 ---
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

# 假设您的主入口、训练器、推理函数都在正确的位置
from main import run_training, run_testing

def main():
    """
    集成测试主函数，按顺序执行训练和推理。
    """
    print("="*80)
    print("🚀 STARTING END-TO-END INTEGRATION TEST 🚀")
    print("="*80)

    # --- 1. 定义测试配置和路径 ---
    TEST_CONFIG_PATH = 'configs/config_integration_test.yaml'
    
    # 从配置文件中动态读取日志和数据路径
    try:
        configs = OmegaConf.load(TEST_CONFIG_PATH)
        test_save_dir = Path(configs.train.save_dir) / configs.train.experiment_name
        test_input_dir = Path(configs.data.test.params.lr_dir).parent.parent # 获取 .../processed_data/test
        test_output_dir = test_save_dir / "inference_results"
    except Exception as e:
        print(f"❌ CRITICAL: Failed to load test config file '{TEST_CONFIG_PATH}'. Error: {e}")
        sys.exit(1)

    # --- 清理旧的测试日志 (保证每次测试都是全新的) ---
    # 获取分布式环境下的rank，只有主进程(rank 0)执行清理操作
    rank = int(os.environ.get('LOCAL_RANK', 0))
    if rank == 0:
        if test_save_dir.exists():
            print(f"🧹 Cleaning up old test directory: {test_save_dir}")
            shutil.rmtree(test_save_dir)
        print(f"📦 Creating new test directory: {test_save_dir}")
        test_save_dir.mkdir(parents=True, exist_ok=True)
        test_output_dir.mkdir(exist_ok=True)

    # 在分布式环境中，等待主进程完成清理
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        import torch.distributed as dist
        dist.barrier()

    # --- 2. 运行一个极短的训练流程 ---
    print("\n" + "-"*30 + " STAGE 1: TESTING TRAINING PIPELINE " + "-"*30)
    try:
        # 我们复用 main.py 的逻辑，但通过命令行参数来控制
        # 伪造一个 argparse 的 Namespace 对象
        train_args = argparse.Namespace(
            cfg_path='configs/'+configs.defaults[0],
            mode='train',
            save_dir=str(test_save_dir), # 确保保存到测试目录
            resume=False,          # 从头开始训练
            ckpt_path=None
        )
        # 加载测试配置
        train_configs = OmegaConf.load(train_args.cfg_path)
        train_configs.train.save_dir = train_args.save_dir # 覆盖保存目录

        # 调用训练流程
        run_training(train_args, train_configs)

        print("\n✅ STAGE 1: TRAINING PIPELINE TEST PASSED!")
    except Exception as e:
        print(f"\n❌ STAGE 1: TRAINING PIPELINE TEST FAILED! Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 3. 运行一个极短的推理流程 ---
    # 只有主进程(rank 0)需要执行推理测试
    if rank == 0:
        print("\n" + "-"*30 + " STAGE 2: TESTING INFERENCE PIPELINE " + "-"*30)
        try:
            # 找到刚刚训练流程中保存的最后一个检查点
            latest_run_dir = max([d for d in test_save_dir.iterdir() if d.is_dir()], key=os.path.getmtime)
            ckpt_dir = latest_run_dir / 'ckpts'
            latest_ckpt = max(list(ckpt_dir.glob('model_*.pth')), key=os.path.getctime)
            
            print(f"Found checkpoint for inference: {latest_ckpt}")

            # 伪造推理所需的命令行参数
            test_args = argparse.Namespace(
                cfg_path=TEST_CONFIG_PATH,
                mode='test',
                ckpt_path=str(latest_ckpt),
                input_dir=str(test_input_dir),
                output_dir=str(test_output_dir)
            )
            # 加载测试配置
            test_configs = OmegaConf.load(test_args.cfg_path)
            
            # 调用推理流程
            run_testing(test_args, test_configs)

            # 检查是否成功生成了输出文件
            if not any(test_output_dir.iterdir()):
                 raise RuntimeError("Inference ran but no output files were generated.")

            print("\n✅ STAGE 2: INFERENCE PIPELINE TEST PASSED!")
        except Exception as e:
            print(f"\n❌ STAGE 2: INFERENCE PIPELINE TEST FAILED! Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "="*80)
    print("🎉 CONGRATULATIONS! END-TO-END INTEGRATION TEST SUCCESSFULLY COMPLETED! 🎉")
    print("="*80)

if __name__ == '__main__':
    # 【新增】为 argparse.Namespace 创建一个 dummy 类，避免在 main.py 中找不到它
    import argparse
    main()


# # 文件路径: run_integration_test.py
# # (一个全面的、用于调试和验证全流程的联调测试脚本)

# import os
# import sys
# import yaml
# import torch
# from pathlib import Path
# from omegaconf import OmegaConf
# import matplotlib.pyplot as plt
# import numpy as np

# # --- 环境设置 ---
# project_root = Path(__file__).parent.resolve()
# sys.path.append(str(project_root))
# from utils.util_common import get_obj_from_str
# from datapipe.datasets import create_dataset
# from trainer import TrainerAlphaSR # 假设您的Trainer类在这里

# # ==============================================================================
# # 1. 核心检查函数
# # ==============================================================================

# def check_data_loading(configs):
#     """检查数据加载和预处理是否正确。"""
#     print("\n--- [Check 1/4] Verifying Data Loading & Preprocessing ---")
#     try:
#         # 只加载一个训练样本和一个验证样本进行检查
#         train_conf = configs.data.train
#         train_conf.params.sample_num = 1
#         train_dataset = create_dataset(train_conf, parent_configs=configs)
#         train_sample = train_dataset[0]

#         val_conf = configs.data.val
#         val_conf.params.sample_num = 1
#         val_dataset = create_dataset(val_conf, parent_configs=configs)
#         val_sample = val_dataset[0]

#         print("✅ Data loading seems OK.")
        
#         # 检查样本结构和数据类型
#         assert isinstance(train_sample, dict), "Sample should be a dictionary."
#         required_keys = ['lr_sequence', 'timestamps', 'gt']
#         assert all(key in train_sample for key in required_keys), f"Sample missing keys. Required: {required_keys}"
#         assert isinstance(train_sample['lr_sequence'], torch.Tensor), "`lr_sequence` should be a Tensor."
        
#         print("✅ Sample structure and types are correct.")
        
#         # 可视化一个样本以供人工检查
#         lr_img = train_sample['lr_sequence'][0].numpy() # 取第一个时相
#         gt_img = train_sample['gt'].numpy()
        
#         # 假设是多通道，我们只显示前3个通道的归一化图
#         def norm_for_vis(img):
#             img = img.transpose(1, 2, 0) # C,H,W -> H,W,C
#             return (img - img.min()) / (img.max() - img.min()) if (img.max() - img.min()) > 1e-6 else img

#         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#         fig.suptitle("Data Loading Sanity Check (Normalized)")
#         axes[0].imshow(norm_for_vis(lr_img[:3])); axes[0].set_title(f"LR Sample (Shape: {lr_img.shape})")
#         axes[1].imshow(norm_for_vis(gt_img[:3])); axes[1].set_title(f"GT Sample (Shape: {gt_img.shape})")
#         plt.savefig(Path(configs.train.save_dir) / "data_loading_check.png")
#         plt.close()
#         print(f"✅ A visualization of one sample has been saved to '{configs.train.save_dir}/data_loading_check.png'")
        
#         return True
#     except Exception as e:
#         print(f"❌ FAILED: An error occurred during data loading check. Error: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def check_model_forward_pass(configs):
#     """检查模型是否能正确处理一个batch的数据。"""
#     print("\n--- [Check 2/4] Verifying Model Forward Pass ---")
#     try:
#         # 使用微型数据集创建一个DataLoader
#         dataset = create_dataset(configs.data.train, parent_configs=configs)
#         loader = torch.utils.data.DataLoader(dataset, batch_size=configs.train.batch[0])
#         sample_batch = next(iter(loader))
        
#         # 实例化模型
#         model = get_obj_from_str(configs.model.target)(**configs.model.params).cuda()
#         model.eval()
        
#         # 将数据移动到GPU并送入模型
#         inputs = {k: v.cuda() for k, v in sample_batch.items() if isinstance(v, torch.Tensor)}
        
#         with torch.no_grad():
#             output = model(inputs)
            
#         # 检查输出维度
#         b, _, h_out, w_out = output.shape
#         h_in, w_in = configs.model.params.img_size, configs.model.params.img_size
#         scale = configs.model.params.upscale
#         assert b == configs.train.batch[0], "Output batch size mismatch."
#         assert h_out == h_in * scale and w_out == w_in * scale, "Output spatial dimensions are incorrect."
        
#         print("✅ Model forward pass is successful.")
#         print(f"   - Input batch LR shape: {sample_batch['lr_sequence'].shape}")
#         print(f"   - Output batch HR shape: {output.shape}")
#         return True
#     except Exception as e:
#         print(f"❌ FAILED: An error occurred during model forward pass. Error: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def check_training_step(configs):
#     """检查一个完整的训练步骤（包括损失计算和反向传播）是否能跑通。"""
#     print("\n--- [Check 3/4] Verifying a Single Training Step ---")
#     try:
#         # 创建一个Trainer实例
#         trainer = TrainerAlphaSR(configs)
        
#         # 手动执行一个训练步骤
#         trainer.model.train()
#         data = next(iter(trainer.dataloaders['train']))
#         data = trainer.prepare_data(data)
        
#         trainer.optimizer.zero_grad()
        
#         # 使用混合精度
#         with torch.cuda.amp.autocast(enabled=configs.train.mixed_precision):
#             predictions = trainer.model(data)
#             loss, loss_dict = trainer.criterion(predictions, data['gt'])
        
#         trainer.scaler.scale(loss).backward()
#         trainer.scaler.step(trainer.optimizer)
#         trainer.scaler.update()
        
#         print("✅ A single training step (forward, loss, backward) completed successfully.")
#         print(f"   - Calculated Total Loss: {loss.item():.4f}")
#         print(f"   - Loss Components: {loss_dict}")
#         return True
#     except Exception as e:
#         print(f"❌ FAILED: An error occurred during the training step. Error: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def run_short_training_and_validation(configs):
#     """运行一个极短的训练-验证流程。"""
#     print("\n--- [Check 4/4] Running a Short End-to-End Training & Validation Loop ---")
#     try:
#         trainer = TrainerAlphaSR(configs)
#         trainer.train() # Trainer内部会根据iterations=100来运行
        
#         print("\n✅ Short training and validation loop completed.")
#         print(f"   - Check the logs and images in: '{configs.train.save_dir}'")
#         print("   - Look for 'training.log' for loss values.")
#         print("   - Look for images in the 'images/val' subdirectory to visually inspect results.")
#         return True
#     except Exception as e:
#         print(f"❌ FAILED: An error occurred during the end-to-end loop. Error: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# # ==============================================================================
# # 2. 主执行函数
# # ==============================================================================

# def main():
#     """主执行函数，按顺序运行所有检查。"""
#     config_path = 'configs/config_integration_test.yaml'
#     print("="*80)
#     print(f"🚀 Starting Full Integration Test using '{config_path}'...")
#     print("="*80)

#     try:
#         configs = OmegaConf.load(config_path)
#         # 确保日志目录存在
#         Path(configs.train.save_dir).mkdir(parents=True, exist_ok=True)
#     except FileNotFoundError:
#         print(f"❌ CRITICAL: Configuration file '{config_path}' not found.")
#         return

#     # 依次执行所有检查，任何一步失败则终止
#     if not check_data_loading(configs):
#         sys.exit(1)
        
#     if not check_model_forward_pass(configs):
#         sys.exit(1)
        
#     if not check_training_step(configs):
#         sys.exit(1)
        
#     if not run_short_training_and_validation(configs):
#         sys.exit(1)

#     print("\n" + "="*80)
#     print("🎉🎉🎉 ALL INTEGRATION TESTS PASSED! 🎉🎉🎉")
#     print("Your full pipeline is robust and ready for large-scale training.")
#     print("="*80 + "\n")

# if __name__ == '__main__':
#     main()
# 文件路径: run_integration_test.py
# (一个全面的、用于调试和验证全流程的联调测试脚本)

# import os, sys, yaml, torch
# from pathlib import Path
# from omegaconf import OmegaConf
# import matplotlib.pyplot as plt
# import numpy as np

# # --- 环境设置 ---
# project_root = Path(__file__).parent.resolve()
# sys.path.append(str(project_root))
# from trainer import TrainerAlphaSR

# def run_integration_test(config_path='configs/config_integration_test.yaml'):
#     """
#     使用指定的配置文件运行一个极短的、端到端的训练-验证流程。
#     """
#     print("="*80)
#     print(f"🚀 Starting Full Integration Test using '{config_path}'...")
#     print("="*80)

#     try:
#         configs = OmegaConf.load(config_path)
#         # 确保日志目录存在
#         Path(configs.train.save_dir).mkdir(parents=True, exist_ok=True)
#     except FileNotFoundError:
#         print(f"❌ CRITICAL: Configuration file '{config_path}' not found.")
#         return

#     try:
#         # 实例化并运行Trainer
#         trainer = TrainerAlphaSR(configs)
#         print("\n--- [Check 1/2] Verifying a single training step ---")
#         # 手动执行一步来快速捕获错误
#         data = next(iter(trainer.dataloaders['train']))
#         data = trainer.prepare_data(data)
#         trainer.training_step(data)
#         print("✅ Single training step successful.")

#         print("\n--- [Check 2/2] Running short training & validation loop ---")
#         trainer.train() # Trainer内部会根据iterations=100来运行
        
#         print("\n✅ Short training and validation loop completed.")
#         print(f"   - Check the logs and images in: '{configs.train.save_dir}'")
#         print("   - Look for 'training.log' for loss values.")
#         print("   - Look for images in the 'images/val' subdirectory to visually inspect results.")

#         print("\n" + "="*80)
#         print("🎉🎉🎉 ALL INTEGRATION TESTS PASSED! 🎉🎉🎉")
#         print("Your full pipeline is robust and ready for large-scale training.")
#         print("="*80 + "\n")

#     except Exception as e:
#         print(f"❌ FAILED: An error occurred during the integration test. Error: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

# if __name__ == '__main__':
#     # 确保您已经创建了 config_integration_test.yaml 文件
#     run_integration_test()
