# 文件路径: trainer_srcnn.py
# SRCNN专用训练器：修复SRCNN简单架构的特殊需求
# - 正确处理SRCNN(tensor) vs SwinIR(dict)的输入差异
# - validation/baseline_visualize中正确调用模型
# - 简化可视化流程（SRCNN无时间维度复杂性）

import os
import sys
import math
import time
import random
import datetime
import shutil
import numpy as np
import torch
from contextlib import nullcontext
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sewar.full_ref import psnr, ssim, ergas, sam
from tqdm import tqdm
import matplotlib.pyplot as plt

# 项目内模块导入
from datapipe.datasets import create_dataset
from utils import util_net, util_common

# ==============================================================================
# 【核心组件】可配置的混合损失类（同trainer.py）
# ==============================================================================
class MixedLoss(torch.nn.Module):
    """
    结合L2(MSE)和SSIM的混合损失函数，权重/数据范围可通过配置文件灵活设置
    """
    def __init__(self, l2_weight=1.0, ssim_weight=1.0, data_range=2.0):
        super().__init__()
        self.l2_weight = l2_weight
        self.ssim_weight = ssim_weight
        self.data_range = data_range
        self.mse = torch.nn.MSELoss()
        # 启动日志，方便调试配置
        if torch.cuda.current_device() == 0:
            print(f"🔥 Initialized MixedLoss | L2_weight={self.l2_weight}, SSIM_weight={self.ssim_weight}, DataRange={self.data_range}")

    def forward(self, prediction, target):
        from pytorch_msssim import ssim as ssim_loss_func
        
        # 计算L2(MSE)损失（可选权重）
        l2_loss = self.mse(prediction, target) if self.l2_weight > 0 else 0.0
        
        # 计算SSIM损失（1 - SSIM值，可选权重）
        if self.ssim_weight > 0:
            ssim_val = ssim_loss_func(prediction, target, data_range=self.data_range, size_average=True)
            ssim_loss = 1.0 - ssim_val
        else:
            ssim_loss = 0.0
        
        # 加权求和得到总损失
        total_loss = (self.l2_weight * l2_loss) + (self.ssim_weight * ssim_loss)
        return total_loss

# ==============================================================================
# 1. SRCNN专用Trainer基类
# ==============================================================================
class TrainerSRCNNBase:
    """SRCNN简化架构的训练器基类"""
    def __init__(self, configs):
        self.configs = configs
        self.init_dist_and_seed()    # 初始化分布式训练和随机种子
        self.init_logger()           # 初始化日志系统
        self.build_dataloader()      # 先构建数据加载器（用于动态检测通道数）
        self.build_model()           # 动态构建模型（适配数据通道数）
        self.setup_optimization()    # 设置优化器/损失函数
        self.resume_from_ckpt()      # 恢复训练（如有检查点）
        self.current_iters = int(getattr(self, 'iters_start', 0))

    def init_dist_and_seed(self):
        """初始化分布式训练环境和随机种子"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = bool(self.configs.train.get('cudnn_benchmark', True))
            allow_tf32 = bool(self.configs.train.get('allow_tf32', True))
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            matmul_precision = str(self.configs.train.get('matmul_precision', 'high'))
            if hasattr(torch, "set_float32_matmul_precision"):
                try:
                    torch.set_float32_matmul_precision(matmul_precision)
                except Exception:
                    pass

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # 分布式训练初始化
            dist.init_process_group(backend='nccl', rank=0, world_size=num_gpus)
            self.rank = 0
            self.num_gpus = num_gpus
        else:
            # 单卡模式
            self.rank = 0
            self.num_gpus = 1 if torch.cuda.is_available() else 0

        # 设置随机种子（保证可重现性）
        if self.configs.train.get('global_seeding', True):
            seed = self.configs.train.get('seed', 42)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if self.rank == 0:
                print(f"🌱 Global seeding with seed={seed}")

    def init_logger(self):
        """初始化日志保存和控制台输出"""
        if hasattr(self.configs, 'resume') and self.configs.resume:
            self.save_dir = Path(self.configs.resume).parents[1]
        else:
            run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = Path(self.configs.train.save_dir) / run_name

        # 仅主进程创建目录和日志
        if self.rank == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.logger = logger
            self.logger.remove()
            
            log_path = self.save_dir / 'training.log'
            self.logger.add(log_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")
            self.logger.info(OmegaConf.to_yaml(self.configs))

        # 检查点目录
        self.ckpt_dir = self.save_dir / 'ckpts'
        if self.rank == 0:
            self.ckpt_dir.mkdir(exist_ok=True)

        # 可视化图片保存目录（仅主进程）
        if self.rank == 0 and self.configs.train.get('local_logging', False):
            self.image_dir = self.save_dir / 'images'
            (self.image_dir / 'val').mkdir(parents=True, exist_ok=True)

    def build_dataloader(self):
        """构建训练/验证数据加载器"""
        def _wrap(loader):
            while True:
                yield from loader

        # 创建数据集
        datasets = {'train': create_dataset(self.configs.data.train, parent_configs=self.configs)}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.val, parent_configs=self.configs)

        # 打印数据集大小
        if self.rank == 0:
            for phase, ds in datasets.items():
                self.logger.info(f'📊 Dataset [{phase}] size: {len(ds)}')

        # 分布式采样器
        sampler = None
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                datasets['train'],
                num_replicas=self.num_gpus,
                rank=self.rank
            )

        # 批次大小
        train_batch_size = self.configs.train.batch[0] // self.num_gpus if self.num_gpus > 0 else self.configs.train.batch[0]
        val_batch_size = self.configs.train.batch[1] if len(self.configs.train.batch) > 1 else 1
        train_num_workers = int(self.configs.train.get('num_workers', 4))
        pin_memory = bool(self.configs.train.get('pin_memory', True))
        persistent_workers = bool(self.configs.train.get('persistent_workers', train_num_workers > 0))
        prefetch_factor = int(self.configs.train.get('prefetch_factor', 4))

        # 构建数据加载器
        train_loader_kwargs = dict(
            batch_size=train_batch_size,
            shuffle=(sampler is None),
            drop_last=True,
            num_workers=train_num_workers,
            sampler=sampler,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and train_num_workers > 0),
        )
        if train_num_workers > 0:
            train_loader_kwargs['prefetch_factor'] = prefetch_factor

        self.dataloaders = {
            'train': _wrap(udata.DataLoader(
                datasets['train'],
                **train_loader_kwargs
            ))
        }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            val_num_workers = int(self.configs.train.get('val_num_workers', min(8, train_num_workers)))
            val_loader_kwargs = dict(
                batch_size=val_batch_size,
                num_workers=val_num_workers,
                pin_memory=pin_memory,
                persistent_workers=(persistent_workers and val_num_workers > 0),
            )
            if val_num_workers > 0:
                val_loader_kwargs['prefetch_factor'] = prefetch_factor
            self.dataloaders['val'] = udata.DataLoader(
                datasets['val'],
                **val_loader_kwargs
            )

        self.datasets = datasets
        self.sampler = sampler

    def build_model(self):
        """动态构建SRCNN模型"""
        if 'train' in self.datasets:
            try:
                # 临时加载一个样本探测输入通道数
                temp_loader = udata.DataLoader(self.datasets['train'], batch_size=1, shuffle=False)
                first_batch = next(iter(temp_loader))
                sample_x = self._resolve_input_tensor(first_batch)
                sample_y = self._resolve_target_tensor(first_batch)
                C = int(sample_x.shape[1])
                out_C = int(sample_y.shape[1])
                self.configs.model.params.in_channels = C
                self.configs.model.params.out_channels = out_C
                
                if self.rank == 0:
                    self.logger.info(f"✅ Dynamically detected in_channels for SRCNN: {C}")
                    self.logger.info(f"✅ Dynamically detected out_channels for SRCNN: {out_C}")
            except Exception as e:
                if self.rank == 0:
                    self.logger.error(f"❌ Error detecting channels: {e}")
                raise

        # 实例化模型
        try:
            self.model = util_common.get_obj_from_str(self.configs.model.target)(**self.configs.model.params)
            self.model = self.model.cuda() if torch.cuda.is_available() else self.model
            
            # DDP包装（多GPU）
            if self.num_gpus > 1:
                self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
            
            if self.rank == 0:
                self.logger.info(f"✅ Model initialized: {self.configs.model.target}")
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                self.logger.info(f"📊 Total trainable parameters: {total_params:,}")
        except Exception as e:
            if self.rank == 0:
                self.logger.error(f"❌ Error building model: {e}")
            raise

    def setup_optimization(self):
        """设置优化器和损失函数"""
        # 优化器
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.configs.train.lr, weight_decay=self.configs.train.weight_decay)
        self.channels_last = bool(self.configs.train.get('channels_last', False))
        self.amp_enabled = bool(self.configs.train.get('amp', True)) and torch.cuda.is_available()
        amp_dtype = str(self.configs.train.get('amp_dtype', 'bf16')).lower()
        self.amp_dtype = torch.float16 if amp_dtype in ('fp16', 'float16', 'half') else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_enabled and self.amp_dtype == torch.float16))
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            **self.configs.train.lr_schedule_params
        )

        # 损失函数
        if hasattr(self.configs, 'loss'):
            self.criterion = util_common.instantiate_from_config(self.configs.loss)
            if self.rank == 0:
                self.logger.info("📝 Using configured loss function")
        else:
            self.criterion = F.l1_loss
            if self.rank == 0:
                self.logger.warning("⚠️ Using default L1 loss")

    def resume_from_ckpt(self):
        """恢复检查点"""
        self.best_metric = 0.0
        self.iters_start = 0
        
        if hasattr(self.configs, 'resume_path') and self.configs.resume_path:
            ckpt_path = self.configs.resume_path
        else:
            ckpt_path = None

        if ckpt_path and Path(ckpt_path).exists():
            if self.rank == 0:
                self.logger.info(f"📂 Resuming from checkpoint: {ckpt_path}")
            
            ckpt = torch.load(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            
            # 加载模型权重
            state_dict = ckpt.get('state_dict', ckpt)
            unwrapped_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                unwrapped_state_dict[name] = v
            
            self.model.load_state_dict(unwrapped_state_dict, strict=True)
            
            # 恢复优化器/迭代数/最佳指标
            if 'optimizer' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            self.iters_start = ckpt.get('iters_start', 0)
            self.best_metric = ckpt.get('best_metric', 0.0)
            
            if self.rank == 0:
                self.logger.info(f"✅ Resumed from iteration {self.iters_start}")

    def prepare_data(self, data):
        """将数据移到GPU"""
        if isinstance(data, dict):
            return {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        return data.cuda(non_blocking=True)

    def save_ckpt(self, best=False):
        """保存检查点"""
        if self.rank == 0:
            filename = 'model_best.pth' if best else f'model_{self.current_iters}.pth'
            ckpt_path = self.ckpt_dir / filename
            
            # 获取模型状态字典
            model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
            
            torch.save({
                'iters_start': self.current_iters,
                'state_dict': model_state,
                'best_metric': self.best_metric,
                'optimizer': self.optimizer.state_dict()
            }, ckpt_path)

            if not best:
                self.logger.info(f"💾 Saved checkpoint to: {ckpt_path}")

    @staticmethod
    def norm_for_vis(img_tensor):
        """可视化专用归一化：将[-1,1]映射到[0,1]"""
        img_clamped = img_tensor.detach().clamp(-1, 1)
        img_01 = (img_clamped + 1) / 2.0
        return img_01.cpu().numpy()

    @staticmethod
    def _resolve_input_tensor(data_dict):
        """兼容不同数据集键名，返回SRCNN输入张量(B,C,H,W)。"""
        for key in ('s1', 'lr', 'lr_sequence'):
            if key in data_dict and isinstance(data_dict[key], torch.Tensor):
                x = data_dict[key]
                if x.dim() == 5:
                    x = x[:, -1, ...]
                if x.dim() != 4:
                    raise ValueError(f"Unsupported SRCNN input shape for key '{key}': {tuple(x.shape)}")
                # SRCNN baseline only consumes optical channels.
                # If LR includes an extra cloud-mask band (e.g., C=10), drop it.
                if x.shape[1] > 9:
                    x = x[:, :9, :, :]
                return x
        raise KeyError(f"No SRCNN input key found. Available keys: {list(data_dict.keys())}")

    @staticmethod
    def _resolve_target_tensor(data_dict):
        """兼容不同数据集键名，返回监督目标张量(B,C,H,W)。"""
        for key in ('gt', 'hr', 'target'):
            if key in data_dict and isinstance(data_dict[key], torch.Tensor):
                y = data_dict[key]
                if y.dim() == 5:
                    y = y[:, -1, ...]
                if y.dim() != 4:
                    raise ValueError(f"Unsupported target shape for key '{key}': {tuple(y.shape)}")
                return y
        raise KeyError(f"No target key found. Available keys: {list(data_dict.keys())}")

    def training_step(self, data):
        raise NotImplementedError("Subclass must implement training_step!")

    def validation(self, phase='val'):
        raise NotImplementedError("Subclass must implement validation!")

    def adjust_lr(self, metrics):
        raise NotImplementedError("Subclass must implement adjust_lr!")

    def baseline_visualize(self):
        pass

    def plot_curves(self):
        pass


# ==============================================================================
# 2. SRCNN专用Trainer（完整实现）
# ==============================================================================
class TrainerSRCNN(TrainerSRCNNBase):
    """SRCNN专用训练器：简化版本无时间维度复杂性"""
    def __init__(self, configs):
        super().__init__(configs)
        if self.rank == 0:
            self.log_data = {
                'train_loss': {'iters': [], 'values': []},
                'val_psnr': {'iters': [], 'values': []},
                'val_ssim': {'iters': [], 'values': []}
            }

    def training_step(self, data):
        """【关键】SRCNN训练步骤：仅传lr_sequence张量给模型"""
        srcnn_input = self._resolve_input_tensor(data)
        target = self._resolve_target_tensor(data)
        if self.channels_last and isinstance(srcnn_input, torch.Tensor) and srcnn_input.ndim == 4:
            srcnn_input = srcnn_input.contiguous(memory_format=torch.channels_last)

        amp_ctx = torch.cuda.amp.autocast(dtype=self.amp_dtype) if self.amp_enabled else nullcontext()
        with amp_ctx:
            predictions = self.model(srcnn_input)
            # 计算损失
            loss = self.criterion(predictions, target)

        # 反向传播
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        # 记录损失
        self.log_step_train(loss)

    def log_step_train(self, loss):
        """记录训练损失"""
        if self.rank == 0:
            self.log_data['train_loss']['iters'].append(self.current_iters)
            self.log_data['train_loss']['values'].append(loss.item())
            if self.current_iters % 100 == 0:
                self.logger.info(f"📈 Iter {self.current_iters} | Train Loss: {loss.item():.6f}")

    @torch.no_grad()
    def validation(self, phase='val'):
        """【关键】SRCNN验证步骤：仅传lr_sequence张量给模型"""
        if self.rank == 0:
            self.model.eval()
            all_metrics = {'psnr': [], 'ssim': [], 'ergas': [], 'sam': []}
            pbar = tqdm(self.dataloaders[phase], desc=f"📌 Val Iter {self.current_iters}")
            
            for ii, data in enumerate(pbar):
                data = self.prepare_data(data)
                srcnn_input = self._resolve_input_tensor(data)
                target = self._resolve_target_tensor(data)
                
                # 【重要】SRCNN只接受单张量输入
                predictions = self.model(srcnn_input)
                
                # 归一化到[0,1]用于指标计算
                gt_01 = self.norm_for_vis(target)
                pred_01 = self.norm_for_vis(predictions)
                
                # 转换为(H,W,C)格式
                gt_numpy = gt_01.transpose(0, 2, 3, 1)[0]
                pred_numpy = pred_01.transpose(0, 2, 3, 1)[0]
                
                # 计算多波段平均指标
                psnr_val = np.mean([psnr(gt_numpy[:, :, b], pred_numpy[:, :, b], MAX=1.0) for b in range(gt_numpy.shape[-1])])
                ssim_val = np.mean([ssim(gt_numpy[:, :, b], pred_numpy[:, :, b], MAX=1.0)[0] for b in range(gt_numpy.shape[-1])])
                
                all_metrics['psnr'].append(psnr_val)
                all_metrics['ssim'].append(ssim_val)
                all_metrics['ergas'].append(ergas(gt_numpy, pred_numpy))
                all_metrics['sam'].append(sam(gt_numpy, pred_numpy))
                
                # 可视化第一个样本
                if ii == 0 and self.configs.train.get('local_logging', False):
                    self.visualize_validation_sample(data, predictions, ii)
            
            # 计算平均指标
            avg_psnr = float(np.mean(all_metrics['psnr']))
            avg_ssim = float(np.mean(all_metrics['ssim']))
            avg_ergas = float(np.mean(all_metrics['ergas']))
            avg_sam = float(np.mean(all_metrics['sam']))
            
            self.logger.info(
                f"📊 Validation Metrics | "
                f"PSNR: {avg_psnr:.4f} | "
                f"SSIM: {avg_ssim:.4f} | "
                f"ERGAS: {avg_ergas:.4f} | "
                f"SAM: {avg_sam:.4f}"
            )
            
            # 记录指标
            self.log_data['val_psnr']['iters'].append(self.current_iters)
            self.log_data['val_psnr']['values'].append(avg_psnr)
            self.log_data['val_ssim']['iters'].append(self.current_iters)
            self.log_data['val_ssim']['values'].append(avg_ssim)
            
            self.model.train()
            return avg_psnr

    def visualize_validation_sample(self, data, predictions, sample_idx):
        """可视化验证样本（简化版，无时间维度）"""
        try:
            srcnn_input = self._resolve_input_tensor(data)
            target = self._resolve_target_tensor(data)

            lr_vis_tensor = srcnn_input[sample_idx]
            gt_vis_tensor = target[sample_idx]
            pred_vis_tensor = predictions[sample_idx]

            lr_vis = self.norm_for_vis(lr_vis_tensor.unsqueeze(0))[0]
            gt_vis = self.norm_for_vis(gt_vis_tensor.unsqueeze(0))[0]
            pred_vis = self.norm_for_vis(pred_vis_tensor.unsqueeze(0))[0]

            error_map = np.abs(pred_vis - gt_vis).mean(axis=2)

            fig, axes = plt.subplots(2, 2, figsize=(14, 14))
            fig.suptitle(f'Validation Iter {self.current_iters}', fontsize=16)

            rgb_chn = self.configs.train.get('rgb_chn', [0, 1, 2])

            axes[0, 0].imshow(lr_vis.transpose(1, 2, 0)[:, :, rgb_chn]); axes[0, 0].set_title('Input LR')
            axes[0, 1].imshow(pred_vis.transpose(1, 2, 0)[:, :, rgb_chn]); axes[0, 1].set_title('Prediction')
            axes[1, 0].imshow(gt_vis.transpose(1, 2, 0)[:, :, rgb_chn]); axes[1, 0].set_title('Ground Truth')
            im = axes[1, 1].imshow(error_map, cmap='hot'); axes[1, 1].set_title('Error Map')
            fig.colorbar(im, ax=axes[1, 1])

            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = self.image_dir / 'val' / f"iter_{self.current_iters}_sample_{sample_idx}.png"
            plt.savefig(str(save_path), dpi=150)
            plt.close(fig)

        except Exception as e:
            self.logger.warning(f"⚠️ Visualization failed: {e}")

    def baseline_visualize(self):
        """训练前生成基线可视化"""
        if self.rank == 0 and self.configs.train.get('local_logging', False):
            self.logger.info("🎨 Generating baseline visualization...")
            self.model.eval()
            data = next(iter(self.dataloaders['val']))
            data = self.prepare_data(data)
            srcnn_input = self._resolve_input_tensor(data)
            
            # 【重要】SRCNN只接受单张量输入
            predictions = self.model(srcnn_input)
            
            self.visualize_validation_sample(data, predictions, 0)
            self.logger.info(f"✅ Baseline visualization saved")
            self.model.train()

    def adjust_lr(self, metrics):
        """根据验证指标调整学习率"""
        self.scheduler.step(metrics)

    def plot_curves(self):
        """生成训练/验证曲线"""
        if self.rank == 0 and hasattr(self, 'log_data'):
            self.logger.info("📊 Generating training curves...")
            fig, ax1 = plt.subplots(figsize=(12, 7))

            color = 'tab:red'
            ax1.set_xlabel('Iterations')
            ax1.set_ylabel('Training Loss (Log Scale)', color=color)
            ax1.plot(
                self.log_data['train_loss']['iters'],
                self.log_data['train_loss']['values'],
                color=color, alpha=0.7, linewidth=2, label='Loss'
            )
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, linestyle=':', alpha=0.7)
            ax1.set_yscale('log')

            if self.log_data['val_psnr']['iters']:
                ax2 = ax1.twinx()
                ax2.set_ylabel('Validation PSNR / SSIM', color='tab:blue')
                ax2.plot(
                    self.log_data['val_psnr']['iters'],
                    self.log_data['val_psnr']['values'],
                    color='tab:blue', marker='o', linestyle='-', markersize=5, label='PSNR'
                )
                ax2.plot(
                    self.log_data['val_ssim']['iters'],
                    self.log_data['val_ssim']['values'],
                    color='tab:green', marker='x', linestyle='--', label='SSIM'
                )
                ax2.tick_params(axis='y', labelcolor='tab:blue')

            fig.suptitle('Training & Validation Curves', fontsize=16)
            fig.tight_layout()
            save_path = self.save_dir / "training_curves.png"
            plt.savefig(str(save_path), dpi=150)
            plt.close(fig)
            self.logger.info(f"✅ Curves saved")

    def train(self):
        """完整训练循环（兼容分布式）"""
        start_time = time.time()
        total_iters = self.configs.train.iterations
        log_freq = self.configs.train.log_freq[0]
        save_freq = self.configs.train.save_freq
        val_freq = self.configs.train.val_freq
        progress_log_freq = int(self.configs.train.get('progress_log_freq', log_freq if log_freq > 0 else 20))

        if self.rank == 0:
            self.logger.info(
                f"🚀 Start training | start_iter={self.iters_start} | total_iters={total_iters} | "
                f"val_freq={val_freq} | save_freq={save_freq}"
            )
            self.logger.info("🖼️ Baseline visualization: begin")
            self.baseline_visualize()  # 训练前可视化
            self.logger.info("🖼️ Baseline visualization: done")

        self.current_iters = self.iters_start
        self.model.train()

        train_loader = self.dataloaders['train']
        pbar = None
        if self.rank == 0:
            pbar = tqdm(
                total=max(total_iters - self.iters_start, 0),
                initial=0,
                desc="🚂 Training(SRCNN)",
                dynamic_ncols=True,
            )
        
        while self.current_iters < total_iters:
            try:
                if self.rank == 0 and self.current_iters == self.iters_start:
                    self.logger.info("📦 Fetching first training batch...")
                data = next(train_loader)
                data = self.prepare_data(data)
                
                # 训练步骤
                self.training_step(data)
                self.current_iters += 1
                if pbar is not None:
                    lr_val = float(self.optimizer.param_groups[0]['lr'])
                    pbar.update(1)
                    pbar.set_postfix(iter=self.current_iters, lr=f"{lr_val:.2e}")
                if self.rank == 0 and self.current_iters == self.iters_start + 1:
                    self.logger.info("✅ First training step completed")

                if self.rank == 0 and (
                    self.current_iters % progress_log_freq == 0 or self.current_iters == total_iters
                ):
                    elapsed = max(time.time() - start_time, 1e-6)
                    iter_per_sec = self.current_iters / elapsed
                    remaining = max(total_iters - self.current_iters, 0)
                    eta_sec = remaining / max(iter_per_sec, 1e-6)
                    lr_val = float(self.optimizer.param_groups[0]['lr'])
                    self.logger.info(
                        f"⏱️ Progress {self.current_iters}/{total_iters} | "
                        f"iter/s={iter_per_sec:.2f} | eta={eta_sec/60:.1f} min | lr={lr_val:.2e}"
                    )
                
                # 验证
                if self.current_iters % val_freq == 0 and self.rank == 0:
                    self.logger.info(f"🧪 Validation start @ iter {self.current_iters}")
                    avg_psnr = self.validation()
                    self.adjust_lr(avg_psnr)
                    self.logger.info(f"🧪 Validation done @ iter {self.current_iters} | psnr={avg_psnr:.4f}")
                    if avg_psnr > self.best_metric:
                        self.best_metric = avg_psnr
                        self.logger.info(f"🏆 New best metric: {self.best_metric:.4f} | Saving best model...")
                        self.save_ckpt(best=True)
                    self.model.train()
                
                # 保存检查点
                if self.current_iters % save_freq == 0:
                    if self.rank == 0:
                        self.logger.info(f"💾 Regular checkpoint save @ iter {self.current_iters}")
                    self.save_ckpt()
                    
            except Exception as e:
                if self.rank == 0:
                    self.logger.error(f"❌ Error in training loop: {e}")
                    import traceback; traceback.print_exc()
                break

        # 训练完成
        if self.rank == 0:
            if pbar is not None:
                pbar.close()
            self.plot_curves()
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ Training completed in {elapsed_time/3600:.2f} hours")


# ==============================================================================
# 3. 入口函数
# ==============================================================================
if __name__ == '__main__':
    # 直接运行测试
    configs = OmegaConf.load('configs/config_srcnn.yaml')
    trainer = TrainerSRCNN(configs)
    trainer.train()
