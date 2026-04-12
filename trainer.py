# 文件路径: trainer.py
# 最终整合版：动态通道检测 + 可配置混合损失 + 健壮可视化 + 完整训练流程
import os
import sys
import math
import time
import random
import datetime
import shutil
import numpy as np
import torch
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
# import hydra  # 用于自动化实例化（保留，兼容配置实例化逻辑）
from pytorch_msssim import ssim as ssim_loss_func  # SSIM损失计算

# 项目内模块导入
from datapipe.datasets import create_dataset
from utils import util_net, util_common

# ==============================================================================
# 【核心组件】可配置的混合损失类（完整实现）
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
# 1. Trainer 基类（完整实现，包含分布式/数据加载/模型构建核心逻辑）
# ==============================================================================
class TrainerBase:
    def __init__(self, configs):
        self.configs = configs
        self.init_dist_and_seed()    # 初始化分布式训练和随机种子
        self.init_logger()           # 初始化日志系统
        self.build_dataloader()      # 先构建数据加载器（用于动态检测通道数）
        self.build_model()           # 动态构建模型（适配数据通道数）
        self.setup_optimization()    # 设置优化器/损失函数
        self.resume_from_ckpt()      # 恢复训练（如有检查点）
        # 训练循环开始前也需要可用的 current_iters（例如 baseline 可视化）
        self.current_iters = int(getattr(self, 'iters_start', 0))

    def init_dist_and_seed(self):
        """初始化分布式训练环境和随机种子"""
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # 分布式训练初始化
            rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                timeout=datetime.timedelta(seconds=3600),
                backend='nccl',
                init_method='env://'
            )
        self.num_gpus = num_gpus
        self.rank = int(os.environ.get('LOCAL_RANK', 0)) if num_gpus > 1 else 0

        # 设置随机种子（分布式下每个进程种子偏移）
        seed = self.configs.train.get('seed', 42)
        if self.num_gpus > 1 and not self.configs.train.get('global_seeding', True):
            seed += self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_logger(self):
        """初始化日志保存和控制台输出"""
        # 确定保存目录（恢复训练时用原有目录，新训练时创建时间戳目录）
        if hasattr(self.configs, 'resume') and self.configs.resume:
            self.save_dir = Path(self.configs.resume).parents[1]
        else:
            run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = Path(self.configs.train.save_dir) / run_name

        # 仅主进程创建目录和日志
        if self.rank == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.logger = logger
            self.logger.remove()  # 清除默认日志器
            # 日志保存到文件
            log_path = self.save_dir / 'training.log'
            self.logger.add(log_path, format="{message}", mode='a', level='INFO')
            # 日志输出到控制台
            self.logger.add(sys.stdout, format="{message}")
            # 打印配置文件（方便追溯）
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
        """构建训练/验证数据加载器（兼容分布式采样）"""
        def _wrap(loader):
            """无限迭代的数据加载器包装器"""
            while True:
                yield from loader

        # 创建数据集
        datasets = {'train': create_dataset(self.configs.data.train, parent_configs=self.configs)}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.val, parent_configs=self.configs)

        # 打印数据集大小（仅主进程）
        if self.rank == 0:
            for phase, ds in datasets.items():
                self.logger.info(f'📊 Dataset [{phase}] size: {len(ds)}')

        # 分布式采样器（多GPU时用）
        sampler = None
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                datasets['train'],
                num_replicas=self.num_gpus,
                rank=self.rank
            )

        # 批次大小（多GPU时均分训练批次）
        train_batch_size = self.configs.train.batch[0] // self.num_gpus if self.num_gpus > 0 else self.configs.train.batch[0]
        val_batch_size = self.configs.train.batch[1] if len(self.configs.train.batch) > 1 else 1

        # 构建数据加载器
        self.dataloaders = {
            'train': _wrap(udata.DataLoader(
                datasets['train'],
                batch_size=train_batch_size,
                shuffle=(sampler is None),
                drop_last=True,
                num_workers=self.configs.train.num_workers,
                sampler=sampler
            ))
        }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            self.dataloaders['val'] = udata.DataLoader(
                datasets['val'],
                batch_size=val_batch_size,
                num_workers=0  # 验证集用0个worker避免数据错乱
            )

        self.datasets = datasets
        self.sampler = sampler

    def build_model(self):
        """动态构建模型（从数据中检测输入通道数）"""
        if 'train' in self.datasets:
            try:
                # 临时加载一个样本探测输入通道数
                temp_loader = udata.DataLoader(self.datasets['train'], batch_size=1, shuffle=False)
                first_batch = next(iter(temp_loader))
                _,_, C, _, _ = first_batch['lr_sequence'].shape  # 从lr_sequence获取通道数

                # 根据模型类型动态注入通道数
                model_name = self.configs.model.target.split('.')[-1].lower()
                if 'srcnn' in model_name:
                    self.configs.model.params.in_channels = C
                    if 'in_chans' in self.configs.model.params:
                        del self.configs.model.params['in_chans']
                    if self.rank == 0:
                        self.logger.info(f"✅ Dynamically detected in_channels for SRCNN: {C}")
                else:
                    self.configs.model.params.in_chans = C
                    if 'in_channels' in self.configs.model.params:
                        del self.configs.model.params['in_channels']
                    if self.rank == 0:
                        self.logger.info(f"✅ Dynamically detected in_chans: {C}")

                # 释放临时变量
                del temp_loader, first_batch
            except Exception as e:
                if self.rank == 0:
                    self.logger.error(f"❌ Failed to detect input channels: {e}")
                # 判断两种情况都没有时才报错
                model_name = self.configs.model.target.split('.')[-1].lower()
                if (('srcnn' in model_name and 'in_channels' not in self.configs.model.params) or
                    ('srcnn' not in model_name and 'in_chans' not in self.configs.model.params)):
                    sys.exit("CRITICAL: input channel detection failed and required parameter missing!")

        # 实例化模型并移到GPU
        self.model = util_common.instantiate_from_config(self.configs.model).cuda()
        # 多GPU时用DDP包装
        if self.num_gpus > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                find_unused_parameters=True
            )

    def setup_optimization(self):
        """设置优化器（子类重写时补充损失函数/调度器）"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.configs.train.lr,
            weight_decay=self.configs.train.get('weight_decay', 0)
        )

    def prepare_data(self, data):
        """将数据移到GPU（支持字典/张量类型）"""
        if isinstance(data, dict):
            return {k: v.cuda(non_blocking=True) for k, v in data.items() if isinstance(v, torch.Tensor)}
        return data.cuda(non_blocking=True)

    def save_ckpt(self, best=False):
        """保存检查点（仅主进程）"""
        if self.rank == 0:
            # 文件名（最佳模型/普通检查点）
            filename = 'model_best.pth' if best else f'model_{self.current_iters}.pth'
            ckpt_path = self.ckpt_dir / filename
            
            # 获取模型状态字典（兼容DDP）
            model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
            
            # 保存检查点（包含迭代数/模型/优化器/最佳指标）
            torch.save({
                'iters_start': self.current_iters,
                'state_dict': model_state,
                'best_metric': self.best_metric,
                'optimizer': self.optimizer.state_dict()
            }, ckpt_path)

            # 可选：将最佳权重镜像到 ./best_ckpts/{exp_name}/model.pth
            if best and self.configs.train.get('export_best_ckpt', True):
                self.export_best_ckpt(ckpt_path)
            
            if not best:
                self.logger.info(f"💾 Saved checkpoint to: {ckpt_path}")

    def export_best_ckpt(self, ckpt_path: Path):
        """将最佳权重导出到统一目录，便于分析脚本和文档引用。"""
        try:
            save_dir = Path(str(self.configs.train.save_dir))
            exp_name = save_dir.name if save_dir.name else "default_exp"
            best_root = Path(self.configs.train.get('best_ckpt_dir', './best_ckpts'))
            target_dir = best_root / exp_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / 'model.pth'

            # 采用复制以保证跨文件系统稳定性
            shutil.copy2(ckpt_path, target_path)
            self.logger.info(f"🏁 Exported best checkpoint to: {target_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to export best checkpoint: {e}")

    def resume_from_ckpt(self):
        """从检查点恢复训练"""
        self.iters_start = 0
        self.best_metric = 0.0
        
        if hasattr(self.configs, 'resume') and self.configs.resume:
            ckpt_path = self.configs.resume
            assert Path(ckpt_path).exists(), f"Checkpoint not found: {ckpt_path}"
            
            if self.rank == 0:
                self.logger.info(f"🔄 Resuming from checkpoint: {ckpt_path}")
            
            # 兼容 PyTorch 2.6: 默认 weights_only=True 可能导致旧 checkpoint 反序列化失败。
            # 对可信本地实验权重，失败后自动回退到 weights_only=False。
            try:
                ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            except Exception as e:
                err = str(e)
                if "Weights only load failed" in err:
                    if self.rank == 0:
                        self.logger.warning(
                            "⚠️ torch.load safe mode failed; retrying with weights_only=False for trusted local checkpoint."
                        )
                    try:
                        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}", weights_only=False)
                    except TypeError:
                        # 兼容旧版 PyTorch（无 weights_only 参数）
                        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
                else:
                    raise
            
            # 加载模型权重（兼容DDP）
            model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
            try:
                load_info = util_net.reload_model(model_to_load, ckpt['state_dict'], strict=True)
            except AssertionError as e:
                if self.rank == 0:
                    self.logger.warning(f"⚠️ Strict weight load failed: {e}")
                    self.logger.warning("⚠️ Falling back to non-strict load (matched keys only).")
                load_info = util_net.reload_model(model_to_load, ckpt['state_dict'], strict=False)

            if self.rank == 0 and isinstance(load_info, dict):
                self.logger.info(
                    f"🔎 Weight load summary | loaded={load_info.get('loaded', 0)} "
                    f"missing={len(load_info.get('missing', []))} "
                    f"shape_mismatch={len(load_info.get('shape_mismatch', []))}"
                )
            
            # 恢复迭代数和最佳指标
            self.iters_start = ckpt.get('iters_start', 0)
            self.best_metric = ckpt.get('best_metric', 0.0)
            
            # 恢复优化器。跨版本/跨结构恢复时，参数组数量可能不一致，失败则降级为新优化器状态继续训练。
            if 'optimizer' in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                except ValueError as e:
                    if self.rank == 0:
                        self.logger.warning(f"⚠️ Optimizer state load skipped due to mismatch: {e}")
                        self.logger.warning("⚠️ Continue with freshly initialized optimizer state.")
            
            if self.rank == 0:
                self.logger.info(f"✅ Resumed from iteration {self.iters_start} | Best metric: {self.best_metric:.4f}")

    def train(self):
        """主训练循环"""
        # 训练前生成基线可视化（仅主进程）
        if self.rank == 0 and self.iters_start == 0:
            self.baseline_visualize()
        
        self.model.train()
        # 迭代训练
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1
            
            # 分布式采样器更新epoch（保证多GPU数据不重复）
            if hasattr(self, 'sampler') and self.sampler is not None:
                self.sampler.set_epoch(ii)
            
            # 加载批次数据并执行训练步
            # import pdb;pdb.set_trace()
            data = self.prepare_data(next(self.dataloaders['train']))
            self.training_step(data)
            
            # 验证（按验证频率）
            if 'val' in self.dataloaders and (self.current_iters % self.configs.train.val_freq) == 0:
                cur_metric = self.validation()
                self.adjust_lr(cur_metric)  # 调整学习率
                # 保存最佳模型
                if self.rank == 0 and cur_metric > self.best_metric:
                    self.best_metric = cur_metric
                    self.logger.info(f"🏆 New best metric: {self.best_metric:.4f} | Saving best model...")
                    self.save_ckpt(best=True)
            
            # 保存普通检查点（按保存频率）
            if (self.current_iters % self.configs.train.save_freq) == 0:
                self.save_ckpt(best=False)
        
        # 训练结束生成曲线
        if self.rank == 0:
            self.plot_curves()
            self.logger.info("🎉 Training Finished Successfully!")

    @staticmethod
    def norm_for_vis(img_tensor):
        """可视化专用归一化：将[-1,1]映射到[0,1]"""
        img_clamped = img_tensor.clamp(-1, 1)
        img_01 = (img_clamped + 1) / 2.0
        return img_01.cpu().numpy()

    # 以下为需要子类实现/重写的方法
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
# 2. AlphaSR 专用 Trainer（完整实现训练/验证/可视化逻辑）
# ==============================================================================
class TrainerAlphaSR(TrainerBase):
    def __init__(self, configs):
        super().__init__(configs)
        # 初始化日志数据（记录损失/指标曲线）
        if self.rank == 0:
            self.log_data = {
                'train_loss': {'iters': [], 'values': []},
                'val_psnr': {'iters': [], 'values': []},
                'val_ssim': {'iters': [], 'values': []}
            }

    def setup_optimization(self):
        """扩展父类方法：添加学习率调度器和损失函数"""
        # 调用父类初始化优化器
        super().setup_optimization()
        
        # 学习率调度器（基于验证指标的ReduceLROnPlateau）
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            **self.configs.train.lr_schedule_params
        )

        # 实例化损失函数（优先从配置加载，否则用默认L1）
        if hasattr(self.configs, 'loss'):
            if self.rank == 0:
                self.logger.info("📝 Instantiating loss function from config")
            self.criterion = util_common.instantiate_from_config(self.configs.loss)
        else:
            if self.rank == 0:
                self.logger.warning("⚠️ Loss config not found | Falling back to L1 loss")
            self.criterion = F.l1_loss

    def adjust_lr(self, metrics):
        """根据验证指标调整学习率"""
        self.scheduler.step(metrics)

    def log_step_train(self, loss):
        """记录训练损失（仅主进程）"""
        if self.rank == 0:
            self.log_data['train_loss']['iters'].append(self.current_iters)
            self.log_data['train_loss']['values'].append(loss.item())
            # 每100迭代打印一次损失
            if self.current_iters % 100 == 0:
                self.logger.info(f"📈 Iter {self.current_iters} | Train Loss: {loss.item():.6f}")

    def training_step(self, data):
        """单步训练逻辑"""
        # 模型前向传播（输入为完整数据字典）
        """
        单步训练逻辑：自动区分 SRCNN（张量输入）和 SwinIR（字典输入）
        """
        # 判断模型类型，SRCNN 只接受张量输入
        model_name = self.model.__class__.__name__.lower()
        if 'srcnn' in model_name:
            # 仅取 lr_sequence 张量作为输入
            predictions = self.model(data['lr_sequence'])
        else:
            # 其他模型保持字典输入
            predictions = self.model(data)

        # 计算损失（可选：按 indicating_mask 对无云像素加权）
        use_ind_mask = self.configs.train.get('use_indicating_mask_in_training', False)
        if use_ind_mask and data.get('indicating_mask') is not None:
            # indicating_mask: (B,T,H,W) 或 (T,H,W) → 聚合到空间维 (B,1,H,W)
            ind_mask = data['indicating_mask'].float()
            if ind_mask.ndim == 3:
                ind_mask = ind_mask.unsqueeze(0)  # (1,T,H,W)
            # 时序聚合：任一时相有效则该像素有效
            spatial_mask = ind_mask.max(dim=1, keepdim=True)[0].clamp(0, 1)  # (B,1,H,W)
            spatial_mask = spatial_mask.expand_as(predictions)
            valid_pixels = spatial_mask.sum().clamp(min=1)
            loss = self.criterion(predictions * spatial_mask, data['gt'] * spatial_mask)
            loss = loss * (float(predictions.numel()) / valid_pixels)
        else:
            loss = self.criterion(predictions, data['gt'])
        
        # 反向传播+优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 记录损失
        self.log_step_train(loss)

    @torch.no_grad()
    def validation(self, phase='val'):
        """验证流程（计算PSNR/SSIM/ERGAS/SAM指标）"""
        if self.rank == 0:
            self.model.eval()
            all_metrics = {'psnr': [], 'ssim': [], 'ergas': [], 'sam': [], 'masked_psnr': []}
            pbar = tqdm(self.dataloaders[phase], desc=f"📌 Val Iter {getattr(self, 'current_iters', 0)}")
            
            for ii, data in enumerate(pbar):
                data = self.prepare_data(data)
                
                # 模型推理
                predictions = self.model(data)
                
                # 归一化到[0,1]用于指标计算
                gt_01 = self.norm_for_vis(data['gt'])
                pred_01 = self.norm_for_vis(predictions)
                
                # 转换为(H,W,C)格式（适配sewar库）
                gt_numpy = gt_01.transpose(0, 2, 3, 1)[0]
                pred_numpy = pred_01.transpose(0, 2, 3, 1)[0]
                
                # 计算多波段平均指标
                psnr_val = np.mean([psnr(gt_numpy[:, :, b], pred_numpy[:, :, b], MAX=1.0) for b in range(gt_numpy.shape[-1])])
                ssim_val = np.mean([ssim(gt_numpy[:, :, b], pred_numpy[:, :, b], MAX=1.0)[0] for b in range(gt_numpy.shape[-1])])
                
                all_metrics['psnr'].append(psnr_val)
                all_metrics['ssim'].append(ssim_val)
                all_metrics['ergas'].append(ergas(gt_numpy, pred_numpy))
                all_metrics['sam'].append(sam(gt_numpy, pred_numpy))

                # 可选：在有效（无云）像素上计算 Masked-PSNR
                if data.get('indicating_mask') is not None:
                    ind_mask = data['indicating_mask'].float()
                    if ind_mask.ndim == 3:
                        ind_mask = ind_mask.unsqueeze(0)  # (1,T,H,W)
                    # 时序聚合并对齐到预测分辨率（避免 64x64 掩膜索引 192x192 图像）
                    spatial_mask = ind_mask[0].max(dim=0, keepdim=True)[0].unsqueeze(0)  # (1,1,H,W)
                    pred_h, pred_w = pred_numpy.shape[0], pred_numpy.shape[1]
                    if spatial_mask.shape[-2:] != (pred_h, pred_w):
                        spatial_mask = F.interpolate(
                            spatial_mask,
                            size=(pred_h, pred_w),
                            mode='bilinear',
                            align_corners=False,
                        )
                    spatial_mask_np = spatial_mask.squeeze(0).squeeze(0).cpu().numpy() > 0.5
                    if spatial_mask_np.sum() > 0:
                        per_band_mpsnr = []
                        for b in range(gt_numpy.shape[-1]):
                            diff_sq = (gt_numpy[:, :, b][spatial_mask_np] - pred_numpy[:, :, b][spatial_mask_np]) ** 2
                            mse = diff_sq.mean()
                            if mse > 0:
                                per_band_mpsnr.append(20.0 * np.log10(1.0 / np.sqrt(mse)))
                        if per_band_mpsnr:
                            all_metrics['masked_psnr'].append(float(np.mean(per_band_mpsnr)))

                # 可视化第一个样本
                if ii == 0 and self.configs.train.get('local_logging', False):
                    self.visualize_validation_sample(data, predictions, ii)
            
            # 计算平均指标（排除空的 masked_psnr 列表）
            avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
            # 打印验证指标
            masked_psnr_str = (
                f" | Masked-PSNR: {avg_metrics['masked_psnr']:.4f}"
                if 'masked_psnr' in avg_metrics else ""
            )
            self.logger.info(
                f"📊 Validation Metrics | "
                f"PSNR: {avg_metrics['psnr']:.4f} | "
                f"SSIM: {avg_metrics['ssim']:.4f} | "
                f"ERGAS: {avg_metrics['ergas']:.4f} | "
                f"SAM: {avg_metrics['sam']:.4f}"
                + masked_psnr_str
            )
            
            # 记录指标
            self.log_data['val_psnr']['iters'].append(self.current_iters)
            self.log_data['val_psnr']['values'].append(avg_metrics['psnr'])
            self.log_data['val_ssim']['iters'].append(self.current_iters)
            self.log_data['val_ssim']['values'].append(avg_metrics['ssim'])
            
            self.model.train()
            return avg_metrics['psnr']

    def visualize_validation_sample(self, data, predictions, sample_idx):
        """健壮的可视化函数：过滤非基础波段，适配T*C扁平化通道"""
        try:
            # 1. 读取基础波段配置
            num_base_bands = self.configs.data.train.params.get('num_lr_bands', 9)
            C_total = self.configs.model.params.in_chans
            
            # 2. 计算时间维度T（拆分T*C合并维度）
            time_band_enabled = self.configs.features.time_band.enabled
            mask_band_enabled = self.configs.features.mask_band.enabled
            T = C_total // (num_base_bands + time_band_enabled + mask_band_enabled)
            
            # 3. 重塑lr_sequence为(B, T, C_per_T, H, W)
            lr_seq_shape = data['lr_sequence'].shape
            lr_reshaped = data['lr_sequence'].view(
                -1, T, C_total // T, lr_seq_shape[-2], lr_seq_shape[-1]
            )
            
            # 4. 取第一个时相的基础波段
            lr_vis_tensor = lr_reshaped[sample_idx, 0, :num_base_bands]
            gt_vis_tensor = data['gt'][sample_idx]
            pred_vis_tensor = predictions[sample_idx]

            # 5. 归一化用于可视化
            lr_vis = self.norm_for_vis(lr_vis_tensor)
            gt_vis = self.norm_for_vis(gt_vis_tensor)
            pred_vis = self.norm_for_vis(pred_vis_tensor)

            # 6. 计算误差图（按通道均值）
            error_map = np.abs(pred_vis - gt_vis).mean(axis=2)

            # 7. 创建可视化面板
            fig, axes = plt.subplots(2, 2, figsize=(14, 14))
            fig.suptitle(f'Validation Iter {self.current_iters}', fontsize=16)

            # 8. RGB通道配置（默认[0,1,2]）
            rgb_chn = self.configs.train.get('rgb_chn', [0, 1, 2])

            # 9. 绘制子图
            axes[0,0].imshow(lr_vis.transpose(1,2,0)[:, :, rgb_chn]); axes[0,0].set_title('Input LR (First Timestep)')
            axes[0,1].imshow(pred_vis.transpose(1,2,0)[:, :, rgb_chn]); axes[0,1].set_title('Prediction (SR)')
            axes[1,0].imshow(gt_vis.transpose(1,2,0)[:, :, rgb_chn]); axes[1,0].set_title('Ground Truth (HR)')
            im = axes[1,1].imshow(error_map, cmap='hot'); axes[1,1].set_title('Absolute Error Map')
            fig.colorbar(im, ax=axes[1,1])

            # 10. 隐藏坐标轴
            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            # 11. 保存图片
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = self.image_dir / 'val' / f"iter_{self.current_iters}_sample_{sample_idx}.png"
            plt.savefig(str(save_path), dpi=150)
            plt.close(fig)

        except Exception as e:
            self.logger.warning(f"⚠️ Visualization failed: {e}")

    def baseline_visualize(self):
        """训练前生成基线可视化（对比初始模型效果）"""
        if self.rank == 0 and self.configs.train.get('local_logging', False):
            self.logger.info("🎨 Generating baseline visualization...")
            self.model.eval()
            # try:
            data = next(iter(self.dataloaders['val']))
            data = self.prepare_data(data)
            with torch.no_grad():
                predictions = self.model(data)
            self.visualize_validation_sample(data, predictions, 0)
            self.logger.info(f"✅ Baseline visualization saved to: {self.image_dir / 'val'}")
            # except Exception as e:
            #     self.logger.warning(f"⚠️ Baseline visualization failed: {e}")
            self.model.train()

    def plot_curves(self):
        """生成训练/验证曲线（损失/PSNR/SSIM）"""
        if self.rank == 0 and hasattr(self, 'log_data'):
            self.logger.info("📊 Generating training curves...")
            fig, ax1 = plt.subplots(figsize=(12, 7))

            # 绘制训练损失（对数刻度）
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

            # 绘制验证PSNR/SSIM
            if self.log_data['val_psnr']['iters']:
                ax2 = ax1.twinx()
                ax2.set_ylabel('Validation PSNR (dB) / SSIM', color='tab:blue')
                ax2.plot(
                    self.log_data['val_psnr']['iters'],
                    self.log_data['val_psnr']['values'],
                    color='tab:blue', marker='o', linestyle='-', markersize=5, label='PSNR (dB)'
                )
                ax2.plot(
                    self.log_data['val_ssim']['iters'],
                    self.log_data['val_ssim']['values'],
                    color='tab:green', marker='x', linestyle='--', label='SSIM'
                )
                ax2.tick_params(axis='y', labelcolor='tab:blue')
                # 合并图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

            # 保存曲线
            fig.suptitle('Training & Validation Curves', fontsize=16)
            fig.tight_layout()
            save_path = self.save_dir / "training_curves.png"
            plt.savefig(str(save_path), dpi=150)
            plt.close(fig)
            self.logger.info(f"✅ Training curves saved to: {save_path}")
