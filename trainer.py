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
import matplotlib

matplotlib.use('Agg')
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
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.local_rank = 0
        if world_size > 1:
            # 分布式训练初始化
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.local_rank = local_rank % max(num_gpus, 1)
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                timeout=datetime.timedelta(seconds=3600),
                backend='nccl',
                init_method='env://'
            )
            self.num_gpus = world_size
            self.rank = int(os.environ.get('RANK', local_rank))
        else:
            self.num_gpus = 1 if torch.cuda.is_available() else 0
            self.rank = 0

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
        if self.rank == 0:
            self.image_dir = self.save_dir / 'images'
            (self.image_dir / 'val').mkdir(parents=True, exist_ok=True)

    def _train_config_has(self, key):
        try:
            return key in self.configs.train
        except Exception:
            return False

    def _validation_visualization_enabled(self):
        for key in ('visualize_validation', 'visualize_val', 'local_logging'):
            if self._train_config_has(key):
                return bool(self.configs.train.get(key, False))
        return True

    def _curve_plotting_enabled(self):
        if self._train_config_has('plot_curves_during_training'):
            return bool(self.configs.train.get('plot_curves_during_training', True))
        return True

    def build_dataloader(self):
        """构建训练/验证数据加载器（兼容分布式采样）"""
        def _wrap(loader):
            """无限迭代的数据加载器包装器"""
            while True:
                yield from loader

        build_val = hasattr(self.configs.data, 'val') and self.rank == 0
        val_skip_reason = None
        if build_val and not bool(self.configs.train.get('build_val_dataloader', True)):
            build_val = False
            val_skip_reason = 'build_val_dataloader=false'
        if build_val and bool(self.configs.train.get('skip_val_if_not_reached', True)):
            val_freq = int(self.configs.train.get('val_freq', 0) or 0)
            iterations = int(self.configs.train.get('iterations', 0) or 0)
            if val_freq <= 0 or val_freq > iterations:
                build_val = False
                val_skip_reason = f'val_freq={val_freq} is outside iterations={iterations}'

        # 创建数据集
        datasets = {'train': create_dataset(self.configs.data.train, parent_configs=self.configs)}
        if build_val:
            datasets['val'] = create_dataset(self.configs.data.val, parent_configs=self.configs)

        # 打印数据集大小（仅主进程）
        if self.rank == 0:
            for phase, ds in datasets.items():
                self.logger.info(f'📊 Dataset [{phase}] size: {len(ds)}')
            if val_skip_reason:
                self.logger.info(f'📊 Dataset [val] skipped: {val_skip_reason}')

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
        train_num_workers = int(self.configs.train.get('num_workers', 0))
        val_num_workers = int(self.configs.train.get('val_num_workers', 0))
        pin_memory = bool(self.configs.train.get('pin_memory', False))
        persistent_workers = bool(self.configs.train.get('persistent_workers', train_num_workers > 0))
        prefetch_factor = int(self.configs.train.get('prefetch_factor', 2))

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

        # 构建数据加载器
        self.dataloaders = {
            'train': _wrap(udata.DataLoader(
                datasets['train'],
                **train_loader_kwargs
            ))
        }
        if build_val:
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
                device_ids=[self.local_rank],
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
            return {
                k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in data.items()
            }
        return data.cuda(non_blocking=True)

    def _model_for_rank0_eval(self):
        """Rank-0-only validation should bypass DDP wrapper to avoid DDP forward collectives."""
        return self.model.module if isinstance(self.model, DDP) else self.model

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
                ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.local_rank}")
            except Exception as e:
                err = str(e)
                if "Weights only load failed" in err:
                    if self.rank == 0:
                        self.logger.warning(
                            "⚠️ torch.load safe mode failed; retrying with weights_only=False for trusted local checkpoint."
                        )
                    try:
                        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.local_rank}", weights_only=False)
                    except TypeError:
                        # 兼容旧版 PyTorch（无 weights_only 参数）
                        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.local_rank}")
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
        if self.num_gpus > 1:
            dist.barrier()
        
        self.model.train()
        save_freq = int(self.configs.train.get('save_freq', 0) or 0)
        val_freq = int(self.configs.train.get('val_freq', 0) or 0)
        save_before_val = bool(self.configs.train.get('save_before_val', True))
        curve_freq = int(
            self.configs.train.get(
                'plot_curve_freq',
                self.configs.train.get('curve_freq', 0),
            ) or 0
        )
        if curve_freq <= 0:
            if val_freq > 0 and 'val' in self.dataloaders:
                curve_freq = val_freq
            else:
                curve_freq = save_freq
        # 迭代训练
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1
            
            # 分布式采样器更新epoch（保证多GPU数据不重复）
            if hasattr(self, 'sampler') and self.sampler is not None:
                self.sampler.set_epoch(ii)
            
            # 加载批次数据并执行训练步
            # import pdb;pdb.set_trace()
            t0 = time.perf_counter()
            batch = next(self.dataloaders['train'])
            data_wait_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            data = self.prepare_data(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            to_device_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            self.training_step(data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            train_step_time = time.perf_counter() - t0
            self._log_train_timing(data_wait_time, to_device_time, train_step_time)

            is_save_iter = save_freq > 0 and (self.current_iters % save_freq) == 0
            is_val_iter = val_freq > 0 and (self.current_iters % val_freq) == 0
            
            # 保存普通检查点。默认先保存再验证，避免长验证/卡住时丢掉当前迭代模型。
            if is_save_iter and save_before_val:
                self.save_ckpt(best=False)
            
            # 验证（按验证频率）
            if is_val_iter:
                if self.num_gpus > 1:
                    dist.barrier()
                cur_metric = None
                if 'val' in self.dataloaders:
                    cur_metric = self.validation()
                if self.num_gpus > 1:
                    metric_tensor = torch.tensor(
                        [float(cur_metric) if cur_metric is not None else float('nan')],
                        device=torch.device('cuda', self.local_rank),
                    )
                    dist.broadcast(metric_tensor, src=0)
                    cur_metric = float(metric_tensor.item())
                if cur_metric is not None and math.isfinite(cur_metric):
                    self.adjust_lr(cur_metric)  # 调整学习率
                    # 保存最佳模型
                    if self.rank == 0 and cur_metric > self.best_metric:
                        self.best_metric = cur_metric
                        self.logger.info(f"🏆 New best metric: {self.best_metric:.4f} | Saving best model...")
                        self.save_ckpt(best=True)
                if self.num_gpus > 1:
                    dist.barrier()
            
            if is_save_iter and not save_before_val:
                self.save_ckpt(best=False)

            if (
                self.rank == 0
                and self._curve_plotting_enabled()
                and curve_freq > 0
                and (self.current_iters % curve_freq) == 0
            ):
                self.plot_curves()
        
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

    @staticmethod
    def norm_for_metric_tensor(img_tensor):
        """将[-1,1]裁剪并映射到[0,1]，保留在当前设备上用于快速指标。"""
        return (img_tensor.clamp(-1, 1) + 1.0) * 0.5

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
                lr_val = float(self.optimizer.param_groups[0]['lr'])
                self.logger.info(f"📈 Iter {self.current_iters} | Train Loss: {loss.item():.6f} | LR: {lr_val:.3e}")

    def _log_train_timing(self, data_wait_time, to_device_time, train_step_time):
        """Optional timing/GPU-memory log for locating CPU input stalls vs CUDA compute."""
        if self.rank != 0:
            return
        freq = int(self.configs.train.get('train_timing_log_freq', 0) or 0)
        if freq <= 0 or self.current_iters % freq != 0:
            return

        msg = (
            f"⏱️ Train timing | iter={self.current_iters} "
            f"data_wait={data_wait_time:.3f}s "
            f"to_device={to_device_time:.3f}s "
            f"forward_backward={train_step_time:.3f}s"
        )
        if torch.cuda.is_available():
            device = torch.device('cuda', self.local_rank)
            allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
            max_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            msg += (
                f" gpu_mem_alloc={allocated_gb:.2f}GB"
                f" gpu_mem_reserved={reserved_gb:.2f}GB"
                f" gpu_mem_max_alloc={max_allocated_gb:.2f}GB"
            )
        self.logger.info(msg)

    def _log_batch_debug(self, data, predictions=None, loss=None, phase='train'):
        """Optional compact tensor statistics for diagnosing train/val drift."""
        if self.rank != 0:
            return
        interval = int(self.configs.train.get('debug_batch_log_freq', 0) or 0)
        max_logs = int(self.configs.train.get('debug_batch_log_max', 20) or 0)
        if interval <= 0:
            return
        counter_name = f'_debug_{phase}_batch_logs'
        emitted = int(getattr(self, counter_name, 0))
        if max_logs > 0 and emitted >= max_logs:
            return
        if self.current_iters > 5 and self.current_iters % interval != 0:
            return

        parts = [f"🔎 {phase} debug iter={self.current_iters}"]
        for key in ('lr_sequence', 'gt', 'mask', 'mask_prob', 'indicating_mask', 'timestamps'):
            tensor = data.get(key) if isinstance(data, dict) else None
            if not torch.is_tensor(tensor):
                continue
            tf = tensor.detach().float()
            finite = torch.isfinite(tf)
            if finite.any():
                vals = tf[finite]
                parts.append(
                    f"{key}: shape={tuple(tensor.shape)} min={vals.min().item():.4f} "
                    f"mean={vals.mean().item():.4f} max={vals.max().item():.4f}"
                )
            else:
                parts.append(f"{key}: shape={tuple(tensor.shape)} all_nonfinite")
        if predictions is not None and torch.is_tensor(predictions):
            pf = predictions.detach().float()
            parts.append(
                f"pred: shape={tuple(predictions.shape)} min={pf.min().item():.4f} "
                f"mean={pf.mean().item():.4f} max={pf.max().item():.4f}"
            )
        if loss is not None and torch.is_tensor(loss):
            parts.append(f"loss={float(loss.detach().item()):.6f}")
        if isinstance(data, dict) and data.get('path') is not None:
            paths = data.get('path')
            preview = [str(p) for p in paths[:3]] if isinstance(paths, (list, tuple)) else [str(paths)]
            parts.append(f"paths={preview}")
        self.logger.info(" | ".join(parts))
        setattr(self, counter_name, emitted + 1)

    def _reduce_temporal_mask(self, mask_bt_hw):
        """将 (B,T,H,W) mask 聚合到 (B,1,H,W)，用于训练/验证中的空间 mask。"""
        strategy = self.configs.train.get('indicating_mask_reduce', 'max')
        mask_bt_hw = mask_bt_hw.clamp(0.0, 1.0)

        if bool(self.configs.train.get('indicating_mask_binarize', False)):
            threshold = float(self.configs.train.get('indicating_mask_threshold', 0.5))
            mask_bt_hw = (mask_bt_hw >= threshold).float()

        if mask_bt_hw.shape[1] == 1:
            return mask_bt_hw
        if strategy == 'mean':
            return mask_bt_hw.mean(dim=1, keepdim=True)
        if strategy == 'max':
            return mask_bt_hw.max(dim=1, keepdim=True).values
        if strategy == 'prob_or':
            return 1.0 - torch.prod(1.0 - mask_bt_hw, dim=1, keepdim=True)

        raise ValueError(f"Unknown indicating_mask_reduce strategy: {strategy}")

    def _normalize_indicating_mask(self, ind_mask, batch_size):
        """将 indicating_mask 统一成 (B,T,H,W) 或 (B,1,H,W)。"""
        if ind_mask is None:
            return None
        ind_mask = ind_mask.float()
        if ind_mask.ndim == 5:
            if ind_mask.shape[1] == 1:
                ind_mask = ind_mask.squeeze(1)
            elif ind_mask.shape[2] == 1:
                ind_mask = ind_mask.squeeze(2)
            else:
                raise ValueError(f"Unsupported indicating_mask shape: {tuple(ind_mask.shape)}")
        elif ind_mask.ndim == 3:
            if ind_mask.shape[0] == batch_size:
                ind_mask = ind_mask.unsqueeze(1)
            else:
                ind_mask = ind_mask.unsqueeze(0)
        elif ind_mask.ndim != 4:
            raise ValueError(f"Unexpected indicating_mask shape: {tuple(ind_mask.shape)}")

        if ind_mask.shape[0] != batch_size:
            if ind_mask.shape[0] == 1:
                ind_mask = ind_mask.expand(batch_size, -1, -1, -1)
            else:
                raise ValueError(
                    f"Batch mismatch between indicating_mask ({ind_mask.shape[0]}) "
                    f"and validation batch ({batch_size})"
                )
        return ind_mask

    def _compute_fast_validation_metrics(self, gt_01, pred_01, spatial_masks=None):
        """GPU 快路径：逐样本、逐波段计算 PSNR，并可选计算 Masked-PSNR。"""
        diff_sq = (gt_01 - pred_01).pow(2)
        mse_bc = diff_sq.flatten(2).mean(dim=2).clamp_min(1e-12)
        band_psnr = 10.0 * torch.log10(1.0 / mse_bc)
        sample_psnr = band_psnr.mean(dim=1)

        metrics = {
            'band_psnr': band_psnr.detach().cpu().numpy(),
            'sample_psnr': sample_psnr.detach().cpu().numpy(),
        }

        if spatial_masks is not None:
            if spatial_masks.shape[-2:] != pred_01.shape[-2:]:
                spatial_masks = F.interpolate(
                    spatial_masks,
                    size=pred_01.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            valid = (spatial_masks > 0.5).float()
            valid_bc = valid.expand_as(diff_sq)
            valid_count_bc = valid_bc.flatten(2).sum(dim=2)
            masked_mse_sum_bc = (diff_sq * valid_bc).flatten(2).sum(dim=2)
            has_valid = valid_count_bc > 0
            masked_mse_bc = torch.where(
                has_valid,
                masked_mse_sum_bc / valid_count_bc.clamp_min(1.0),
                torch.ones_like(masked_mse_sum_bc),
            ).clamp_min(1e-12)
            masked_band_psnr = 10.0 * torch.log10(1.0 / masked_mse_bc)
            masked_sample_valid = has_valid.any(dim=1)
            if masked_sample_valid.any():
                masked_sample_psnr = masked_band_psnr.mean(dim=1)[masked_sample_valid]
                metrics['masked_sample_psnr'] = masked_sample_psnr.detach().cpu().numpy()

        return metrics

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
            # indicating_mask 通常在 LR 空间，而预测在 SR 空间；需要先对齐分辨率再监督。
            ind_mask = data['indicating_mask'].float()

            # 归一化到 (B,T,H,W) 或 (B,1,H,W)
            if ind_mask.ndim == 5:
                if ind_mask.shape[1] == 1:
                    ind_mask = ind_mask.squeeze(1)
                elif ind_mask.shape[2] == 1:
                    ind_mask = ind_mask.squeeze(2)
                else:
                    raise ValueError(f"Unsupported indicating_mask shape: {tuple(ind_mask.shape)}")
            elif ind_mask.ndim == 3:
                if ind_mask.shape[0] == predictions.shape[0]:
                    ind_mask = ind_mask.unsqueeze(1)
                else:
                    ind_mask = ind_mask.unsqueeze(0)
            elif ind_mask.ndim != 4:
                raise ValueError(f"Unexpected indicating_mask ndim={ind_mask.ndim}, shape={tuple(ind_mask.shape)}")

            if ind_mask.shape[0] != predictions.shape[0]:
                if ind_mask.shape[0] == 1:
                    ind_mask = ind_mask.expand(predictions.shape[0], -1, -1, -1)
                else:
                    raise ValueError(
                        f"Batch mismatch between indicating_mask ({ind_mask.shape[0]}) and predictions ({predictions.shape[0]})"
                    )

            spatial_mask = self._reduce_temporal_mask(ind_mask)
            # 对齐到 SR 尺度
            if spatial_mask.shape[-2:] != predictions.shape[-2:]:
                spatial_mask = F.interpolate(
                    spatial_mask,
                    size=predictions.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            spatial_mask = spatial_mask.expand_as(predictions)
            valid_pixels = spatial_mask.sum().clamp(min=1)
            loss = self.criterion(predictions * spatial_mask, data['gt'] * spatial_mask)
            loss = loss * (float(predictions.numel()) / valid_pixels)
        else:
            loss = self.criterion(predictions, data['gt'])

        self._log_batch_debug(data, predictions=predictions, loss=loss, phase='train')
        
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
            per_band_psnr_values = []
            per_sample_psnr_values = []
            val_debug_max = int(self.configs.train.get('val_debug_max_samples', 8) or 0)
            compute_expensive_metrics = bool(self.configs.train.get('val_compute_expensive_metrics', False))
            log_timing = bool(self.configs.train.get('val_timing_log', True))
            timing = {'data_wait': 0.0, 'to_device': 0.0, 'forward': 0.0, 'metrics': 0.0, 'visualize': 0.0}
            val_loader = self.dataloaders[phase]
            val_iter = iter(val_loader)
            total_val_batches = len(val_loader)
            run_val_batches = total_val_batches
            limit_reasons = []
            val_max_batches = int(self.configs.train.get('val_max_batches', 0) or 0)
            val_max_samples = int(self.configs.train.get('val_max_samples', 0) or 0)
            if val_max_batches > 0:
                run_val_batches = min(run_val_batches, val_max_batches)
                limit_reasons.append(f"val_max_batches={val_max_batches}")
            if val_max_samples > 0:
                batch_size = int(getattr(val_loader, 'batch_size', 1) or 1)
                sample_limited_batches = int(math.ceil(val_max_samples / max(batch_size, 1)))
                run_val_batches = min(run_val_batches, sample_limited_batches)
                limit_reasons.append(f"val_max_samples={val_max_samples}")
            if run_val_batches <= 0:
                self.logger.warning("⚠️ Validation skipped: no validation batches selected.")
                self.model.train()
                return None
            if run_val_batches < total_val_batches:
                self.logger.info(
                    f"🔎 Validation capped | batches={run_val_batches}/{total_val_batches} "
                    f"({', '.join(limit_reasons)})"
                )
            pbar = tqdm(range(run_val_batches), desc=f"📌 Val Iter {getattr(self, 'current_iters', 0)}")
            
            for ii in pbar:
                t0 = time.perf_counter()
                data = next(val_iter)
                timing['data_wait'] += time.perf_counter() - t0

                t0 = time.perf_counter()
                data = self.prepare_data(data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing['to_device'] += time.perf_counter() - t0
                
                # 模型推理
                eval_model = self._model_for_rank0_eval()
                t0 = time.perf_counter()
                predictions = eval_model(data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing['forward'] += time.perf_counter() - t0
                self._log_batch_debug(data, predictions=predictions, phase=phase)
                
                t0 = time.perf_counter()
                # 归一化到[0,1]用于指标计算
                gt_01_tensor = self.norm_for_metric_tensor(data['gt'])
                pred_01_tensor = self.norm_for_metric_tensor(predictions)

                ind_mask = data.get('indicating_mask', None)
                if ind_mask is not None:
                    ind_mask = self._normalize_indicating_mask(ind_mask, batch_size=gt_01_tensor.shape[0])
                    spatial_masks = self._reduce_temporal_mask(ind_mask)
                else:
                    spatial_masks = None

                fast_metrics = self._compute_fast_validation_metrics(
                    gt_01_tensor,
                    pred_01_tensor,
                    spatial_masks=spatial_masks,
                )
                batch_band_psnr = fast_metrics['band_psnr']
                batch_sample_psnr = fast_metrics['sample_psnr']
                sample_offset = len(per_sample_psnr_values)
                per_band_psnr_values.extend(batch_band_psnr.tolist())
                per_sample_psnr_values.extend(float(v) for v in batch_sample_psnr)
                all_metrics['psnr'].extend(float(v) for v in batch_sample_psnr)
                if 'masked_sample_psnr' in fast_metrics:
                    all_metrics['masked_psnr'].extend(float(v) for v in fast_metrics['masked_sample_psnr'])

                if compute_expensive_metrics:
                    gt_01 = gt_01_tensor.cpu().numpy()
                    pred_01 = pred_01_tensor.cpu().numpy()
                    gt_numpy_batch = gt_01.transpose(0, 2, 3, 1)
                    pred_numpy_batch = pred_01.transpose(0, 2, 3, 1)

                    for batch_idx in range(gt_numpy_batch.shape[0]):
                        gt_numpy = gt_numpy_batch[batch_idx]
                        pred_numpy = pred_numpy_batch[batch_idx]

                        # CPU 参考指标较慢；训练中调度和 best checkpoint 只依赖 PSNR。
                        ssim_val = np.mean([ssim(gt_numpy[:, :, b], pred_numpy[:, :, b], MAX=1.0)[0] for b in range(gt_numpy.shape[-1])])
                        all_metrics['ssim'].append(ssim_val)
                        all_metrics['ergas'].append(ergas(gt_numpy, pred_numpy))
                        all_metrics['sam'].append(sam(gt_numpy, pred_numpy))

                for batch_idx, psnr_val in enumerate(batch_sample_psnr):
                    global_sample_idx = sample_offset + batch_idx
                    if val_debug_max > 0 and global_sample_idx < val_debug_max:
                        path_info = None
                        if isinstance(data, dict) and data.get('path') is not None:
                            paths = data.get('path')
                            if isinstance(paths, (list, tuple)) and batch_idx < len(paths):
                                path_info = paths[batch_idx]
                            else:
                                path_info = paths
                        self.logger.info(
                            f"🔎 Val sample debug | iter={self.current_iters} idx={global_sample_idx} "
                            f"psnr={float(psnr_val):.4f} "
                            f"gt_mean={float(gt_01_tensor[batch_idx].mean().item()):.4f} "
                            f"pred_mean={float(pred_01_tensor[batch_idx].mean().item()):.4f} "
                            f"path={path_info}"
                        )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing['metrics'] += time.perf_counter() - t0

                # 可视化验证样本
                val_visualize_max = int(
                    self.configs.train.get(
                        'val_visualize_max_samples',
                        self.configs.train.get('val_vis_max_samples', 1),
                    ) or 0
                )
                if ii == 0 and self._validation_visualization_enabled() and val_visualize_max > 0:
                    t0 = time.perf_counter()
                    batch_vis_count = min(val_visualize_max, int(predictions.shape[0]))
                    for sample_idx in range(batch_vis_count):
                        self.visualize_validation_sample(data, predictions, sample_idx)
                    timing['visualize'] += time.perf_counter() - t0
            
            # 计算平均指标（排除空的 masked_psnr 列表）
            avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
            if 'psnr' not in avg_metrics:
                self.logger.warning("⚠️ Validation produced no PSNR values; skip metric update.")
                self.model.train()
                return None
            # 打印验证指标
            masked_psnr_str = (
                f" | Masked-PSNR: {avg_metrics['masked_psnr']:.4f}"
                if 'masked_psnr' in avg_metrics else ""
            )
            slow_metrics_str = (
                f" | SSIM: {avg_metrics['ssim']:.4f} | "
                f"ERGAS: {avg_metrics['ergas']:.4f} | SAM: {avg_metrics['sam']:.4f}"
                if compute_expensive_metrics
                else " | SSIM/ERGAS/SAM: skipped"
            )
            self.logger.info(
                f"📊 Validation Metrics | "
                f"PSNR: {avg_metrics['psnr']:.4f}"
                + slow_metrics_str
                + masked_psnr_str
            )
            if per_sample_psnr_values:
                sample_arr = np.asarray(per_sample_psnr_values, dtype=np.float32)
                self.logger.info(
                    f"🔎 Validation PSNR spread | min={sample_arr.min():.4f} "
                    f"median={np.median(sample_arr):.4f} max={sample_arr.max():.4f} n={sample_arr.size}"
                )
            if per_band_psnr_values:
                band_arr = np.asarray(per_band_psnr_values, dtype=np.float32)
                band_mean = band_arr.mean(axis=0)
                band_msg = ", ".join(f"b{i}:{v:.2f}" for i, v in enumerate(band_mean[:16]))
                if band_mean.size > 16:
                    band_msg += ", ..."
                self.logger.info(f"🔎 Validation per-band PSNR mean | {band_msg}")
            if log_timing:
                total_timing = sum(timing.values())
                self.logger.info(
                    "⏱️ Validation timing | "
                    f"data_wait={timing['data_wait']:.2f}s "
                    f"to_device={timing['to_device']:.2f}s "
                    f"forward={timing['forward']:.2f}s "
                    f"metrics={timing['metrics']:.2f}s "
                    f"visualize={timing['visualize']:.2f}s "
                    f"total={total_timing:.2f}s"
                )
            
            # 记录指标
            self.log_data['val_psnr']['iters'].append(self.current_iters)
            self.log_data['val_psnr']['values'].append(avg_metrics['psnr'])
            if 'ssim' in avg_metrics:
                self.log_data['val_ssim']['iters'].append(self.current_iters)
                self.log_data['val_ssim']['values'].append(avg_metrics['ssim'])
            
            self.model.train()
            return avg_metrics['psnr']

    def visualize_validation_sample(self, data, predictions, sample_idx):
        """健壮的可视化函数：过滤非基础波段，适配T*C扁平化通道"""
        try:
            # 1. 读取基础波段配置
            train_params = self.configs.data.train.params
            num_base_bands = int(train_params.get('reflectance_band_count', train_params.get('num_lr_bands', 9)) or 9)

            # 2. 取第一个时相的基础波段
            lr_sequence = data['lr_sequence']
            if lr_sequence.ndim == 5:
                lr_vis_tensor = lr_sequence[sample_idx, 0, :num_base_bands]
            elif lr_sequence.ndim == 4:
                lr_vis_tensor = lr_sequence[sample_idx, :num_base_bands]
            else:
                raise ValueError(f"Unsupported lr_sequence shape for visualization: {tuple(lr_sequence.shape)}")
            gt_vis_tensor = data['gt'][sample_idx]
            pred_vis_tensor = predictions[sample_idx]

            # 3. 归一化用于可视化
            lr_vis = self.norm_for_vis(lr_vis_tensor)
            gt_vis = self.norm_for_vis(gt_vis_tensor)
            pred_vis = self.norm_for_vis(pred_vis_tensor)

            # 4. 计算误差图（按通道均值）
            error_map = np.abs(pred_vis - gt_vis).mean(axis=0)

            # 5. 创建可视化面板
            fig, axes = plt.subplots(2, 2, figsize=(14, 14))
            fig.suptitle(f'Validation Iter {self.current_iters}', fontsize=16)

            # 6. RGB通道配置（默认[0,1,2]）
            rgb_chn = [int(x) for x in self.configs.train.get('rgb_chn', [0, 1, 2])]
            lr_rgb = [min(max(ch, 0), lr_vis.shape[0] - 1) for ch in rgb_chn]
            pred_rgb = [min(max(ch, 0), pred_vis.shape[0] - 1) for ch in rgb_chn]
            gt_rgb = [min(max(ch, 0), gt_vis.shape[0] - 1) for ch in rgb_chn]

            # 7. 绘制子图
            axes[0,0].imshow(lr_vis.transpose(1,2,0)[:, :, lr_rgb]); axes[0,0].set_title('Input LR (First Timestep)')
            axes[0,1].imshow(pred_vis.transpose(1,2,0)[:, :, pred_rgb]); axes[0,1].set_title('Prediction (SR)')
            axes[1,0].imshow(gt_vis.transpose(1,2,0)[:, :, gt_rgb]); axes[1,0].set_title('Ground Truth (HR)')
            im = axes[1,1].imshow(error_map, cmap='hot'); axes[1,1].set_title('Absolute Error Map')
            fig.colorbar(im, ax=axes[1,1])

            # 8. 隐藏坐标轴
            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            # 9. 保存图片
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            (self.image_dir / 'val').mkdir(parents=True, exist_ok=True)
            save_path = self.image_dir / 'val' / f"iter_{self.current_iters}_sample_{sample_idx}.png"
            plt.savefig(str(save_path), dpi=150)
            plt.close(fig)

        except Exception as e:
            self.logger.warning(f"⚠️ Visualization failed: {e}")

    def baseline_visualize(self):
        """训练前生成基线可视化（对比初始模型效果）"""
        if self.rank == 0 and self._validation_visualization_enabled() and 'val' in self.dataloaders:
            self.logger.info("🎨 Generating baseline visualization...")
            self.model.eval()
            # try:
            data = next(iter(self.dataloaders['val']))
            data = self.prepare_data(data)
            with torch.no_grad():
                eval_model = self._model_for_rank0_eval()
                predictions = eval_model(data)
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
