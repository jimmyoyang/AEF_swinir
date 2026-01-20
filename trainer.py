# 文件路径: trainer.py
# (v_industrial_final - 工业级封装，修复了helper函数作用域问题)

import os, sys, math, time, random, datetime
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

# 假设您的其他模块路径正确
from datapipe.datasets import create_dataset
from utils import util_net, util_common
from models.losses import CompoundLoss # 假设您的复合损失在这里

# ==============================================================================
# 1. Trainer 基类 (封装通用逻辑)
# ==============================================================================
class TrainerBase:
    def __init__(self, configs):
        self.configs = configs
        self.init_dist_and_seed()
        self.init_logger()
        self.build_model()
        self.setup_optimization()
        self.build_dataloader()
        self.resume_from_ckpt()

    def init_dist_and_seed(self):
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            rank = int(os.environ.get('LOCAL_RANK', 0)); torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(timeout=datetime.timedelta(seconds=3600), backend='nccl', init_method='env://')
        self.num_gpus, self.rank = num_gpus, int(os.environ.get('LOCAL_RANK', 0)) if num_gpus > 1 else 0
        
        seed = self.configs.train.get('seed', 42)
        if self.num_gpus > 1 and not self.configs.train.get('global_seeding', True): seed += self.rank
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    def init_logger(self):
        if hasattr(self.configs, 'resume') and self.configs.resume:
            self.save_dir = Path(self.configs.resume).parents[1]
        else:
            run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = Path(self.configs.train.save_dir) / run_name
        
        if self.rank == 0: 
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.logger = logger; self.logger.remove()
            log_path = self.save_dir / 'training.log'
            self.logger.add(log_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")
            self.logger.info(OmegaConf.to_yaml(self.configs))
        
        self.ckpt_dir = self.save_dir / 'ckpts'
        if self.rank == 0: self.ckpt_dir.mkdir(exist_ok=True)
        
        if self.rank == 0 and self.configs.train.get('local_logging', False):
            self.image_dir = self.save_dir / 'images'
            (self.image_dir / 'val').mkdir(parents=True, exist_ok=True)

    def build_model(self):
        self.model = util_common.instantiate_from_config(self.configs.model).cuda()
        if self.num_gpus > 1: self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)

    def setup_optimization(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.configs.train.lr, weight_decay=self.configs.train.get('weight_decay', 0))

    def build_dataloader(self):
        def _wrap(loader):
            while True: yield from loader
        
        datasets = {'train': create_dataset(self.configs.data.train, parent_configs=self.configs)}
        if hasattr(self.configs.data, 'val') and self.rank == 0: 
            datasets['val'] = create_dataset(self.configs.data.val, parent_configs=self.configs)
        if self.rank == 0: [self.logger.info(f'Images in {p} set: {len(d)}') for p, d in datasets.items()]
        
        sampler = udata.distributed.DistributedSampler(datasets['train'], num_replicas=self.num_gpus, rank=self.rank) if self.num_gpus > 1 else None
        train_batch_size = self.configs.train.batch[0] // self.num_gpus if self.num_gpus > 0 else self.configs.train.batch[0]
        val_batch_size = self.configs.train.batch[1] if len(self.configs.train.batch) > 1 else 1

        self.dataloaders = {'train': _wrap(udata.DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=(sampler is None), drop_last=True, num_workers=self.configs.train.num_workers, sampler=sampler))}
        if hasattr(self.configs.data, 'val') and self.rank == 0: self.dataloaders['val'] = udata.DataLoader(datasets['val'], batch_size=val_batch_size, num_workers=0)
        self.datasets, self.sampler = datasets, sampler

    def prepare_data(self, data):
        if isinstance(data, dict):
            return {k: v.cuda(non_blocking=True) for k, v in data.items() if isinstance(v, torch.Tensor)}
        return data.cuda(non_blocking=True)

    def save_ckpt(self, best=False):
        if self.rank == 0:
            filename = 'model_best.pth' if best else f'model_{self.current_iters}.pth'
            ckpt_path = self.ckpt_dir / filename
            model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
            torch.save({'iters_start': self.current_iters, 'state_dict': model_state, 'best_metric': self.best_metric, 'optimizer': self.optimizer.state_dict()}, ckpt_path)
            if not best: self.logger.info(f"Saved safety checkpoint to {ckpt_path}")

    def resume_from_ckpt(self):
        self.iters_start, self.best_metric = 0, 0.0
        if hasattr(self.configs, 'resume') and self.configs.resume:
            ckpt_path = self.configs.resume
            assert Path(ckpt_path).exists(), f"Checkpoint not found at {ckpt_path}"
            if self.rank == 0: self.logger.info(f"=> Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
            util_net.reload_model(model_to_load, ckpt['state_dict'])
            self.iters_start = ckpt.get('iters_start', 0)
            self.best_metric = ckpt.get('best_metric', 0.0)
            if 'optimizer' in ckpt: self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.rank == 0: self.logger.info(f"=> Resumed from iteration {self.iters_start}. Previous best metric: {self.best_metric:.4f}")

    def train(self):
        if self.rank == 0 and self.iters_start == 0: self.baseline_visualize()
        self.model.train()
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1
            if hasattr(self, 'sampler') and self.sampler is not None: self.sampler.set_epoch(ii)
            
            data = self.prepare_data(next(self.dataloaders['train']))
            self.training_step(data)
            
            if 'val' in self.dataloaders and (self.current_iters % self.configs.train.val_freq) == 0:
                cur_metric = self.validation()
                self.adjust_lr(cur_metric)
                if self.rank == 0 and cur_metric > self.best_metric:
                    self.best_metric = cur_metric
                    self.logger.info(f"🎉 New best metric: {self.best_metric:.4f}. Saving best model...")
                    self.save_ckpt(best=True)
            
            if (self.current_iters % self.configs.train.save_freq) == 0: self.save_ckpt(best=False)
                
        if self.rank == 0: self.plot_curves(); self.logger.info("--- Training Finished ---")

# ==============================================================================
# 2. 针对AlphaSR任务的专用Trainer
# ==============================================================================
class TrainerAlphaSR(TrainerBase):
    def __init__(self, configs):
        super().__init__(configs)
        if self.rank == 0:
            self.log_data = {'train_total_loss':{'iters':[],'values':[]}, 'val_psnr':{'iters':[],'values':[]}, 'val_ssim':{'iters':[],'values':[]}}
            if hasattr(self.configs.train, 'loss'):
                for k in self.configs.train.loss.weights.keys():
                    self.log_data[f'train_{k}_loss'] = {'iters':[], 'values':[]}

    def setup_optimization(self):
        super().setup_optimization()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", **self.configs.train.lr_schedule_params)
        self.criterion = CompoundLoss(self.configs.train.loss.weights).cuda()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.configs.train.get('mixed_precision', False))

    def adjust_lr(self, metrics):
        self.scheduler.step(metrics)

    def training_step(self, data):
        with torch.cuda.amp.autocast(enabled=self.configs.train.mixed_precision):
            predictions = self.model(data)
            loss, loss_dict = self.criterion(predictions, data['gt'])
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.log_step_train(loss, loss_dict)

    def log_step_train(self, loss, loss_dict):
        if self.rank == 0:
            self.log_data['train_total_loss']['values'].append(loss.item())
            self.log_data['train_total_loss']['iters'].append(self.current_iters)
            for k, v in loss_dict.items():
                if f'train_{k}' not in self.log_data: self.log_data[f'train_{k}'] = {'iters':[], 'values':[]}
                self.log_data[f'train_{k}']['values'].append(v)
                self.log_data[f'train_{k}']['iters'].append(self.current_iters)

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                components_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                self.logger.info(f"Iter: {self.current_iters:06d}, Total Loss: {loss.item():.4e}, lr:{self.optimizer.param_groups[0]['lr']:.2e} | {components_str}")

    @torch.no_grad()
    def validation(self, phase='val'):
        if self.rank == 0:
            self.model.eval()
            all_metrics = {'psnr':[], 'ssim':[]}
            pbar = tqdm(self.dataloaders[phase], desc=f"Validation iter {self.current_iters}")
            
            for ii, data in enumerate(pbar):
                data = self.prepare_data(data)
                with torch.cuda.amp.autocast(enabled=self.configs.train.mixed_precision):
                    predictions = self.model(data)
                
                # 【核心修正】通过 self.norm_for_vis 调用
                gt_01, pred_01 = self.norm_for_vis(data['gt']), self.norm_for_vis(predictions)
                gt_numpy, pred_numpy = gt_01.transpose(0,2,3,1)[0], pred_01.transpose(0,2,3,1)[0]
                
                psnr_val=np.mean([psnr(gt_numpy[:,:,b], pred_numpy[:,:,b], MAX=1.0) for b in range(gt_numpy.shape[-1])])
                ssim_val=np.mean([ssim(gt_numpy[:,:,b], pred_numpy[:,:,b], MAX=1.0)[0] for b in range(gt_numpy.shape[-1])])
                all_metrics['psnr'].append(psnr_val); all_metrics['ssim'].append(ssim_val)
                
                if ii == 0 and self.configs.train.get('local_logging', False):
                    self.visualize_validation_sample(data, predictions, ii)
            
            avg_metrics={k:np.mean(v) for k,v in all_metrics.items()}
            self.logger.info(f"Validation Metrics -> " + " | ".join([f"{k.upper()}: {v:.4f}" for k, v in avg_metrics.items()]))
            
            self.log_data['val_psnr']['iters'].append(self.current_iters); self.log_data['val_psnr']['values'].append(avg_metrics['psnr'])
            self.log_data['val_ssim']['iters'].append(self.current_iters); self.log_data['val_ssim']['values'].append(avg_metrics['ssim'])
                
            self.model.train()
            return avg_metrics['psnr']
    
    def visualize_validation_sample(self, data, predictions, sample_idx):
        """封装的可视化函数"""
        lr_vis_tensor = data['lr_sequence'][sample_idx, 0] # 可视化第一个时相
        gt_vis_tensor = data['gt'][sample_idx]
        pred_vis_tensor = predictions[sample_idx]
        
        # 【核心修正】通过 self.norm_for_vis 调用
        lr_vis = self.norm_for_vis(lr_vis_tensor)
        gt_vis = self.norm_for_vis(gt_vis_tensor)
        pred_vis = self.norm_for_vis(pred_vis_tensor)
        
        error_map = np.abs(pred_vis - gt_vis).mean(axis=2) # H,W,C -> H,W
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        fig.suptitle(f'Validation at Iteration {self.current_iters}', fontsize=16)
        
        rgb_chn = self.configs.train.get('rgb_chn', [0, 1, 2])
        
        axes[0,0].imshow(lr_vis[:, :, rgb_chn]); axes[0,0].set_title('Input LR (First Timestep)')
        axes[0,1].imshow(pred_vis[:, :, rgb_chn]); axes[0,1].set_title('Prediction (SR)')
        axes[1,0].imshow(gt_vis[:, :, rgb_chn]); axes[1,0].set_title('Ground Truth (HR)')
        im = axes[1,1].imshow(error_map, cmap='hot'); axes[1,1].set_title('Absolute Error Map'); fig.colorbar(im, ax=axes[1,1])
        
        for ax in axes.flatten(): ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = self.image_dir / 'val' / f"iter_{self.current_iters}_sample_{sample_idx}.png"
        plt.savefig(str(save_path), dpi=150)
        plt.close(fig)

    def baseline_visualize(self):
        """【补全】适配字典式数据的 baseline 可视化"""
        if self.rank == 0 and self.configs.train.get('local_logging', False):
            self.logger.info("Generating baseline visualization...")
            self.model.eval()
            try:
                data = next(iter(self.dataloaders['val']))
                data = self.prepare_data(data)
                
                with torch.no_grad(): predictions = self.model(data)
                
                self.visualize_validation_sample(data, predictions, 0)
                save_path = self.image_dir / 'val' / "iter_0_baseline_comparison.png"
                self.logger.info(f"Baseline visualization saved to: {save_path}")
            except Exception as e:
                self.logger.warning(f"Could not generate baseline visualization. Error: {e}")
            self.model.train()

    def plot_curves(self):
        """【补全】适配复合损失的绘图逻辑"""
        if self.rank == 0 and hasattr(self, 'log_data'):
            self.logger.info("Generating training curves plot...")
            num_losses = len([k for k in self.log_data.keys() if 'train' in k and k != 'train_total_loss'])
            fig, axes = plt.subplots(2 + num_losses, 1, figsize=(12, 6 * (2 + num_losses)), sharex=True)
            
            # 绘制总损失
            axes[0].plot(self.log_data['train_total_loss']['iters'], self.log_data['train_total_loss']['values'], color='tab:red', label='Total Loss')
            axes[0].set_ylabel('Total Loss (Log Scale)'); axes[0].set_yscale('log'); axes[0].grid(True, linestyle=':')
            
            # 绘制PSNR和SSIM
            ax_psnr = axes[1]
            ax_ssim = ax_psnr.twinx()
            ax_psnr.plot(self.log_data['val_psnr']['iters'], self.log_data['val_psnr']['values'], color='tab:blue', marker='o', linestyle='-', markersize=4, label='PSNR (dB)')
            ax_ssim.plot(self.log_data['val_ssim']['iters'], self.log_data['val_ssim']['values'], color='tab:green', marker='x', linestyle='--', markersize=4, label='SSIM')
            ax_psnr.set_ylabel('Validation PSNR (dB)', color='tab:blue'); ax_ssim.set_ylabel('Validation SSIM', color='tab:green')
            ax_psnr.grid(True, linestyle=':')
            
            # 绘制每个损失分量
            plot_idx = 2
            for k in self.configs.train.loss.weights.keys():
                log_key = f'train_{k}_loss'
                if log_key in self.log_data and self.log_data[log_key]['values']:
                    axes[plot_idx].plot(self.log_data[log_key]['iters'], self.log_data[log_key]['values'], label=f'{k} Loss')
                    axes[plot_idx].set_ylabel(f'{k} Loss'); axes[plot_idx].grid(True); axes[plot_idx].set_yscale('log')
                    plot_idx += 1
            
            fig.suptitle('Training & Validation Curves', fontsize=16)
            plt.xlabel('Iterations')
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = self.save_dir / "training_curves.png"
            plt.savefig(str(save_path), dpi=150)
            plt.close(fig)
            self.logger.info(f"Training curves plot saved to: {save_path}")
