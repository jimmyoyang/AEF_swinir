# 文件路径: trainer.py
# (v_final_unified - 与 main.py v2 完全兼容的最终版)

import os, sys, math, time, random, datetime, numpy as np, torch
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
import torch.nn.functional as F, torch.utils.data as udata, torch.distributed as dist, torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from sewar.full_ref import psnr, ssim, ergas, sam
from tqdm import tqdm
import matplotlib.pyplot as plt
from datapipe.datasets import create_dataset
from utils import util_net, util_common

# ==============================================================================
# 1. Base Class (最终版)
# ==============================================================================
def my_worker_init_fn(worker_id): np.random.seed(np.random.get_state()[1][0] + worker_id)

class TrainerBase:
    def __init__(self, configs):
        self.configs = configs
        self.setup_dist()
        self.setup_seed()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            rank = int(os.environ.get('LOCAL_RANK', 0)); torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(timeout=datetime.timedelta(seconds=3600), backend='nccl', init_method='env://')
        self.num_gpus, self.rank = num_gpus, int(os.environ.get('LOCAL_RANK', 0)) if num_gpus > 1 else 0

    def setup_seed(self, seed=None):
        seed = self.configs.train.get('seed', 12345) if seed is None else seed
        if self.num_gpus > 1 and not self.configs.train.get('global_seeding', True): seed += self.rank
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    def init_logger(self):
        # 恢复时，save_dir 由 resume_from_ckpt 设置
        if self.configs.resume:
            self.save_dir = Path(self.configs.resume).parents[1]
        else: # 从头开始训练
            run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = Path(self.configs.train.save_dir) / run_name
        
        if self.rank == 0: 
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.logger = logger; self.logger.remove()
            log_path = self.save_dir / 'training.log'
            self.logger.add(log_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")
        
        self.ckpt_dir = self.save_dir / 'ckpts'
        if self.rank == 0: self.ckpt_dir.mkdir(exist_ok=True)
        
        self.local_logging = self.configs.train.get('local_logging', False)
        if self.rank == 0 and self.local_logging:
            self.image_dir = self.save_dir / 'images'
            (self.image_dir / 'val').mkdir(parents=True, exist_ok=True)
        
        self.rgb_chn = self.configs.train.rgb_chn
        if self.rank == 0: self.logger.info(OmegaConf.to_yaml(self.configs))

    def build_model(self):
        self.model = util_common.instantiate_from_config(self.configs.model).cuda()
        if self.num_gpus > 1: self.model = DDP(self.model, device_ids=[self.rank])

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.configs.train.lr, weight_decay=self.configs.train.get('weight_decay', 0))

    def build_dataloader(self):
        def _wrap(loader):
            while True: yield from loader
        
        datasets = {'train': create_dataset(self.configs.data.train)}
        if hasattr(self.configs.data, 'val') and self.rank == 0: datasets['val'] = create_dataset(self.configs.data.val)
        if self.rank == 0: [self.logger.info(f'Images in {p} set: {len(d)}') for p, d in datasets.items()]
        
        sampler = udata.distributed.DistributedSampler(datasets['train'], num_replicas=self.num_gpus, rank=self.rank) if self.num_gpus > 1 else None
        
        train_batch_size = self.configs.train.batch[0] // self.num_gpus if self.num_gpus > 0 else self.configs.train.batch[0]
        val_batch_size = self.configs.train.batch[1] if len(self.configs.train.batch) > 1 else 1

        self.dataloaders = {'train': _wrap(udata.DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=(sampler is None), drop_last=True, num_workers=self.configs.train.num_workers, sampler=sampler))}
        if hasattr(self.configs.data, 'val') and self.rank == 0: self.dataloaders['val'] = udata.DataLoader(datasets['val'], batch_size=val_batch_size, num_workers=0)
        self.datasets, self.sampler = datasets, sampler

    def prepare_data(self, data):
        # 确保只将 Tensor 移动到GPU
        return {k: v.cuda() for k, v in data.items() if isinstance(v, torch.Tensor)}

    def save_ckpt(self, best=False):
        if self.rank == 0:
            filename = 'model_best.pth' if best else f'model_{self.current_iters}.pth'
            ckpt_path = self.ckpt_dir / filename
            model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
            torch.save({
                'iters_start': self.current_iters,
                'state_dict': model_state,
                'best_psnr': self.best_psnr,
                'optimizer': self.optimizer.state_dict(),
            }, ckpt_path)
            if not best:
                self.logger.info(f"Saved safety checkpoint to {ckpt_path}")

    def resume_from_ckpt(self):
        self.iters_start, self.best_psnr = 0, 0.0
        if self.configs.resume:
            ckpt_path = self.configs.resume
            assert Path(ckpt_path).exists(), f"Checkpoint not found at {ckpt_path}"
            if self.rank == 0: self.logger.info(f"=> Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            
            model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
            util_net.reload_model(model_to_load, ckpt['state_dict'])
            
            self.iters_start = ckpt.get('iters_start', 0)
            self.best_psnr = ckpt.get('best_psnr', 0.0)
            if 'optimizer' in ckpt: self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.rank == 0: self.logger.info(f"=> Resumed from iteration {self.iters_start}. Previous best PSNR: {self.best_psnr:.4f}")

    def train(self):
        self.init_logger()
        self.build_model()
        self.setup_optimizaton()
        self.resume_from_ckpt()
        self.build_dataloader()
        
        if self.rank == 0 and self.iters_start == 0:
            self.baseline_visualize()
            
        self.model.train()
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1
            if hasattr(self, 'sampler') and self.sampler is not None and self.num_gpus > 1:
                self.sampler.set_epoch(ii)
            
            data = self.prepare_data(next(self.dataloaders['train']))
            # import pdb;pdb.set_trace()
            self.training_step(data)
            
            if 'val' in self.dataloaders and (self.current_iters % self.configs.train.val_freq) == 0:
                cur_psnr = self.validation()
                self.adjust_lr(cur_psnr)
                if self.rank == 0 and cur_psnr > self.best_psnr:
                    self.best_psnr = cur_psnr
                    self.logger.info(f"🎉 New best PSNR: {self.best_psnr:.4f}. Saving best model...")
                    self.save_ckpt(best=True)
            
            if (self.current_iters % self.configs.train.save_freq) == 0:
                self.save_ckpt(best=False)
                
        if self.rank == 0:
            self.plot_curves()
            self.logger.info("--- Training Finished ---")

# ==============================================================================
# 2. 继承并重写 TrainerAlphaSR
# ==============================================================================
class TrainerAlphaSR(TrainerBase):
    def __init__(self, configs):
        super().__init__(configs)
        self.local_logging = self.configs.train.get('local_logging', False)
        if self.rank == 0:
            self.log_data = {'train_loss':{'iters':[],'values':[]}, 'val_psnr':{'iters':[],'values':[]}, 'val_ssim':{'iters':[],'values':[]}}
    
    def setup_optimizaton(self):
        super().setup_optimizaton()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", 
                                                                    patience=self.configs.train.lr_schedule_params.get('patience', 5), 
                                                                    factor=self.configs.train.lr_schedule_params.get('factor', 0.5), 
                                                                    min_lr=self.configs.train.get('lr_min', 1e-6))

    def adjust_lr(self, metrics):
        self.scheduler.step(metrics)

    def training_step(self, data):
        """为 SR 任务定制的训练逻辑。"""
        inputs, gt = data['s1'], data['gt']
        predictions = self.model(inputs)
        loss = F.l1_loss(predictions, gt)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.log_step_train(loss)

    def log_step_train(self, loss):
        """记录训练 loss 用于绘图和打印。"""
        if self.rank == 0:
            self.log_data['train_loss']['iters'].append(self.current_iters)
            self.log_data['train_loss']['values'].append(loss.item())
            if self.current_iters % self.configs.train.log_freq[0] == 0:
                log_str = f"Train Iter: {self.current_iters:06d}/{self.configs.train.iterations}, Loss: {loss.item():.4e}, lr:{self.optimizer.param_groups[0]['lr']:.2e}"
                self.logger.info(log_str)

    def validation(self, phase='val'):
        """注入了所有调试、评估和可视化功能的验证逻辑。"""
        if self.rank == 0:
            self.model.eval()
            all_metrics = {'psnr':[], 'ssim':[], 'ergas':[], 'sam':[]}
            pbar = tqdm(self.dataloaders[phase], desc=f"Validation iter {self.current_iters}")
            
            for ii, data in enumerate(pbar):
                lr_path = data.get('path', ['N/A'])[0]
                data = self.prepare_data(data)
                inputs, im_gt = data['s1'], data['gt']
                
                with torch.no_grad(): predictions = self.model(inputs)
                
                if ii == 0:
                    print("\n" + "-"*80 + f"\nDEBUG STATS at Validation Iter {self.current_iters}")
                    print(f"--- Tensor Stats (Range [-1, 1]) ---")
                    print(f"  - Input (LR): Min={inputs.min():.2f}, Max={inputs.max():.2f}, Mean={inputs.mean():.2f}, Std={inputs.std():.2f}")
                    print(f"  - GT (HR):    Min={im_gt.min():.2f}, Max={im_gt.max():.2f}, Mean={im_gt.mean():.2f}, Std={im_gt.std():.2f}")
                    print(f"  - Pred (SR):  Min={predictions.min():.2f}, Max={predictions.max():.2f}, Mean={predictions.mean():.2f}, Std={predictions.std():.2f}")
                
                gt_01, pred_01 = (im_gt.clamp(-1,1)+1)/2, (predictions.clamp(-1,1)+1)/2
                gt_numpy, pred_numpy = gt_01.cpu().numpy().transpose(0,2,3,1)[0], pred_01.cpu().numpy().transpose(0,2,3,1)[0]
                psnr_val=np.mean([psnr(gt_numpy[:,:,b], pred_numpy[:,:,b], MAX=1.0) for b in range(gt_numpy.shape[-1])])
                ssim_val=np.mean([ssim(gt_numpy[:,:,b], pred_numpy[:,:,b], MAX=1.0)[0] for b in range(gt_numpy.shape[-1])])
                all_metrics['psnr'].append(psnr_val); all_metrics['ssim'].append(ssim_val)
                all_metrics['ergas'].append(ergas(gt_numpy,pred_numpy)); all_metrics['sam'].append(sam(gt_numpy,pred_numpy))
                
                if ii == 0 and self.local_logging:
                    lr_vis=((inputs.clamp(-1,1)+1)/2).cpu()
                    error_map = np.abs(pred_numpy - gt_numpy).mean(axis=-1)
                    fig,axes=plt.subplots(2,2,figsize=(14,14)); fig.suptitle(f'Validation at Iteration {self.current_iters}',fontsize=16)
                    def norm(img): return (img - img.min()) / (img.max() - img.min()) if (img.max() - img.min()) > 0 else img
                    axes[0,0].imshow(norm(lr_vis[0,self.rgb_chn,:,:].permute(1,2,0).numpy())); axes[0,0].set_title('Input (Low-Res, Interpolated)')
                    axes[0,1].imshow(norm(pred_numpy[:,:,self.rgb_chn])); axes[0,1].set_title('Prediction (Super-Resolved)')
                    axes[1,0].imshow(norm(gt_numpy[:,:,self.rgb_chn])); axes[1,0].set_title('Ground Truth (High-Res)')
                    im=axes[1,1].imshow(error_map,cmap='hot'); axes[1,1].set_title('Absolute Error Map'); fig.colorbar(im,ax=axes[1,1])
                    for ax in axes.flatten(): ax.set_xticks([]); ax.set_yticks([])
                    plt.tight_layout(rect=[0,0,1,0.96])
                    save_path=self.image_dir/phase/f"iter_{self.current_iters}_comparison_grid.png"; plt.savefig(str(save_path),dpi=150); plt.close(fig)
                    print(f"  - Comparison grid for {Path(lr_path).name} saved.\n" + "-"*80)
            
            avg_metrics={k:np.mean(v) for k,v in all_metrics.items()}
            log_str = f"Validation Metric -> " + " | ".join([f"{k.upper()}: {v:.4f}" for k, v in avg_metrics.items()])
            self.logger.info(log_str); self.logger.info("=" * 100)
            
            if self.rank == 0:
                self.log_data['val_psnr']['iters'].append(self.current_iters); self.log_data['val_psnr']['values'].append(avg_metrics['psnr'])
                self.log_data['val_ssim']['iters'].append(self.current_iters); self.log_data['val_ssim']['values'].append(avg_metrics['ssim'])
                
            self.adjust_lr(avg_metrics['psnr'])
            self.model.train()
            return avg_metrics['psnr']
    
    def baseline_visualize(self, phase='val'):
        if self.rank == 0:
            print("\n" + "="*80 + "\n[INFO] Generating baseline visualization for a fixed tile...\n" + "="*80)
            self.model.eval()
            
            target_data = None
            try:
                target_data = next(iter(self.dataloaders[phase]))
                target_file_name = Path(target_data['path'][0]).name
            except Exception as e:
                self.logger.warning(f"Error getting first validation sample for baseline: {e}. Skipping."); self.model.train(); return
            
            if target_data is None:
                self.logger.warning(f"Could not find a sample for baseline visualization. Skipping."); self.model.train(); return
            
            target_data = self.prepare_data(target_data)
            inputs, im_gt = target_data['s1'], target_data['gt']
            with torch.no_grad(): predictions = self.model(inputs)
            
            gt_01, pred_01 = (im_gt.clamp(-1,1)+1)/2, (predictions.clamp(-1,1)+1)/2
            lr_vis = ((inputs.clamp(-1,1)+1)/2).cpu()
            error_map = np.abs(pred_01[0,self.rgb_chn,:,:].cpu().permute(1,2,0).numpy() - gt_01[0,self.rgb_chn,:,:].cpu().permute(1,2,0).numpy()).mean(axis=-1)
            
            fig,axes = plt.subplots(2,2,figsize=(14,14)); fig.suptitle(f'Baseline Visualization (Before Training) for {target_file_name}',fontsize=16)
            def norm(img): return(img-img.min())/(img.max()-img.min()) if(img.max()-img.min())>0 else img
            axes[0,0].imshow(norm(lr_vis[0,self.rgb_chn,:,:].permute(1,2,0).numpy())); axes[0,0].set_title('Input')
            axes[0,1].imshow(norm(pred_01[0,self.rgb_chn,:,:].cpu().permute(1,2,0).numpy())); axes[0,1].set_title('Prediction (Untrained Model)')
            axes[1,0].imshow(norm(gt_01[0,self.rgb_chn,:,:].cpu().permute(1,2,0).numpy())); axes[1,0].set_title('Ground Truth')
            im=axes[1,1].imshow(error_map,cmap='hot'); axes[1,1].set_title('Absolute Error Map'); fig.colorbar(im,ax=axes[1,1])
            for ax in axes.flatten(): ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout(rect=[0,0,1,0.96])
            
            save_path = self.image_dir/phase/f"iter_0_baseline_comparison.png"; plt.savefig(str(save_path),dpi=150); plt.close(fig)
            self.logger.info(f"Baseline visualization saved to: {save_path}")
            
            self.model.train()

    def plot_curves(self):
        if self.rank == 0 and hasattr(self, 'log_data'):
            print("\n[INFO] Generating training curves plot...")
            fig,ax1 = plt.subplots(figsize=(12, 7))
            color='tab:red'; ax1.set_xlabel('Iterations'); ax1.set_ylabel('Training L1 Loss (Log Scale)',color=color)
            ax1.plot(self.log_data['train_loss']['iters'], self.log_data['train_loss']['values'], color=color, alpha=0.7, linewidth=2, label='Loss')
            ax1.tick_params(axis='y',labelcolor=color); ax1.grid(True,linestyle=':',alpha=0.7); ax1.set_yscale('log')
            
            if self.log_data['val_psnr']['iters']:
                ax2=ax1.twinx(); color='tab:blue'; ax2.set_ylabel('Validation PSNR (dB) / SSIM',color=color)
                ax2.plot(self.log_data['val_psnr']['iters'], self.log_data['val_psnr']['values'], color='tab:blue', marker='o', linestyle='-', markersize=5, label='PSNR (dB)')
                ax2.plot(self.log_data['val_ssim']['iters'], self.log_data['val_ssim']['values'], color='tab:green', marker='x', linestyle='--', label='SSIM')
                ax2.tick_params(axis='y',labelcolor=color)
                lines,labels=ax1.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='best')
                
            fig.suptitle('Training & Validation Curves',fontsize=16); fig.tight_layout()
            save_path=self.save_dir/"training_curves.png"; plt.savefig(str(save_path),dpi=150); plt.close(fig)
            print(f"[INFO] Training curves plot saved to: {save_path}")
