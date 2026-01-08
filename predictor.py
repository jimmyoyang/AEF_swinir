# 文件路径: predictor.py
# (v_final_unified - 与 main.py v2 完全兼容的最终版)

import torch, tqdm, pathlib, numpy as np, PIL.Image, math, torch.nn.functional as F, yaml, argparse
from sewar.full_ref import psnr, ssim, ergas, sam
from utils.util_common import get_obj_from_str

class Predictor:
    def __init__(self, configs, ckpt_path):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("[Predictor INFO] Building and loading model...")
        self.model = get_obj_from_str(self.configs.model['target'])(**self.configs.model['params']).to(self.device)
        
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[Predictor INFO] Model loaded from: {ckpt_path}")
        
        if hasattr(self.configs, 'inference'): self.padding_offset = self.configs.inference.get('padding_offset', 16)
        else: self.padding_offset = 16

    def predict_step(self, x):
        offset, (ori_h, ori_w) = self.padding_offset, x.shape[2:]
        if not (ori_h%offset==0 and ori_w%offset==0):
            flag_pad = True; pad_h=(math.ceil(ori_h/offset))*offset-ori_h; pad_w=(math.ceil(ori_w/offset))*offset-ori_w; x=F.pad(x,pad=(0,pad_w,0,pad_h),mode='reflect')
        else: flag_pad = False
        with torch.no_grad(): result = self.model(x)
        if flag_pad: result = result[:, :, :ori_h, :ori_w]
        return result.clamp_(-1.0, 1.0)

    def run_inference(self, test_loader, out_dir):
        out_path, results_path = pathlib.Path(out_dir), pathlib.Path(out_dir) / 'results'
        results_path.mkdir(parents=True, exist_ok=True)
        all_metrics, has_gt = {'psnr':[], 'ssim':[], 'ergas':[], 'sam':[]}, False
        
        pbar = tqdm.tqdm(test_loader, desc=f'Inference on {out_dir.name}')
        for i, data in enumerate(pbar):
            lr_path = data['path'][0]
            inputs, gt = data['s1'].to(self.device), data.get('gt', None)
            
            if i < 5:
                print(f"\n--- CHECKPOINT: Stats for sample {i} ({pathlib.Path(lr_path).name}) ---")
                print(f"  - Input (LR, range -1 to 1): Min={inputs.min():.3f}, Max={inputs.max():.3f}, Mean={inputs.mean():.3f}")
            
            predictions = self.predict_step(inputs)
            
            if i < 5:
                print(f"  - Pred (SR, range -1 to 1):  Min={predictions.min():.3f}, Max={predictions.max():.3f}, Mean={predictions.mean():.3f}")
                print("-" * 50)
                
            pred_01 = (predictions.clamp(-1,1) + 1) / 2
            pred_numpy = pred_01.cpu().numpy().transpose(0, 2, 3, 1)[0]
            pred_to_save = (pred_numpy[:, :, self.configs.train.rgb_chn] * 255).astype(np.uint8)
            
            filename = pathlib.Path(lr_path).name.replace('.tif', '.png')
            PIL.Image.fromarray(pred_to_save).save(results_path / filename)
            
            if gt is not None:
                has_gt = True; gt_01 = (gt.clamp(-1,1) + 1) / 2
                gt_numpy = gt_01.cpu().numpy().transpose(0, 2, 3, 1)[0]
                
                psnr_val = np.mean([psnr(gt_numpy[:,:,b], pred_numpy[:,:,b], MAX=1.0) for b in range(gt_numpy.shape[-1])])
                ssim_val = np.mean([ssim(gt_numpy[:,:,b], pred_numpy[:,:,b], MAX=1.0)[0] for b in range(gt_numpy.shape[-1])])
                
                all_metrics['psnr'].append(psnr_val); all_metrics['ssim'].append(ssim_val)
                all_metrics['ergas'].append(ergas(gt_numpy, pred_numpy)); all_metrics['sam'].append(sam(gt_numpy, pred_numpy))
                
        print(f"\n[Predictor INFO] Inference complete. Results saved to: {results_path}")
        if has_gt:
            avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
            print(f"  [Evaluation INFO] Average Metrics -> " + " | ".join([f"{k.upper()}: {v:.4f}" for k, v in avg_metrics.items()]))
            with open(out_path/'inference_metrics.txt', 'w') as f:
                f.write("Average Metrics:\n"); [f.write(f"  - {k.upper()}: {v:.4f}\n") for k,v in avg_metrics.items()]
            print(f"  [Evaluation INFO] Metrics saved to: {out_path/'inference_metrics.txt'}")

