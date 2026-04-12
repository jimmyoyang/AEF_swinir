#!/usr/bin/env python
# -*- coding:utf-8 -*-

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sewar.full_ref import psnr, ssim, ergas, sam
from tqdm import tqdm

from trainer import TrainerAlphaSR


class TrainerAlphaSRPostV2(TrainerAlphaSR):
    """Parallel v2 trainer with context injection + postprocessor stat logging."""

    def __init__(self, configs):
        super().__init__(configs)
        self._post_stats_path = Path(self.save_dir) / "post_v2_stats.csv"
        self._post_stats_header_written = False

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _set_post_context(self, stage, batch_size=0):
        model = self._unwrap_model()
        if not hasattr(model, "set_runtime_context"):
            return
        context = {
            "stage": stage,
            "current_iter": int(getattr(self, "current_iters", 0)),
            "max_iters": int(self.configs.train.iterations),
            "batch_size": int(batch_size),
        }
        model.set_runtime_context(context)

    def _collect_post_stats(self, stage):
        if self.rank != 0:
            return
        model = self._unwrap_model()
        if not hasattr(model, "get_postprocessor_stats"):
            return
        stats = model.get_postprocessor_stats() or {}
        if not stats:
            return

        row = {
            "iter": int(getattr(self, "current_iters", 0)),
            "stage": stage,
            "post_tau": float(stats.get("post_tau", 0.0)),
            "routing_entropy": float(stats.get("routing_entropy", 0.0)),
            "active_groups": float(stats.get("active_groups", 0.0)),
        }

        write_header = (not self._post_stats_header_written) and (not self._post_stats_path.exists())
        with open(self._post_stats_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._post_stats_header_written = True
            writer.writerow(row)

    def training_step(self, data):
        self._set_post_context(stage="train", batch_size=data["gt"].shape[0])

        predictions = self.model(data)
        use_ind_mask = self.configs.train.get("use_indicating_mask_in_training", False)
        if use_ind_mask and data.get("indicating_mask") is not None:
            ind_mask = data["indicating_mask"].float()
            if ind_mask.ndim == 3:
                ind_mask = ind_mask.unsqueeze(0)
            spatial_mask = ind_mask.max(dim=1, keepdim=True)[0].clamp(0, 1)
            if spatial_mask.shape[-2:] != predictions.shape[-2:]:
                spatial_mask = F.interpolate(
                    spatial_mask,
                    size=predictions.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            spatial_mask = spatial_mask.expand_as(predictions)
            valid_pixels = spatial_mask.sum().clamp(min=1)
            loss = self.criterion(predictions * spatial_mask, data["gt"] * spatial_mask)
            loss = loss * (float(predictions.numel()) / valid_pixels)
        else:
            loss = self.criterion(predictions, data["gt"])

        # Optional extra losses from v2 postprocessor.
        model = self._unwrap_model()
        if hasattr(model, "get_postprocessor_extra_losses"):
            extra_losses = model.get_postprocessor_extra_losses() or {}
            for _, loss_tensor in extra_losses.items():
                loss = loss + loss_tensor

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_step_train(loss)
        self._collect_post_stats(stage="train")

    @torch.no_grad()
    def validation(self, phase="val"):
        if self.rank == 0:
            self.model.eval()
            all_metrics = {"psnr": [], "ssim": [], "ergas": [], "sam": [], "masked_psnr": []}
            pbar = tqdm(self.dataloaders[phase], desc=f"[V2] Val Iter {getattr(self, 'current_iters', 0)}")

            for ii, data in enumerate(pbar):
                data = self.prepare_data(data)
                self._set_post_context(stage="val", batch_size=data["gt"].shape[0])
                predictions = self.model(data)

                gt_01 = self.norm_for_vis(data["gt"])
                pred_01 = self.norm_for_vis(predictions)

                gt_numpy = gt_01.transpose(0, 2, 3, 1)[0]
                pred_numpy = pred_01.transpose(0, 2, 3, 1)[0]

                psnr_val = np.mean([psnr(gt_numpy[:, :, b], pred_numpy[:, :, b], MAX=1.0) for b in range(gt_numpy.shape[-1])])
                ssim_val = np.mean([ssim(gt_numpy[:, :, b], pred_numpy[:, :, b], MAX=1.0)[0] for b in range(gt_numpy.shape[-1])])

                all_metrics["psnr"].append(psnr_val)
                all_metrics["ssim"].append(ssim_val)
                all_metrics["ergas"].append(ergas(gt_numpy, pred_numpy))
                all_metrics["sam"].append(sam(gt_numpy, pred_numpy))

                if data.get("indicating_mask") is not None:
                    ind_mask = data["indicating_mask"].float()
                    if ind_mask.ndim == 3:
                        ind_mask = ind_mask.unsqueeze(0)
                    spatial_mask = ind_mask[0].max(dim=0, keepdim=True)[0].unsqueeze(0)
                    pred_h, pred_w = pred_numpy.shape[0], pred_numpy.shape[1]
                    if spatial_mask.shape[-2:] != (pred_h, pred_w):
                        spatial_mask = F.interpolate(
                            spatial_mask,
                            size=(pred_h, pred_w),
                            mode="bilinear",
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
                            all_metrics["masked_psnr"].append(float(np.mean(per_band_mpsnr)))

                if ii == 0 and self.configs.train.get("local_logging", False):
                    self.visualize_validation_sample(data, predictions, ii)

            avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
            masked_psnr_str = (
                f" | Masked-PSNR: {avg_metrics['masked_psnr']:.4f}"
                if "masked_psnr" in avg_metrics else ""
            )
            self.logger.info(
                f"[V2] Validation Metrics | "
                f"PSNR: {avg_metrics['psnr']:.4f} | "
                f"SSIM: {avg_metrics['ssim']:.4f} | "
                f"ERGAS: {avg_metrics['ergas']:.4f} | "
                f"SAM: {avg_metrics['sam']:.4f}"
                + masked_psnr_str
            )

            self.log_data["val_psnr"]["iters"].append(self.current_iters)
            self.log_data["val_psnr"]["values"].append(avg_metrics["psnr"])
            self.log_data["val_ssim"]["iters"].append(self.current_iters)
            self.log_data["val_ssim"]["values"].append(avg_metrics["ssim"])

            self._collect_post_stats(stage="val")
            self.model.train()
            return avg_metrics["psnr"]
