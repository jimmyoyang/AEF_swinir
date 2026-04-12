#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F

from trainer_post_v2 import TrainerAlphaSRPostV2


class TrainerAlphaSRPostV2MaskAblation(TrainerAlphaSRPostV2):
    """V2 trainer with mask-reduce ablation support (mean/max/prob_or).

    This keeps post_v2 context injection/stat logging, while adopting
    the robust indicating_mask reduction strategy used in mask ablations.
    """

    def _to_bthw(self, indicating_mask, batch_size):
        if indicating_mask is None:
            return None

        mask = indicating_mask
        # (B,1,T,H,W) / (B,T,1,H,W) -> (B,T,H,W)
        if mask.ndim == 5:
            if mask.shape[1] == 1:
                mask = mask.squeeze(1)
            elif mask.shape[2] == 1:
                mask = mask.squeeze(2)
            else:
                raise ValueError(f"Unsupported indicating_mask shape: {tuple(mask.shape)}")

        # (B,H,W) -> (B,1,H,W)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        if mask.ndim != 4:
            raise ValueError(f"indicating_mask should be 4D after normalization, got {tuple(mask.shape)}")

        if mask.shape[0] != batch_size:
            if mask.shape[0] == 1:
                mask = mask.expand(batch_size, -1, -1, -1)
            else:
                raise ValueError(
                    f"Batch mismatch between indicating_mask ({mask.shape[0]}) and predictions ({batch_size})"
                )

        return mask

    def _reduce_temporal_mask(self, mask_bt_hw):
        strategy = self.configs.train.get("indicating_mask_reduce", "prob_or")
        mask_bt_hw = mask_bt_hw.clamp(0.0, 1.0)

        if mask_bt_hw.shape[1] == 1:
            return mask_bt_hw

        if strategy == "mean":
            return mask_bt_hw.mean(dim=1, keepdim=True)
        if strategy == "max":
            return mask_bt_hw.max(dim=1, keepdim=True).values
        if strategy == "prob_or":
            return 1.0 - torch.prod(1.0 - mask_bt_hw, dim=1, keepdim=True)

        raise ValueError(f"Unknown indicating_mask_reduce strategy: {strategy}")

    def training_step(self, data):
        self._set_post_context(stage="train", batch_size=data["gt"].shape[0])

        predictions = self.model(data)
        gt = data["gt"]
        indicating_mask = data.get("indicating_mask", None)

        use_ind_mask = self.configs.train.get("use_indicating_mask_in_training", True)
        if use_ind_mask and indicating_mask is not None:
            mask_bt_hw = self._to_bthw(indicating_mask, batch_size=predictions.shape[0])
            mask_b1hw = self._reduce_temporal_mask(mask_bt_hw)

            # Align LR mask to SR output size.
            if mask_b1hw.shape[-2:] != predictions.shape[-2:]:
                mask_b1hw = F.interpolate(
                    mask_b1hw,
                    size=predictions.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            mask_bchw = mask_b1hw.expand(-1, predictions.shape[1], -1, -1)
            masked_predictions = predictions * mask_bchw
            masked_gt = gt * mask_bchw
            loss = self.criterion(masked_predictions, masked_gt)

            valid_pixels = mask_bchw.sum().clamp(min=1.0)
            loss = loss * (float(predictions.numel()) / valid_pixels)
        else:
            loss = self.criterion(predictions, gt)

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
