import torch

from trainer import TrainerAlphaSR


class TrainerAlphaSRMaskAblation(TrainerAlphaSR):
    """独立试验版 Trainer：
    - 不修改原 trainer.py
    - 仅替换 indicating_mask 的时间维聚合策略
    - 默认使用 prob_or（1 - Π(1-m)），避免简单 mean 抹平时相差异
    """

    def _to_bthw(self, indicating_mask, batch_size):
        """将 indicating_mask 统一为 (B, T, H, W) 或 (B, 1, H, W)。"""
        if indicating_mask is None:
            return None

        mask = indicating_mask

        # 常见 collate 形状：
        # (B, 1, T, H, W) / (B, T, 1, H, W) / (B, T, H, W) / (B, H, W)
        if mask.ndim == 5:
            if mask.shape[1] == 1:
                mask = mask.squeeze(1)  # (B, T, H, W)
            elif mask.shape[2] == 1:
                mask = mask.squeeze(2)  # (B, T, H, W)
            else:
                raise ValueError(f"Unsupported indicating_mask shape: {tuple(mask.shape)}")

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)

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
        """将 (B,T,H,W) 聚合到 (B,1,H,W)。"""
        # 可在配置中切换：mean / max / prob_or
        strategy = self.configs.train.get('indicating_mask_reduce', 'prob_or')
        mask_bt_hw = mask_bt_hw.clamp(0.0, 1.0)

        if mask_bt_hw.shape[1] == 1:
            return mask_bt_hw

        if strategy == 'mean':
            return mask_bt_hw.mean(dim=1, keepdim=True)
        if strategy == 'max':
            return mask_bt_hw.max(dim=1, keepdim=True).values
        if strategy == 'prob_or':
            # 任一时相可见则提高监督权重：1 - Π(1-m_t)
            return 1.0 - torch.prod(1.0 - mask_bt_hw, dim=1, keepdim=True)

        raise ValueError(f"Unknown indicating_mask_reduce strategy: {strategy}")

    def training_step(self, data):
        predictions = self.model(data)  # (B, C, H, W)
        gt = data['gt']
        indicating_mask = data.get('indicating_mask', None)

        if indicating_mask is not None:
            mask_bt_hw = self._to_bthw(indicating_mask, batch_size=predictions.shape[0])
            mask_b1hw = self._reduce_temporal_mask(mask_bt_hw)  # (B,1,H,W)
            mask_bchw = mask_b1hw.expand(-1, predictions.shape[1], -1, -1)

            masked_predictions = predictions * mask_bchw
            masked_gt = gt * mask_bchw
            loss = self.criterion(masked_predictions, masked_gt)

            valid_pixels = mask_bchw.sum()
            if valid_pixels > 0:
                loss = loss * (predictions.numel() / valid_pixels)
        else:
            loss = self.criterion(predictions, gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_step_train(loss)
