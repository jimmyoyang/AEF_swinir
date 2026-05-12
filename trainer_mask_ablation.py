import torch
import torch.nn.functional as F

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

        # 可选：先把软权重阈值化为硬掩膜，从而实现“云像素完全剔除、非加权”。
        # 说明：先二值化再 prob_or/max/mean，能保证 prob_or 输出也为 {0,1}。
        if bool(self.configs.train.get('indicating_mask_binarize', False)):
            thr = float(self.configs.train.get('indicating_mask_threshold', 0.5))
            mask_bt_hw = (mask_bt_hw >= thr).float()

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

            # Dataset mask is LR-space (e.g., 64x64) while model output is SR-space (e.g., 192x192).
            # Resize mask to prediction resolution before channel expansion.
            if mask_b1hw.shape[-2:] != predictions.shape[-2:]:
                mask_b1hw = F.interpolate(
                    mask_b1hw,
                    size=predictions.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )

            mask_bchw = mask_b1hw.expand(-1, predictions.shape[1], -1, -1)
            if (
                self.rank == 0
                and int(self.configs.train.get('debug_batch_log_freq', 0) or 0) > 0
                and (self.current_iters <= 5 or self.current_iters % int(self.configs.train.get('debug_batch_log_freq', 1)) == 0)
            ):
                self.logger.info(
                    f"🔎 train mask debug iter={self.current_iters} "
                    f"reduce={self.configs.train.get('indicating_mask_reduce', 'prob_or')} "
                    f"mask_mean={mask_b1hw.detach().float().mean().item():.4f} "
                    f"mask_gt_0.5={(mask_b1hw.detach().float() > 0.5).float().mean().item():.4f}"
                )

            valid_pixels = mask_bchw.sum()
            if valid_pixels <= 0:
                if self.rank == 0:
                    self.logger.warning(
                        f"Iter {self.current_iters}: indicating_mask is all-zero after resize; skip step (would risk NaN in SSIM/MSE)."
                    )
                return

            masked_predictions = predictions * mask_bchw
            masked_gt = gt * mask_bchw
            loss = self.criterion(masked_predictions, masked_gt)
            loss = loss * (predictions.numel() / valid_pixels)
        else:
            loss = self.criterion(predictions, gt)

        self._log_batch_debug(data, predictions=predictions, loss=loss, phase='train')

        if not torch.isfinite(loss).all():
            if self.rank == 0:
                self.logger.warning(
                    f"Iter {self.current_iters}: non-finite loss={loss.detach()}; skip backward. Check LR/GT for NaN (run test_lr_tile_values.py)."
                )
            return

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_step_train(loss)


class TrainerAlphaSRMaskTemporalLossAblation(TrainerAlphaSRMaskAblation):
    """独立试验版 Trainer（时相级损失）：
    - 不修改原 trainer.py
    - 不将 indicating_mask 先压缩到单张空间掩膜
    - 逐时相计算 masked loss，再按策略聚合
    """

    def _compute_single_masked_loss(self, predictions, gt, mask_b1hw):
        """在单个空间掩膜下计算一次归一化损失。"""
        # Dataset mask is LR-space while model output is SR-space.
        if mask_b1hw.shape[-2:] != predictions.shape[-2:]:
            mask_b1hw = F.interpolate(
                mask_b1hw,
                size=predictions.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        mask_bchw = mask_b1hw.expand(-1, predictions.shape[1], -1, -1)
        valid_pixels = mask_bchw.sum()
        if valid_pixels <= 0:
            return predictions.new_tensor(0.0)

        masked_predictions = predictions * mask_bchw
        masked_gt = gt * mask_bchw
        loss = self.criterion(masked_predictions, masked_gt) * (predictions.numel() / valid_pixels)
        return loss

    def training_step(self, data):
        predictions = self.model(data)  # (B, C, H, W)
        gt = data['gt']
        indicating_mask = data.get('indicating_mask', None)

        if indicating_mask is None:
            loss = self.criterion(predictions, gt)
        else:
            mask_bt_hw = self._to_bthw(indicating_mask, batch_size=predictions.shape[0])  # (B,T,H,W)
            mask_bt_hw = mask_bt_hw.clamp(0.0, 1.0)

            if bool(self.configs.train.get('indicating_mask_binarize', False)):
                thr = float(self.configs.train.get('indicating_mask_threshold', 0.5))
                mask_bt_hw = (mask_bt_hw >= thr).float()

            temporal_validity = data.get('mask', None)  # (B,T) or (T,)
            if temporal_validity is not None:
                if temporal_validity.ndim == 1:
                    temporal_validity = temporal_validity.unsqueeze(0)
                if temporal_validity.shape[0] != predictions.shape[0]:
                    if temporal_validity.shape[0] == 1:
                        temporal_validity = temporal_validity.expand(predictions.shape[0], -1)
                    else:
                        raise ValueError(
                            f"Batch mismatch between temporal mask ({temporal_validity.shape[0]}) "
                            f"and predictions ({predictions.shape[0]})"
                        )

            loss_list = []
            weight_list = []
            B, T, _, _ = mask_bt_hw.shape
            for t in range(T):
                mask_b1hw = mask_bt_hw[:, t:t + 1, :, :]  # (B,1,H,W)
                loss_t = self._compute_single_masked_loss(predictions, gt, mask_b1hw)

                if temporal_validity is not None:
                    # 对 batch 内各样本做平均，作为该时相的聚合权重
                    w_t = temporal_validity[:, t].float().mean().detach()
                else:
                    w_t = torch.tensor(1.0, device=predictions.device)

                loss_list.append(loss_t)
                weight_list.append(w_t)

            temporal_loss_reduce = self.configs.train.get('temporal_loss_reduce', 'weighted_mean')
            weights = torch.stack(weight_list).clamp(min=0.0)
            losses = torch.stack(loss_list)

            if temporal_loss_reduce == 'mean':
                loss = losses.mean()
            elif temporal_loss_reduce == 'max':
                loss = losses.max()
            elif temporal_loss_reduce == 'weighted_mean':
                denom = weights.sum().clamp(min=1e-6)
                loss = (losses * weights).sum() / denom
            else:
                raise ValueError(f"Unknown temporal_loss_reduce strategy: {temporal_loss_reduce}")

        self._log_batch_debug(data, predictions=predictions, loss=loss, phase='train')

        if not torch.isfinite(loss).all():
            if self.rank == 0:
                self.logger.warning(
                    f"Iter {self.current_iters}: non-finite temporal loss; skip backward. Check data / masks."
                )
            return

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_step_train(loss)
