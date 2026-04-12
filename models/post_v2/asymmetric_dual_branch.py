#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from models.post_v2.base import BaseSpectralPostprocessor


class AsymmetricDualBranchPostProcessorV2(BaseSpectralPostprocessor):
    """Scheme 3 (v2): hard split with optional variance-based dynamic split."""

    def __init__(self, n_channels=64, split_ratio=0.5, split_mode="fixed"):
        super().__init__()
        assert split_mode in ["fixed", "variance_dynamic"]
        self.n_channels = n_channels
        self.split_mode = split_mode

        high_c = int(n_channels * split_ratio)
        low_c = n_channels - high_c
        assert high_c > 0 and low_c > 0, "Invalid split_ratio"

        self.high_c = high_c
        self.low_c = low_c

        self.high_branch = nn.Sequential(
            nn.Conv2d(high_c, high_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_c, high_c, 3, padding=1),
        )

        reduction = max(4, low_c // 4)
        self.low_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(low_c, reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, low_c, 1),
            nn.Sigmoid(),
        )

    def _split(self, x):
        if self.split_mode == "fixed":
            return x[:, : self.high_c], x[:, self.high_c :]

        # variance_dynamic: per-batch channel variance sorting.
        b, c, h, w = x.shape
        var = x.view(b, c, -1).var(dim=2).mean(dim=0)
        _, idx = torch.sort(var, descending=True)
        hi_idx = idx[: self.high_c]
        lo_idx = idx[self.high_c :]
        x_high = x[:, hi_idx, :, :]
        x_low = x[:, lo_idx, :, :]
        return x_high, x_low

    def _forward_impl(self, x, context):
        x_high, x_low = self._split(x)

        high_refined = x_high + self.high_branch(x_high)
        low_weight = self.low_se(x_low)
        low_refined = x_low * low_weight

        out = torch.cat([high_refined, low_refined], dim=1)
        self._last_stats = {"post_tau": 0.0, "routing_entropy": 0.0, "active_groups": 2.0}
        self._last_extra_losses = {}
        return x + out
