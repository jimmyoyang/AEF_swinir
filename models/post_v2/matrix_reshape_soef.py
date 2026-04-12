#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn

from models.post_v2.base import BaseSpectralPostprocessor


class MatrixReshapeSOEFPostProcessorV2(BaseSpectralPostprocessor):
    """Scheme 6 (v2): matrix reshape + lightweight multi-scale pseudo-2D fusion."""

    def __init__(self, n_channels=64, hidden_channels=12):
        super().__init__()
        grid = int(math.sqrt(n_channels))
        assert grid * grid == n_channels, "n_channels must be a perfect square"
        self.n_channels = n_channels
        self.grid = grid

        self.conv3 = nn.Conv2d(1, hidden_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(1, hidden_channels, 5, padding=2)
        self.fuse = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, 1, 1),
        )

    def _forward_impl(self, x, context):
        b, c, h, w = x.shape
        x_grid = x.permute(0, 2, 3, 1).contiguous().view(b * h * w, 1, self.grid, self.grid)
        x3 = self.conv3(x_grid)
        x5 = self.conv5(x_grid)
        y = self.fuse(torch.cat([x3, x5], dim=1))

        x_out = y.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        self._last_stats = {"post_tau": 0.0, "routing_entropy": 0.0, "active_groups": 1.0}
        self._last_extra_losses = {}
        return x + x_out
