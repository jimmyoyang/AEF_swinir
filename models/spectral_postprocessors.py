#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelRoutingPostProcessor(nn.Module):
    """
    Scheme 1: Gumbel-Softmax dynamic channel routing.

    Inspired by differentiable routing ideas:
    - Jang et al., 2016 (Gumbel-Softmax)
    - Bengio et al., 2014 (stochastic neurons for conditional computation)
    """

    def __init__(self, n_channels=64, n_groups=4, hidden_ratio=0.5, tau=1.0):
        super().__init__()
        assert n_groups > 1, "n_groups must be > 1"
        self.n_channels = n_channels
        self.n_groups = n_groups
        self.tau = tau

        hidden_dim = max(8, int(n_channels * hidden_ratio))
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, n_channels * n_groups, 1),
        )

        self.group_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_channels, n_channels, 3, padding=1, groups=n_channels),
                nn.Conv2d(n_channels, n_channels, 1),
            )
            for _ in range(n_groups)
        ])

    def forward(self, x):
        b, c, _, _ = x.shape
        logits = self.router(x).view(b, c, self.n_groups)

        routing_mask = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)

        out = torch.zeros_like(x)
        for k in range(self.n_groups):
            mask_k = routing_mask[:, :, k].unsqueeze(-1).unsqueeze(-1)
            routed = x * mask_k
            out = out + self.group_processors[k](routed)

        return x + out


class AsymmetricDualBranchPostProcessor(nn.Module):
    """
    Scheme 3: Asymmetric dual-branch refinement.

    Inspired by frequency-aware split processing in lightweight SR designs.
    """

    def __init__(self, n_channels=64, split_ratio=0.5):
        super().__init__()
        high_c = int(n_channels * split_ratio)
        low_c = n_channels - high_c
        assert high_c > 0 and low_c > 0, "Invalid split ratio for channels"

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

    def forward(self, x):
        x_high = x[:, : self.high_c, :, :]
        x_low = x[:, self.high_c :, :, :]

        high_refined = x_high + self.high_branch(x_high)
        low_weight = self.low_se(x_low)
        low_refined = x_low * low_weight

        out = torch.cat([high_refined, low_refined], dim=1)
        return x + out


class MatrixReshapeSOEFPostProcessor(nn.Module):
    """
    Scheme 6: Matrix reshape + pseudo-2D multi-scale fusion.

    Treat channel dimension as a square grid (e.g., 64 -> 8x8), then apply
    lightweight 2D convolutions on the pseudo-grid.
    """

    def __init__(self, n_channels=64):
        super().__init__()
        grid = int(math.sqrt(n_channels))
        assert grid * grid == n_channels, "n_channels must be a perfect square"
        self.n_channels = n_channels
        self.grid = grid

        self.fuse = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x_grid = x.permute(0, 2, 3, 1).contiguous().view(b * h * w, 1, self.grid, self.grid)
        x_grid = self.fuse(x_grid)

        x_out = x_grid.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x + x_out


def build_spectral_postprocessor(pp_type, pp_params):
    pp_type = (pp_type or "").lower()
    pp_params = pp_params or {}

    if pp_type in ["scheme1", "gumbel", "gumbel_routing"]:
        return GumbelRoutingPostProcessor(**pp_params)
    if pp_type in ["scheme3", "adbr", "asymmetric_dual_branch"]:
        return AsymmetricDualBranchPostProcessor(**pp_params)
    if pp_type in ["scheme6", "soef", "matrix_reshape"]:
        return MatrixReshapeSOEFPostProcessor(**pp_params)

    raise ValueError(f"Unknown spectral postprocessor type: {pp_type}")
