#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.post_v2.base import BaseSpectralPostprocessor


class GumbelRoutingPostProcessorV2(BaseSpectralPostprocessor):
    """Scheme 1 (v2): Gumbel routing with optional tau schedule and stats."""

    def __init__(
        self,
        n_channels=64,
        n_groups=4,
        hidden_ratio=0.5,
        tau=1.0,
        tau_start=None,
        tau_end=None,
        tau_steps=None,
    ):
        super().__init__()
        assert n_groups > 1, "n_groups must be > 1"
        self.n_channels = n_channels
        self.n_groups = n_groups
        self.tau = float(tau)
        self.tau_start = float(tau_start) if tau_start is not None else None
        self.tau_end = float(tau_end) if tau_end is not None else None
        self.tau_steps = int(tau_steps) if tau_steps is not None else None

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

    def _current_tau(self, context):
        if self.tau_start is None or self.tau_end is None or self.tau_steps is None:
            return self.tau
        it = int((context or {}).get("current_iter", 0))
        if self.tau_steps <= 0:
            return self.tau_end
        progress = max(0.0, min(1.0, float(it) / float(self.tau_steps)))
        return self.tau_start + (self.tau_end - self.tau_start) * progress

    def _forward_impl(self, x, context):
        b, c, _, _ = x.shape
        logits = self.router(x).view(b, c, self.n_groups)

        tau = self._current_tau(context)
        routing_mask = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

        out = torch.zeros_like(x)
        for k in range(self.n_groups):
            mask_k = routing_mask[:, :, k].unsqueeze(-1).unsqueeze(-1)
            routed = x * mask_k
            out = out + self.group_processors[k](routed)

        # Stats for training diagnostics.
        with torch.no_grad():
            probs = routing_mask.float().mean(dim=(0, 1))
            probs = probs / probs.sum().clamp(min=1e-8)
            entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum().item()
            active_groups = float((probs > 0.01).sum().item())
            self._last_stats = {
                "post_tau": float(tau),
                "routing_entropy": float(entropy),
                "active_groups": active_groups,
            }
            self._last_extra_losses = {}

        return x + out
