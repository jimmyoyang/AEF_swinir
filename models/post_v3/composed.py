#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from models.post_v2.base import BaseSpectralPostprocessor


class SequentialComposedPostProcessorV3(BaseSpectralPostprocessor):
    """Compose two postprocessors with optional staged and gated fusion.

    Default behavior remains the previous fixed residual sum:
    output = x + r1 + r2, where r1 = first(x) - x, r2 = second(x) - x.
    """

    def __init__(
        self,
        first_module,
        second_module,
        use_learnable_gates=False,
        gate_init_alpha=0.5,
        gate_init_beta=0.5,
        fixed_alpha=1.0,
        fixed_beta=1.0,
        staged_training=None,
    ):
        super().__init__()
        self.first = first_module
        self.second = second_module

        self.use_learnable_gates = bool(use_learnable_gates)
        self.fixed_alpha = float(fixed_alpha)
        self.fixed_beta = float(fixed_beta)
        self.staged_training = dict(staged_training or {})

        if self.use_learnable_gates:
            self.alpha_logit = nn.Parameter(self._inv_sigmoid(gate_init_alpha))
            self.beta_logit = nn.Parameter(self._inv_sigmoid(gate_init_beta))
        else:
            self.register_parameter("alpha_logit", None)
            self.register_parameter("beta_logit", None)

    @staticmethod
    def _inv_sigmoid(p):
        p = float(max(1e-4, min(1.0 - 1e-4, p)))
        return torch.tensor(torch.log(torch.tensor(p / (1.0 - p))), dtype=torch.float32)

    def _get_base_gates(self, device):
        if self.use_learnable_gates:
            alpha = torch.sigmoid(self.alpha_logit)
            beta = torch.sigmoid(self.beta_logit)
            return alpha, beta
        alpha = torch.tensor(self.fixed_alpha, dtype=torch.float32, device=device)
        beta = torch.tensor(self.fixed_beta, dtype=torch.float32, device=device)
        return alpha, beta

    def _get_stage_scales(self, context):
        cfg = self.staged_training
        if not cfg or not bool(cfg.get("enabled", False)):
            return 1.0, 1.0, 2.0

        it = int((context or {}).get("current_iter", 0))
        warmup = int(cfg.get("warmup_steps", 0))
        ramp = int(cfg.get("ramp_steps", 0))

        first_w = float(cfg.get("first_scale_warmup", 1.0))
        second_w = float(cfg.get("second_scale_warmup", 0.0))
        first_f = float(cfg.get("first_scale_final", 1.0))
        second_f = float(cfg.get("second_scale_final", 1.0))

        if it < warmup:
            return first_w, second_w, 0.0

        if ramp <= 0:
            return first_f, second_f, 2.0

        progress = max(0.0, min(1.0, float(it - warmup) / float(ramp)))
        first_s = first_w + (first_f - first_w) * progress
        second_s = second_w + (second_f - second_w) * progress
        phase = 1.0 if progress < 1.0 else 2.0
        return first_s, second_s, phase

    def set_runtime_context(self, context):
        super().set_runtime_context(context)
        if hasattr(self.first, "set_runtime_context"):
            self.first.set_runtime_context(context)
        if hasattr(self.second, "set_runtime_context"):
            self.second.set_runtime_context(context)

    def _forward_impl(self, x, context):
        y = self.first(x)
        z = self.second(x)

        r1 = y - x
        r2 = z - x

        alpha, beta = self._get_base_gates(x.device)
        s1, s2, stage_phase = self._get_stage_scales(context)

        alpha_eff = alpha * s1
        beta_eff = beta * s2
        output = x + alpha_eff * r1 + beta_eff * r2

        first_stats = self.first.get_last_stats() if hasattr(self.first, "get_last_stats") else {}
        second_stats = self.second.get_last_stats() if hasattr(self.second, "get_last_stats") else {}

        # Keep default keys for trainer csv compatibility.
        with torch.no_grad():
            r1_norm = float(r1.detach().pow(2).mean().sqrt().item())
            r2_norm = float(r2.detach().pow(2).mean().sqrt().item())

        self._last_stats = {
            "post_tau": float(first_stats.get("post_tau", 0.0)),
            "routing_entropy": float(first_stats.get("routing_entropy", 0.0)),
            "active_groups": float(first_stats.get("active_groups", 0.0)),
            "second_active_groups": float(second_stats.get("active_groups", 0.0)),
            "gate_alpha": float(alpha.detach().item()),
            "gate_beta": float(beta.detach().item()),
            "stage_first_scale": float(s1),
            "stage_second_scale": float(s2),
            "stage_phase": float(stage_phase),
            "residual_first_l2": float(r1_norm),
            "residual_second_l2": float(r2_norm),
        }

        extra = {}
        if hasattr(self.first, "get_last_extra_losses"):
            extra.update(self.first.get_last_extra_losses() or {})
        if hasattr(self.second, "get_last_extra_losses"):
            extra.update(self.second.get_last_extra_losses() or {})
        self._last_extra_losses = extra

        return output
