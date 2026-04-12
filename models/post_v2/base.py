#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch.nn as nn


class BaseSpectralPostprocessor(nn.Module):
    """Base class for v2 spectral postprocessors.

    Implement `_forward_impl` in subclasses. Runtime context is an optional dict,
    e.g. {"current_iter": 10, "max_iters": 5000, "stage": "train"}.
    """

    def __init__(self):
        super().__init__()
        self._runtime_context = {}
        self._last_stats = {}
        self._last_extra_losses = {}

    def set_runtime_context(self, context):
        self._runtime_context = context or {}

    def get_last_stats(self):
        return dict(self._last_stats)

    def get_last_extra_losses(self):
        return dict(self._last_extra_losses)

    def _forward_impl(self, x, context):
        raise NotImplementedError

    def forward(self, x):
        out = self._forward_impl(x, self._runtime_context)
        return out
