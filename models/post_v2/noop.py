#!/usr/bin/env python
# -*- coding:utf-8 -*-

from models.post_v2.base import BaseSpectralPostprocessor


class NoOpPostProcessor(BaseSpectralPostprocessor):
    def __init__(self, n_channels=64):
        super().__init__()
        self.n_channels = n_channels

    def _forward_impl(self, x, context):
        self._last_stats = {"post_tau": 0.0, "routing_entropy": 0.0, "active_groups": 0.0}
        self._last_extra_losses = {}
        return x
