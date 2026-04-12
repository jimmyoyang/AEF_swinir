#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch.nn as nn


class ContextAwarePostprocessorWrapper(nn.Module):
    """Adapter so legacy network forward(x) can drive context-aware processors."""

    def __init__(self, postprocessor):
        super().__init__()
        self.postprocessor = postprocessor

    def set_runtime_context(self, context):
        if hasattr(self.postprocessor, "set_runtime_context"):
            self.postprocessor.set_runtime_context(context)

    def get_last_stats(self):
        if hasattr(self.postprocessor, "get_last_stats"):
            return self.postprocessor.get_last_stats()
        return {}

    def get_last_extra_losses(self):
        if hasattr(self.postprocessor, "get_last_extra_losses"):
            return self.postprocessor.get_last_extra_losses()
        return {}

    def forward(self, x):
        return self.postprocessor(x)
