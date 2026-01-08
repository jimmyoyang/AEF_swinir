import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .basic_ops import (
    conv_nd
)

class SampleNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        
        self.end2end = nn.Sequential(
            conv_nd(2, self.in_channels, self.model_channels, 3, padding=1), #二维卷积通用框架
            nn.ReLU(),
            conv_nd(2, self.model_channels, self.out_channels, 3, padding=1),
        )

    def forward(self, x):
        out =self.end2end(x)
        return out