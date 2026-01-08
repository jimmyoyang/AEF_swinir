# 文件路径: models/model_b2.py
# (重构后，与 AlphaSR 框架深度集成)

import torch.nn as nn
from .basic_ops import conv_nd  # 【核心修改】从框架的 basic_ops 导入通用的卷积模块

class SRCNN_B2(nn.Module):
    """
    SRCNN_B2 模型 (AlphaSR 框架集成版)

    这个版本进行了重构，以充分利用 AlphaSR 框架提供的工具：
    - 使用 `basic_ops.conv_nd` 替代了 `torch.nn.Conv2d`，与框架中的其他模型 (如 samplenet.py) 保持一致。
    - 将卷积层和激活函数包装在 `nn.Sequential` 中，使代码更清晰。
    - 【重要】在 forward 方法的末尾添加了 `nn.Tanh()` 激活函数。这是因为我们的数据管道
      (datasets.py) 将数据归一化到了 [-1, 1] 范围，此操作能确保模型输出与目标范围匹配，
      是模型能正常收敛的关键。
    """
    def __init__(self, in_channels=9, out_channels=3):
        super(SRCNN_B2, self).__init__()

        # ------------------- 架构修改 -------------------
        # 我们现在使用 conv_nd(dims=2, ...) 来定义所有卷积层
        
        self.conv_block1 = nn.Sequential(
            conv_nd(dims=2, in_channels=in_channels, out_channels=64, kernel_size=9, padding=4, bias=True),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block2 = nn.Sequential(
            conv_nd(dims=2, in_channels=64, out_channels=32, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block3 = conv_nd(dims=2, in_channels=32, out_channels=out_channels, kernel_size=5, padding=2, bias=True)
        
        # ------------------- 关键激活函数 -------------------
        self.final_activation = nn.Tanh()

    def forward(self, x):
        """
        定义前向传播路径
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        out = self.final_activation(x)
        return out

