import torch
from models.model_b2 import SRCNN_B2

model = SRCNN_B2(in_channels=9, out_channels=3)
x = torch.randn(1, 9, 64, 64)  # 假设输入是9通道，64x64
y = model(x)
print(y.shape)