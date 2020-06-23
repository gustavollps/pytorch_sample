import torch
from torch import nn
import torch.nn.functional as F


class CustomModel(nn.Module):

    def __init__(self, size, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(
            3, 64,  # input layers (3 = RGB), output layers
            kernel_size=3, stride=1,
            padding=True, bias=False
        )
        self.conv2 = nn.Conv2d(
            64, 128,  # input layers, output layers
            kernel_size=3, stride=1,
            padding=True, bias=False
        )
        self.flat = size[0] * size[1] * 128
        self.final = nn.Linear(self.flat, n_classes)

    def forward(self, input):
        x = self.conv(input)
        x = self.conv2(x)
        x = F.relu(x)  # ReLu activation
        x = x.view(x.shape[0], -1)  # flattening the tensor for the linear layer
        return self.final(x)
