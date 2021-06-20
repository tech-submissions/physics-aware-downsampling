# Lint as: python3
import torch
import torch.nn as nn

__all__ = ['average_net']


class AverageNet(nn.Module):

    def __init__(self, downsample_factor: int = 16):
        super(AverageNet, self).__init__()
        self.dummy_parameter = nn.Parameter(torch.zeros(1), True)
        self.avg = nn.AvgPool2d(kernel_size=downsample_factor)

    def forward(self, x):
        x = self.avg(x)
        return x


def average_net():
    return AverageNet()
