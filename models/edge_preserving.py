# Lint as: python3
import torch
import torch.nn as nn
import cv2
import numpy as np

__all__ = ['edge_preserve']


class EdgePreserve(nn.Module):

    def __init__(self, downsample_factor: int = 16):
        super(EdgePreserve, self).__init__()
        self.kernel_size = 3
        self.sigma_space = 50
        self.sigma_color = 2
        self.dummy_parameter = nn.Parameter(torch.zeros(1), True)
        self.max_pool = nn.MaxPool2d(kernel_size=downsample_factor)
        self.avg_pool = nn.AvgPool2d(kernel_size=downsample_factor)

    def forward(self, x):
        filtered_samples = []
        for xx in x:
            filtered_xx = cv2.bilateralFilter(np.float32(xx.cpu().squeeze()),
                                              d=self.kernel_size,
                                              sigmaColor=self.sigma_color,
                                              sigmaSpace=self.sigma_space)
            filtered_samples.append(torch.tensor(filtered_xx))
        x = torch.stack(filtered_samples).to('cuda')
        x = x.unsqueeze(1)
        x = self.avg_pool(x)
        return x


def edge_preserve():
    return EdgePreserve()
