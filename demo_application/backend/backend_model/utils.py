import torch.nn as nn
import math
import torch
from PIL import Image
import numpy as np
import os


def linear_interpolation(a, b, t):
    return a + (b - a) * t


class EqualLRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        nn.init.normal_(self.conv.weight, mean=0, std=1)
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = math.sqrt(2 / fan_in)

    def forward(self, input):
        return self.conv(input * self.scale)


class EqualLRLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.normal_(self.linear.weight, mean=0, std=1)
        self.scale = math.sqrt(2 / in_dim)

    def forward(self, input):
        return self.linear(input * self.scale)
