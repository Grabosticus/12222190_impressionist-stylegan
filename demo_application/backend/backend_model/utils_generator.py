import torch
import torch.nn as nn
from . import globals
import torch.nn.functional as F
from .utils import EqualLRConv2d, EqualLRLinear


class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)


class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()

        self.affine = EqualLRLinear(w_dim, channels * 2)

    def forward(self, x, w):
        style = self.affine(w).unsqueeze(-1).unsqueeze(-1)
        y_s, y_b = style.chunk(2, dim=1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = (
            x.var(dim=(2, 3), unbiased=False, keepdim=True).add(1e-8).sqrt()
        )
        x = (x - mean) / std
        return ((1 + y_s) * x) + y_b


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise


class StyledConv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        w_dim=globals.W_DIM,
        kernel_size=globals.KERNEL_SIZE,
        upsample=False,
    ):
        super().__init__()
        self.upsample = upsample

        self.conv = EqualLRConv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)

        self.noise = NoiseInjection(out_ch)
        self.adain = AdaIN(out_ch, w_dim)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w, noise=None):
        if self.upsample:
            x = F.interpolate(
                x, scale_factor=2, mode=globals.INTERPOLATION_MODE, align_corners=False
            )
        x = self.conv(x)
        x = self.noise(x, noise)
        x = self.adain(x, w)
        x = self.lrelu(x)
        return x


class ToRGB(nn.Module):
    def __init__(self, in_ch, w_dim=globals.W_DIM):
        super().__init__()

        self.conv = EqualLRConv2d(in_ch, 3, 1)

    def forward(self, x, w):
        x = self.conv(x)
        return x
