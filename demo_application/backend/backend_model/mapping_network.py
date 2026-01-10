import torch
import torch.nn as nn
from .utils import EqualLRLinear
from .utils_generator import PixelNorm
from . import globals
import torch.nn.functional as F


class MappingNetwork(nn.Module):
    def __init__(
        self,
        z_dim=globals.Z_DIM,
        w_dim=globals.W_DIM,
        n_layers=globals.N_LAYERS_MAPPING_NETWORK,
        lr_mul=globals.LR_MAPPING_NETWORK,
    ):
        super().__init__()
        layers = []
        layers.append(
            PixelNorm()
        )
        dim_in = z_dim
        for _ in range(n_layers):
            layer = EqualLRLinear(dim_in, w_dim)
            layers.append(layer)
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            dim_in = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
