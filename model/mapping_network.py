import torch
import torch.nn as nn
from utils import EqualLRLinear
from utils_generator import PixelNorm
import globals
import torch.nn.functional as F


class MappingNetwork(nn.Module):
    """
    The Mapping Network is the defining difference between a PGGAN and a StyleGAN.
    It takes a latent vector z, consisting of random Gaussian noise, as input and transforms it
    to a vector w. The goal of this is to reduce "entanglement" of the input vector.
    What is "entanglement"?
    In theory we want each entry of our input vector to the Generator to represent a specific property of
    the output image e.g. hair color. When we change the value for the hair color property, we want said color to change.
    If the input vector is entangled, however, changing only the hair color property will also change other properties.
    e.g. changing only the hair color also increases the age of the person in the output.
    """

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
        )  # our first layer is Pixel-Wise Feature Normalization
        dim_in = z_dim
        for _ in range(n_layers):
            layer = EqualLRLinear(dim_in, w_dim)
            layers.append(layer)
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            dim_in = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
