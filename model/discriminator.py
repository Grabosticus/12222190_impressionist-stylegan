import torch
import torch.nn as nn
import math
import globals
from utils import EqualLRConv2d, EqualLRLinear, linear_interpolation
from utils_discriminator import *
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    The Discriminator of StyleGAN is the same as the Discriminator for PGGAN.
    """

    def __init__(self, base_channels=globals.BASE_CHANNELS, max_res=globals.MAX_RES):
        super().__init__()
        self.from_rgbs = nn.ModuleDict()
        self.blocks = nn.ModuleDict()

        # we one again calculate the number of channels for each resolution
        self.n_channels = globals.CHANNELS_PER_RES

        # the discriminator is a mirror of the generator. We start from high resolution and go down
        for i in range(int(math.log2(max_res)), 2, -1):
            current_resolution = 2**i
            current_channels = self.n_channels[
                current_resolution
            ]  # the num of channels used at this resolution layer

            self.from_rgbs[str(current_resolution)] = FromRGB(current_channels)

            # output_channels is the number of channels used in the next (lower) resolution layer
            output_channels = max(
                globals.FLOOR_CHANNELS, self.n_channels[current_resolution // 2]
            )
            self.blocks[str(current_resolution)] = DiscriminatorBlock(
                current_channels, output_channels, downsample=True
            )

        # the final layer in the discriminator produces a prediction on the "realness" of the input image
        self.final_from_rgb = FromRGB(base_channels)
        self.minibatch_std = (
            MiniBatchStandardDeviation()
        )  # adds one channel to the output of the final FromRGB
        self.final_block = nn.Sequential(
            DiscriminatorBlock(base_channels + 1, base_channels, downsample=False),
            EqualLRConv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            EqualLRLinear(base_channels * (4 * 4), 1),
        )

        self.current_res = 4
        self.register_buffer("layer_opacity", torch.tensor(1.0))

    def fade_in(self, new_res):
        """
        This function is called when we start to fade in the next layer i.e. the one corresponding to the
        next higher resolution.
        At the start of this fading in process, 0% of the new layer is used. At the end, 100%
        """
        self.current_res = new_res
        self.layer_opacity.fill_(0)

    def set_layer_opacity(self, layer_opacity):
        self.layer_opacity.fill_(float(layer_opacity))

    def forward(self, x):
        current_resolution = x.shape[-1]

        # here we fade in the current highest resolution layer
        if current_resolution != 4:
            from_rgb = self.from_rgbs[str(current_resolution)]
            block = self.blocks[str(current_resolution)]

            # this is the output of the current highest resolution Discriminator layer
            x_new = from_rgb(x)
            x_new = F.leaky_relu(x_new, 0.2, inplace=True)
            x_new = block(x_new)

            # here we interpolate the output of the previous resolution layer with the one of the current layer
            x_down = F.interpolate(
                x,
                scale_factor=0.5,
                mode=globals.INTERPOLATION_MODE,
                align_corners=False,
            )
            down_resolution = current_resolution // 2
            prev_from_rgb = (
                self.from_rgbs[str(down_resolution)]
                if (down_resolution > 4)
                else self.final_from_rgb
            )
            x_prev = prev_from_rgb(x_down)
            x_prev = F.leaky_relu(x_prev, 0.2, inplace=True)

            if self.layer_opacity.item() < 1.0:
                x = linear_interpolation(x_prev, x_new, self.layer_opacity.item())
            else:
                x = x_new  # we already faded in the new layer completely

            current_resolution //= 2
            while current_resolution > 4:
                block = self.blocks[str(current_resolution)]
                x = block(x)
                current_resolution //= 2

        # if current_resolution == 4, we can immediately call the final fromRGB
        else:
            x = self.final_from_rgb(x)
            x = F.leaky_relu(x, 0.2, inplace=True)

        x = self.minibatch_std(x)
        return self.final_block(x)
