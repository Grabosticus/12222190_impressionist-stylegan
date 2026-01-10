import torch
import torch.nn as nn
import math
import random
from . import globals
from .utils import linear_interpolation
from .utils_generator import *
from . import mapping_network
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(
        self,
        z_dim=globals.Z_DIM,
        w_dim=globals.W_DIM,
        base_channels=globals.BASE_CHANNELS,
        max_res=globals.MAX_RES,
    ):
        super().__init__()
        self.current_res = 4  # we always start with a 4x4 image
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.mapping = mapping_network.MappingNetwork(z_dim, w_dim)

        # the learned constant input to the generator
        self.const_input = nn.Parameter(torch.randn(1, base_channels, 4, 4))

        # the number of channels used at each resolution
        self.n_channels = globals.CHANNELS_PER_RES

        # here we build the model for each resolution
        self.styled_convs = nn.ModuleDict()
        self.to_rgbs = nn.ModuleDict()

        # at the 4x4 resolution, we don't apply upsampling at the start to build the first low-resolution image
        styled_conv = nn.ModuleList(
            [
                StyledConv(base_channels, base_channels),
                StyledConv(base_channels, base_channels),
            ]
        )
        self.styled_convs[str(4)] = styled_conv
        self.to_rgbs[str(4)] = ToRGB(base_channels)

        # starting from the 8x8 resolution, we apply an upsampling operation after two StyledConv blocks and on ToRGB block
        in_channels = base_channels
        for i in range(3, int(math.log2(max_res)) + 1):
            out_resolution = 2**i
            out_channels = self.n_channels[out_resolution]

            styled_conv = nn.ModuleList(
                [
                    StyledConv(in_channels, out_channels, upsample=True),
                    StyledConv(out_channels, out_channels),
                ]
            )
            self.styled_convs[str(out_resolution)] = styled_conv
            self.to_rgbs[str(out_resolution)] = ToRGB(out_channels)
            in_channels = out_channels

        # here we build the mechanism to gradually fade in the layers that correspond to higher resolutions
        self.register_buffer("layer_opacity", torch.tensor(1.0))

    def fade_in(self, new_res):
        self.current_res = new_res
        self.layer_opacity.fill_(0)

    def set_layer_opacity(self, layer_opacity):
        self.layer_opacity.fill_(float(layer_opacity))

    def forward(self, z, noise=None):

        # we map the latent vector z to its (hopefully) disentangled version w
        w = self.mapping(z)

        # we implement style mixing here. That means we switch the style vector w at some point in the forward process
        if torch.rand(1).item() < globals.STYLE_MIXING_PROB:
            z2 = torch.randn_like(z)
            w2 = self.mapping(z2)
            # the change point from where w2 is used
            change_point = random.randint(1, int(math.log2(self.current_res)) - 1)
        else:
            change_point = None

        # here we copy the constant 1xBASE_CHANNELSx4x4 input BATCH_SIZE times
        x = self.const_input.repeat(z.size(0), 1, 1, 1)

        # we start by using the 4x4 blocks
        if change_point and change_point == 1:
            w = w2
        styled_conv = self.styled_convs["4"]
        to_rgb = self.to_rgbs["4"]
        x = styled_conv[0](x, w)
        x = styled_conv[1](x, w)
        rgb = to_rgb(x, w)

        # if the current resolution is 4 i.e. we didn't start to fade in new layers yet, we can just return
        if self.current_res == 4:
            return torch.tanh(rgb)

        # we apply the blocks for each resolution <= current_res
        for resolution_key in sorted(self.styled_convs.keys(), key=lambda k: int(k)):
            if int(resolution_key) == 4:
                continue

            if change_point and change_point == int(math.log2(int(resolution_key))) - 1:
                w = w2

            styled_conv = self.styled_convs[resolution_key]
            to_rgb = self.to_rgbs[resolution_key]

            x = styled_conv[0](x, w)
            x = styled_conv[1](x, w)
            new_rgb = to_rgb(x, w)

            if int(resolution_key) == self.current_res:
                # we have reached the highest resolution (for now). We can return the image after interpolating it

                # the output is an interpolation of the new high-res rgb image and the rgb image from the next lower
                if self.layer_opacity.item() < 1.0:
                    prev_rgb_upsampled = F.interpolate(
                        rgb,
                        scale_factor=2,
                        mode=globals.INTERPOLATION_MODE,
                        align_corners=False,
                    )
                    output = linear_interpolation(
                        prev_rgb_upsampled, new_rgb, self.layer_opacity.item()
                    )
                else:
                    output = new_rgb  # we already faded the new layer in --> no interpolation required
                rgb = output
                break
            else:
                rgb = new_rgb  # we didn't reach the layer that is faded in yet --> no interpolation required

        return torch.tanh(rgb)
