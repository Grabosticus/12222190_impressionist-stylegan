import torch
import torch.nn as nn
import globals
import torch.nn.functional as F
from utils import EqualLRConv2d
from ADA import ADA

class FromRGB(nn.Module):
    """
    This block is used in the Discriminator and is the equivalent of the ToRGB block in the Generator.
    We once again use a 1x1 convolution to convert the RGB image to a feature map with out_ch channels.
    """
    def __init__(self, out_ch):
        super().__init__()

        self.conv = EqualLRConv2d(3, out_ch, 1)
    
    def forward(self, x):
        return self.conv(x)


class DiscriminatorBlock(nn.Module):
    """
    The core of the Discriminator.
    The Discriminator block for a StyleGAN is essentially the same as the Discriminator block for a PGGAN.
    This essentially means replacing the Styled Convolution blocks used in the Generator with normal Convolutions.
    """
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()

        self.conv1 = EqualLRConv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = EqualLRConv2d(in_ch, out_ch, 3, padding=1)
        self.downsample = downsample
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        if self.downsample:
            x = F.interpolate(x, scale_factor=0.5, mode=globals.INTERPOLATION_MODE, align_corners=False)
        return x
    

class MiniBatchStandardDeviation(nn.Module):
    """
    MiniBatch Standard Deviation (MBSTD) is a module used to counter mode-collapse.
    Mode-Collapse is what happens, when the Generator only produces images that look almost the same, because each individual image
    fools the Discriminator. MBSTD counters this by adding an additional channel to the batch:
    This channel contains the standard deviation of the individual pixels in each location. If this value is low everywhere,
    the Generator likely output almost identical images for this batch. By providing this information the Discriminator
    can penalize this behaviour.
    """
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size # the size of sub-batches we compute the standard deviation over

    def forward(self, x):
        batch_size, n_channels, height, width = x.shape
        actual_group_size = min(self.group_size, batch_size)

        remainder = batch_size % actual_group_size
        if remainder != 0:
            x = x[:batch_size - remainder]
        groups = x.view(actual_group_size, -1, n_channels, height, width) # we reshape the tensor to reflect the group size
        variance = torch.var(groups, dim=0, unbiased=False) + 1e-8 # the variance for pixels each group
        stddev = torch.sqrt(variance)
        means_of_stddev = stddev.mean(dim=[1,2,3], keepdim=True) # the actual average stddev for this group (a single value)
        means_of_stddev = means_of_stddev.repeat(actual_group_size, 1, height, width) # we just resize the average stddev to match our input shape
        return torch.cat([x, means_of_stddev], dim=1)
    

def d_loss_non_saturating_r1(D, real_imgs, fake_imgs, d_step, ada: ADA, gamma=10, log=False):
    """
    The loss function of the Discriminator
    I originally used WGAN-GP, but ADA didn't work with it, since almost all signs
    of the D output were always negative, even if D overfitted heavily.
    """
    real_augmented = ada.augment(real_imgs)
    fake_augmented = ada.augment(fake_imgs)

    real_pred = D(real_augmented)
    fake_pred = D(fake_augmented)

    logistic_loss = F.softplus(fake_pred).mean() + F.softplus(-real_pred).mean()

    # ADA update
    with torch.no_grad():
        ada_pred = D(real_imgs)
        rt = (ada_pred > 0).float().mean().item()
        ada.update_p(rt, real_imgs.size(0))

    # R1 penalty
    r1_penalty = 0.0

    if d_step % 16 == 0:
        real_imgs_r1 = real_imgs.detach().requires_grad_(True)
        real_pred_r1 = D(real_imgs_r1)
        grad_real = torch.autograd.grad(
            outputs=real_pred_r1.sum(),
            inputs = real_imgs_r1,
            create_graph=True,
            retain_graph=True
        )[0]

        r1 = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        r1_penalty = (gamma / 2) * r1 * 16

    D_loss = logistic_loss + r1_penalty

    if log:
        print(f"REAL SCORES: {real_pred[:10]}")
        print(f"FAKE SCORES: {fake_pred[:10]}")
        print(f"DISCRIMINATOR_LOSS: {D_loss}")
        print(f"ADA rt: {rt:.4f}, p: {ada.p:.4f}")

    return D_loss

