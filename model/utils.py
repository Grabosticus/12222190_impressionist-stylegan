import torch.nn as nn
import globals
import math
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import numpy as np
import os


def linear_interpolation(a, b, t):
    return a + (b - a) * t


class EqualLRConv2d(nn.Module):
    """
    This is a convolutional layer that uses Equalized Learning Rate.
    Equalized Learning Rate is a concept that helps stabilize convergence by updating
    each layer with approximately the same learning rate, regardless of the layers size
    (and therefore its effect on the gradient)
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        nn.init.normal_(self.conv.weight, mean=0, std=1)
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = math.sqrt(2 / fan_in)

    def forward(self, input):
        return self.conv(input * self.scale)


class EqualLRLinear(nn.Module):
    """
    This is a linear layer that uses Equalized Learning Rate.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.normal_(self.linear.weight, mean=0, std=1)
        self.scale = math.sqrt(2 / in_dim)

    def forward(self, input):
        return self.linear(input * self.scale)


def get_generator_params(generator):
    """
    The learning rate for the mapping network in the generator is lower than that for the
    generator itself. Therefore we need to get the parameters of the mapping network and the
    actual generator seperately
    """
    mapping_params = []
    generator_params = []
    for name, param in generator.named_parameters():
        if "mapping" in name:
            mapping_params.append(param)
        else:
            generator_params.append(param)
    return mapping_params, generator_params


def compute_fid(
    G,
    G_EMA,
    dataset,
    res,
    percent_this_phase,
    fid: FrechetInceptionDistance,
    output_file="fid/fids.txt",
    max_imgs: int | None = None,
):
    """
    Here we compute the Frechet Inception Distance i.e. the distance between our real dataset and the generated images.
    Lower FID values mean our generator performs better.
    """
    os.makedirs("fid", exist_ok=True)
    transform = transforms.Compose([transforms.Resize((299, 299))])
    batch_size = globals.BATCH_SIZES_PER_RES[res]

    fid.reset()

    result_fids = dict()

    # Normal Generator FID
    # here we add the real images from the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4
    )
    dataloader_cut = len(dataloader)
    if max_imgs:
        dataloader_cut = max_imgs // batch_size
    for i, batch in enumerate(dataloader):
        if i > dataloader_cut:
            break
        real_images = transform(batch)  # FID needs images to be 299x299
        real_images = real_images.to(globals.DEVICE)
        # FID needs images to be in [0,255]
        real_images = (real_images + 1) / 2.0
        real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
        fid.update(real_images, real=True)

    # here we add the fake images
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            real_batch_size = min(batch_size, len(dataset) - i)
            z = torch.randn(real_batch_size, globals.Z_DIM).to(globals.DEVICE)
            fake_images = G(z)
            fake_images = transform(fake_images)
            fake_images = (fake_images + 1) / 2.0
            fake_images = (fake_images * 255).clamp(0, 255).to(torch.uint8)
            fid.update(fake_images, real=False)

    fid_score = fid.compute()
    with open(output_file, "a") as f:
        f.write(
            f"Generator FID at {res}x{res} Resolution phase {percent_this_phase:.1f}% done: {fid_score.item()}\n"
        )
    result_fids["G"] = fid_score.item()

    fid.reset()

    # EMA Generator FID
    # here we add the real images from the dataset
    for i, batch in enumerate(dataloader):
        if i > dataloader_cut:
            break
        real_images = transform(batch)
        real_images = real_images.to(globals.DEVICE)
        real_images = (real_images + 1) / 2.0
        real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
        fid.update(real_images, real=True)

    # here we add the fake images
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            real_batch_size = min(batch_size, len(dataset) - i)
            z = torch.randn(real_batch_size, globals.Z_DIM).to(globals.DEVICE)
            fake_images = G_EMA(z)
            fake_images = transform(fake_images)
            fake_images = (fake_images + 1) / 2.0
            fake_images = (fake_images * 255).clamp(0, 255).to(torch.uint8)
            fid.update(fake_images, real=False)

    fid_score = fid.compute()
    with open(output_file, "a") as f:
        f.write(
            f"EMA Generator FID at {res}x{res} Resolution phase {percent_this_phase:.1f}% done: {fid_score.item()}\n"
        )
    result_fids["G_EMA"] = fid_score.item()

    return result_fids


def generate_grid_image(G, fid_score, resolution, output_dir="results"):
    """
    Generates a 32x32 grid of images generated by our model. It then saves the image grid along with our current FID score in a file.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_rows = 32
    n_cols = 32
    n_images = n_rows * n_cols

    grid_width = n_cols * resolution
    grid_height = n_rows * resolution
    canvas = Image.new("RGB", (grid_width, grid_height))
    generated = 0

    while generated < n_images:
        current_batch = min(4, n_images - generated)

        z = torch.randn(current_batch, globals.Z_DIM, device=globals.DEVICE)
        with torch.no_grad():
            imgs = G(z)

        for i in range(current_batch):
            idx = generated + i
            r = idx // n_cols
            c = idx % n_cols

            img = imgs[i]
            img = (img + 1) / 2.0
            img_np = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode="RGB")

            x = c * resolution
            y = r * resolution
            canvas.paste(pil_img, (x, y))

        generated += current_batch
    canvas.save(f"{output_dir}/{resolution}x{resolution}_FID_{fid_score}.png")
