import torch
from PIL import Image
from torchvision import datasets, transforms
import torch.nn.functional as F
import os


class MNISTDataset(torch.utils.data.Dataset):
    """
    A simple dataset class using the MNIST dataset.
    Since MNIST is 28x28, we add black pixels to the edges to make
    it 32x32
    """

    def __init__(self, resolution, root="./MNIST"):
        self.resolution = resolution

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),  # convert to RGB
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

        os.makedirs("MNIST", exist_ok=True)
        self.mnist = datasets.MNIST(root=root, download=True, transform=self.transform)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, _ = self.mnist[idx]
        return image
