import torch
from torchvision import transforms, datasets
import os
from PIL import Image
import pandas as pd

class ImpressionistDataset(torch.utils.data.Dataset):
    """
    This is our dataset consisting of 20.000 paintings by Impressionist artists.
    Since the dataset is very diverse, and GANs struggle with such a level of variance, I performed clustering on the images
    and asigned an id to each cluster. The dataset class now returns a dataset with images belonging to the specified cluster_ind
    """
    def __init__(self, resolution, cluster_ind, root=None):
        df = pd.read_csv("../filtered_impressionist_clusters_2.csv")
        cluster = df[df["Cluster_ind"] == cluster_ind]
        cluster_ids = cluster["Id"].tolist()
        cluster_filenames = [f"{i}.jpg" for i in cluster_ids]

        if root is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root = os.path.join(current_dir, "..", "impressionist")
            root = os.path.normpath(root)

        self.image_paths = [
            os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".jpg") and f in cluster_filenames
        ]

        self.resolution = resolution

        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3) # we rescale the pixel values from [0, 1] to [-1, 1] (this is the output of a tanh function)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image
