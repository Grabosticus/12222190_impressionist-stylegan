"""
This script clusters the images of our impressionist artwork dataset into 6 clusters using a K-Means algorithm.
To extract useful features from the images, I used OpenAIs CLIP model.
"""

import os
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


IMAGE_DIR = "../impressionist"
OUTPUT_DIR = "clusters"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

image_paths = []
embeddings = None
labels = None
aspect_ratios = []

print(f"Getting image paths...")
for path in tqdm(os.listdir(IMAGE_DIR)):
    img_path = os.path.join(IMAGE_DIR, path)
    image_paths.append(img_path)

print(f"Computing aspect ratios...")
for img_path in tqdm(image_paths):
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            ratio = w / h
            aspect_ratios.append(ratio)
    except Exception as e:
        print(f"Exception occured: {e}")
        print(f"THIS IS A CRITICAL EXCEPTION")

print(f"Extracting features...")
embeddings_list = []
batch_size = 8

for i in tqdm(range(0, len(image_paths), batch_size)):
    batch_paths = image_paths[i : i + batch_size]
    images = []

    for img_path in batch_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Exception occured: {e}")
            print(f"THIS IS ACRITICAL EXCEPTION")

    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

    embeddings_list.append(image_features.cpu().numpy())

embeddings = np.vstack(embeddings_list)


print(f"Clustering the embeddings with KMEANS and 6 clusters...")

clusterer = KMeans(n_clusters=6, n_init=10)
labels = clusterer.fit_predict(embeddings)

unique, counts = np.unique(labels, return_counts=True)
print(f"Found clusters with the following sizes:")
for cluster_ind, count in zip(unique, counts):
    print(f" Cluster {cluster_ind}: {count} images")


print(f"Visualizing clusters...")


def visualize_clusters(
    image_paths,
    embeddings,
    labels,
    aspect_ratios,
    n_samples_per_cluster=20,
    perplexity=30,
):
    global OUTPUT_DIR
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot t-SNE
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.6,
        s=10,
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title("t-SNE Visualization of Image Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tsne_clusters.png", dpi=150)
    plt.close()

    print("Creating sample image grids for each cluster...")
    n_clusters = len(np.unique(labels))

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        n_samples = min(n_samples_per_cluster, len(cluster_indices))
        sample_indices = np.random.choice(cluster_indices, n_samples, replace=False)

        # Create grid
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten() if n_samples > 1 else [axes]

        for idx, ax in enumerate(axes):
            if idx < n_samples:
                img_path = image_paths[sample_indices[idx]]
                img = Image.open(img_path)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(
                    f"{img_path}...\nAR: {aspect_ratios[sample_indices[idx]]:.2f}",
                    fontsize=8,
                )
            else:
                ax.axis("off")

        plt.suptitle(
            f"Cluster {cluster_id} - {len(cluster_indices)} images", fontsize=16
        )
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/cluster_{cluster_id}_samples.png", dpi=100)
        plt.close()
        print(f"  Saved samples for cluster {cluster_id}")


visualize_clusters(image_paths, embeddings, labels, aspect_ratios)

CSV = "filtered_impressionist.csv"
df = pd.read_csv(CSV)
ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in image_paths]
cluster_map = dict(zip(ids, labels))
df["Cluster_ind"] = df["Id"].map(cluster_map)
df.to_csv("filtered_impressionist_clusters.csv", index=False)
print(df["Cluster_ind"].value_counts())
