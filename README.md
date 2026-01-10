# Impressionist StyleGAN

This README provides an overview of the project and documents the changes made to it.

## Project Structure

Below is a brief description of the project’s directories and files. For more details, refer to the code itself—each file includes comments explaining its purpose and usage.

- **environment_cpu.yml** — YAML file listing the required packages to run the application *without* an NVIDIA GPU.
- **environment_nvidia.yml** — YAML file listing the required packages to run the application with CUDA support.
- **filtered_impressionist_clusters.csv** — CSV file containing the dataset. You can download it from the **Impressionist Artworks v1.0** GitHub Release.

- **preprocessing/** — Code used to preprocess the dataset.
  - **similarity_clustering.py** — Python script for K-Means clustering.
  - **clusters/** — Images and example outputs for the generated clusters.
  - **filtered_impressionist_noclusters.csv** — CSV dataset without clusters. You can download it from the **Impressionist Artworks v1.0** GitHub Release.

- **model/** — Main model implementation.
  - **utils.py** — Utility functions used across the model.
  - **utils_generator.py** — Utility functions specific to the Generator.
  - **utils_discriminator.py** — Utility functions specific to the Discriminator.
  - **MNISTTrain.ipynb** — Notebook for training StyleGAN on the MNIST dataset.
  - **MNISTDataset.py** — MNIST dataset implementation as a PyTorch `Dataset`.
  - **mapping_network.py** — Mapping Network used in the Generator.
  - **ImpressionistTrain.ipynb** — Notebook for training StyleGAN on the impressionist dataset.
  - **globals.py** — Model hyperparameters.
  - **generator.py** — StyleGAN Generator implementation.
  - **discriminator.py** — StyleGAN Discriminator implementation.
  - **ADA.py** — Adaptive Discriminator Augmentation implementation.
  - **weights/** — Directory where model weights are saved.
  - **training_imgs/** — Generated images saved throughout different training stages.
  - **results/** — Image grids produced by the final Generator / EMA Generator.
  - **fid/** — Text files tracking the model’s FID score across different training stages.
  - **test_stylegan.py** — Python script containing a few sanity-check tests for the model architecture.

- **impressionist/** — Directory containing the dataset images. You can download it from the **Impressionist Artworks v1.0** GitHub Release.
  - **{Id}.jpg** — Image file with ID `{Id}`.

## Running the application
**Python Version: 3.10** \
Below are the Conda commands needed to set up and run the application. If you have an NVIDIA GPU, install `environment_nvidia.yml`, otherwise install `environment_cpu.yml`.

```
conda env create -f environment_nvidia.yml
conda activate impressionist-stylegan
```

The GitHub repository includes a release called **Impressionist Artworks v1.0**, which contains the dataset images, the dataset CSV files (with and without clusters), and the weights for a pretrained StyleGAN.

Unpack the images and the dataset CSV with the clusters into the root directory of the repository. After extracting, the root directory should contain:
- a folder named `impressionist`
- a file named `filtered_impressionist_clusters.csv`

The weights file `ada_stylegan_64_more_channels.pth` should be placed in the directory `model/weights`.

### Running the preprocessing K-Means script
Running the preprocessing script is **optional**. Its output (`filtered_impressionist_clusters.csv`) is already available in the GitHub release. If you still want to generate it yourself, download `filtered_impressionist_noclusters.csv` from the GitHub release, place it in the preprocessing directory, and run:

```
conda activate impressionist-stylegan
cd preprocessing
python similarity_clustering.py
```

### Running the StyleGAN Training Notebook
Execute the `ImpressionistTrain.ipynb` notebook cell-by-cell. Near the end of the notebook, you’ll find a section that lets you save your newly trained StyleGAN weights to the `weights` directory.

The final section loads my pretrained StyleGAN weights (`ada_stylegan_64_more_channels.pth`, available in the **Impressionist Artworks v1.0** GitHub release) and generates an image grid similar to the one in **model/results**.

## Summary
In this project, I implemented a full StyleGAN pipeline to generate images of impressionist artworks. Due to GPU limitations, I trained the model only up to a resolution of **64×64**, and a full training run required **over 24 hours**.

If you’d like to try the StyleGAN on a simpler dataset, run the `MNISTTrain.ipynb` notebook. It trains on MNIST and reaches visually convincing results much faster.

### Error Metric: Frechet Inception Distance (FID)
I used the Frechet Inception Distance (FID) to measure how closely the distribution of generated images matches the distribution of real images. FID is a standard metric for evaluating GAN image quality and is used in most GAN-related research.

### Error Metric Target: 40
My goal was to achieve an FID of 40 or lower (smaller values indicate better results). Selecting a suitable target was challenging because FID can produce higher (worse) values for equally good-looking samples when the underlying dataset is relatively small, as in my case.

### Achieved FID Value: 40.38
By the end of training, the EMA Generator achieved an FID of 40.38. You can see images generated by this Generator in `model/results/grid_more_channels_ema.png`.

### Work Breakdown
Below is a breakdown of the hours I spent on the individual tasks specified in the original project proposal. \
Format: `TASK: HOURS_SPENT (INCREASE_COMPARED_TO_ORIGINAL_ESTIMATE)`

- Researching project ideas: 2h
- Planning project: 2h
- Finding dataset sources: 6h (+4h)
- Writing code for dataset collection: 5h (+1h)
- Collecting all dataset entries: 10h
- Refining dataset: 4h
- Building model: 40h (+30h)
- Training the model: 25h (+10h)
- Building an application to present the results: NOT DONE YET
- Preparing the final report and presentation: NOT DONE YET

## Dataset
I collected a dataset of 23,328 impressionist artworks. My original plan was to rely primarily on museum APIs, but I found that they did not provide enough works to build a dataset of the desired size. As a result, I switched to collecting the images from WikiArt [1]. This approach was especially convenient because WikiArt provides text lists of artworks for a given style, along with metadata such as the artwork’s genre.

During early training runs, I observed that the StyleGAN struggled to produce convincing images. I hypothesized that this was caused by the high variance within the dataset—for example, it includes both portraits and landscape paintings, which differ substantially in composition and structure. To reduce this variance, I applied K-Means clustering to partition the dataset into groups of visually and semantically similar images. This change significantly improved the quality of the generated results.

The final dataset is found in the file `filtered_impressionist_clusters.csv`. It contains the following columns: \
Id, Author, Style, Title, Date, Genre, Image_urls, URL, Cluster_ind \
**Id**... The ID of the dataset entry \
**Author**... The author of the artwork \
**Style**... The style of the artwork (e.g. Impressionism, Post-Impressionism, etc.) \
**Title**... The title of the artwork \
**Date**... The estimated date/year in which the artwork was created \
**Genre**... The theme of the artwork (e.g. still life, portrait, landscape, etc.) \
**Image_urls**... A list of URLs to the artwork image \
**URL**... The URL to the webpage the artwork was taken from \
**Cluster_ind**... The ID of the cluster the image was grouped into during K-Means clustering

## Model Architecture
I implemented a StyleGAN [2] largely following the original paper, but incorporated several modifications introduced in later variants to improve performance on my dataset. Below, I summarize the main design choices and the motivation behind each one.

### Adaptive Discriminator Augmentation (ADA)
I observed that the discriminator tended to overfit—especially after clustering, which effectively reduces the dataset size per cluster. To address this, I implemented Adaptive Discriminator Augmentation (ADA) to detect signs of discriminator overfitting and apply data augmentation when necessary. ADA is a technique used in later StyleGAN architectures, such as StyleGAN2-ADA [3].

### Loss Function: Non-Saturating Logistic Loss (with R1 penalty for the discriminator)
In early training runs, I used the Wasserstein loss. However, after introducing ADA, I needed a loss formulation whose discriminator scores are centered around zero, since ADA detects overfitting based on the sign of the discriminator output on real images. The non-saturating logistic loss satisfies this requirement, and I additionally applied an R1 penalty to regularize the discriminator.

### Exponential Moving Average (EMA) Generator
I maintained an EMA version of the Generator, where the weights are computed as a moving average of the main Generator’s weights. This produces smoother weight updates and typically yields more stable and higher-quality outputs than using the raw Generator weights alone.

### Hyperparameters
I selected hyperparameters based on those used in official StyleGAN implementations.


## Performing Sanity Check Tests
To perform the sanity check tests in `test_stylegan.py`, run the following commands:
```
conda activate impressionist-stylegan
cd model
python test_stylegan.py
```

## References
[1]: WikiArt, https://www.wikiart.org/ \
[2]: Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019. \
[3]: Karras, Tero, et al. "Training generative adversarial networks with limited data." Advances in neural information processing systems 33 (2020): 12104-12114.
