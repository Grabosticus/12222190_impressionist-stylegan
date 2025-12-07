# Impressionist StyleGAN
This readme contains informations about the project as well as changes made to it.

## Project Structure
Here I will provide a short description of all directories and files in the project. For more information on the code, see the files themselves. Each file contains comments describing the code and what it is used for.

- requirements.txt... The packages needed to run the application
- filtered_impressionist_clusters_2.csv... The CSV file containing the dataset
- preprocessing... Directory with code used to preprocess the dataset
    - similarity_clustering.py... Python Script for K-Means clustering
    - clusters... Contains images and examples of the clusters
    - filtered_impressionist.csv... The CSV dataset without clusters
- model... Directory with main model code
    - utils.py... Contains utility functions for the whole model
    - utils_generator.py... Contains utility functions for the Generator
    - utils_discriminator.py... Contians utility functions for the Discriminator
    - MNISTTrain.ipynb... The notebook to train the StyleGAN on the MNIST Dataset
    - MNISTDataset.py... The MNIST dataset as a PyTorch dataset
    - mapping_network.py... The Mapping Network used in the Generator
    - ImpressionistTrain.ipynb... The notebook to train the StyleGAN on the impressionist dataset
    - globals.py... The hyperparameters used for the model
    - generator.py... Contains the code for the StyleGAN Generator
    - discriminator.py... Contains the code for the StyleGAN Discriminator
    - ADA.py... Contains the code for Adaptive Discriminator Augmentation
    - weights... The directoy in which the model weights are saved
    - training_imgs... Contains generated images throughout different training stages
    - results... Contains a grid of images produced by the final Generator/EMA Generator
    - fid... Contains text files that track the FID score of the model during different stages of training
- impressionist... The directory with the dataset images.
    - {Id}.jpg... The the image with id {Id}

## Running the application
**Python Version: 3.10** \
Here are some conda commands to initialize the virtual environment and install all needed packages.

```
conda create -n impressionist-stylegan python=3.10
conda activate impressionist-stylegan
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

In the GitHub repository, there is a release **Impressionist Artworks v1.0** which contains the dataset images and weights of a pretrained StyleGAN. Unpack the images into the main directory of the repository. The main directory should now contain a directory called `impressionist`. The weights file **ada_stylegan_64_more_channels.pth** should be saved in a directory called `weights`

### Running the preprocessing K-Means script
Keep in mind, that is isn't required to run the preprocessing script. Its result - **filtered_impressionist_clusters_2.csv**  - is already in the main directory. If you still want to do it however, you can run the following commands:

```
conda activate impressionist-stylegan
cd preprocessing
python similarity_clustering.py
```

### Running the StyleGAN Notebook
Run the notebook from top to bottom

## Summary
### Error Metric: Frechet Inception Distance (FID)
I used the FID metric to calculate the distance between the distribution of the real images and the distribution of the generated images. FID is the norm to evaluate the quality of GANs and has been used in nearly all papers.

### Error Metric Target: 40
I wanted to reach an FID of at most 40 (smaller FID values are better).

### Achieved FID Value: 40.38
The EMA Generator achieved this FID value at the end of the model training.

### Work Breakdown
Here is a breakdown of the hours I spent on the individual tasks specified in the original project proposal. It is of this format: \
TASK: HOURS_SPENT (INCREASE_COMPARED_TO_ORIGINAL_ESTIMATE)

Researching project ideas: 2h \
Planning project: 2h \
Finding dataset sources: 6h (+4h) \
Writing code for dataset collection: 5h (+1h) \
Collecting all dataset entries: 10h \
Refining dataset: 4h \
Building model: 40h (+30h) \
Training the model: 25h (+10h) \
Building an application to present the results: NOT DONE YET \
Preparing the final report and presentation: NOT DONE YET 

## Dataset
I collected a dataset of 23328 impressionist artworks. Originally, I wanted to use mostly museum APIs to do this, but found out, that they don't contain enough artworks to build a large dataset. Therefore, I switched to collecting them from WikiArt [1]. This was very convenient since WikiArt offers text lists of all artworks belonging to a specific style, along with some metadata like the genre of the artwork. \
In early training runs, I noticed that the StyleGAN had problems in generating good-looking images. I hypothesized that this was due to the high variance in the individual artworks of the dataset. For example, the dataset contains portraits of people as well as landscape painting - two very different genres. Therefore I decided to use K-Means clustering to seperate the dataset into clusters, where each cluster contains images that are visually and semantically similar to one another. This improved the quality to of the generated images drastically.

The final dataset is found in the file **filtered_impressionist_clusters_2.csv**. It contains the following colums: \
Id, Author, Style, Title, Date, Genre, Image_urls, URL, Cluster_ind \
**Id**... The ID of the dataset entry \
**Author**... The author of the artwork \
**Style**... The style of the artwork (e.g. Impressionism, Post-Impressionism, etc.) \
**Title**... The title of the artwork \
**Date**... The estimated date/year in which the artwork was created \
**Genre**... The theme of the artwork (e.g. still life, portrait, landscape, etc.) \
**Image_urls**... A list of urls to the artwork image \
**URL**... The URL to the webpage the artwork was taken from \
**Cluster_ind**... The ID of the cluster the image was grouped into during K-Means clustering

## Model Architecture
I created a StyleGAN [2], as described in the original paper, but with some changes described in later architectures to improve the performance of my model on the dataset. Here, I will list my design choices and why they have been chosen.

### Adaptive Discriminator Augmentation (ADA)
I noticed that, especially after clustering and therefore decreasing the effective dataset size, my model was prone to discriminator overfitting. Therefore I created an ADA mechanism to recognize discriminator overfitting and perform data augmentation if it is detected. ADA is a mechanism, that is used in later StyleGAN architectures, such as StyleGAN2-ADA [3].

### Loss Function: Non-Saturating Logistic Loss (with R1 penalty for the discriminator)
In early training runs I used the Wasserstein Loss function, but after I implemented ADA, I had to switch to a loss function which is centered around 0, since ADA recognizes discriminator overfitting by the signage of the score on the real images. Non-Saturating Logistic Loss satisfies this property.

### Exponential Moving Average (EMA) Generator
I used a EMA Generator, whose weights are a moving average of the Generator weights. This makes the EMA Generator weight changes smoother than those for the original Generator, resulting in better overall performance. 


### Hyperparameters
I based my hyperparameter selection on hyperparameters used in official StyleGAN implementations.


## References
[1]: WikiArt, https://www.wikiart.org/ \
[2]: Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019. \
[3]: Karras, Tero, et al. "Training generative adversarial networks with limited data." Advances in neural information processing systems 33 (2020): 12104-12114.