import torch

Z_DIM = 512
W_DIM = 512
N_LAYERS_MAPPING_NETWORK = 8
LR_MODEL = 0.002
LR_MODEL_PER_RES = {4: 0.002, 8: 0.002, 16: 0.002, 32: 0.002, 64: 0.002, 128: 0.002}
LR_MAPPING_NETWORK_PER_RES = {res: 0.1 * lr for res, lr in LR_MODEL_PER_RES.items()}
LR_MAPPING_NETWORK = 0.01 * LR_MODEL
KERNEL_SIZE = 3
BASE_CHANNELS = 512
MAX_RES = 128  # the resolution in one dimension of the output image (MAX_RESxMAX_RES)
FLOOR_CHANNELS = 128  # the minimum number of channels for all resolutions.
INTERPOLATION_MODE = "bilinear"

IMAGES_PER_RESOLUTION = {
    4: 300_000,
    8: 350_000,
    16: 400_000,
    32: 450_000,
    64: 600_000,
    128: 800_000,
}
CHANNELS_PER_RES = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128}
FADE_IN_PERCENTAGE = (
    0.5  # the percentage of images per resolution that are used to fade in new layers
)
ADAM_BETA1 = 0.0
ADAM_BETA2 = 0.99
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISCRIMINATOR_STEPS = 1
BATCH_SIZES_PER_RES = {4: 128, 8: 128, 16: 64, 32: 32, 64: 32, 128: 16}
STYLE_MIXING_PROB = 0.5
