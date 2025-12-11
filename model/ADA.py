import kornia.augmentation as K
import torch


class ADA:
    """
    Adaptive Discriminator Augmentation.
    This prevents discriminator overfitting for small datasets,
    by applying augmentations, which make the discriminator's task
    harder. We update the probability of applying these augmentations,
    by estimating if the discriminator is currently too good i.e. overfitting.
    """

    def __init__(self):
        self.p = 0.0  # augmentation probability
        self.target_rt = (
            0.6  # target fraction of real images discriminator scores as real
        )
        self.rt_sum = 0.0
        self.batch_count = 0
        self.update_interval = 4

        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=1.0),
            K.RandomRotation(degrees=10, p=1.0),
            K.RandomAffine(
                degrees=0, translate=(0.125, 0.125), scale=(0.9, 1.1), p=1.0
            ),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        )

    def augment(self, images):
        if torch.rand(1).item() < self.p:
            return self.aug(images)
        else:
            return images

    def update_p(self, rt, batch_size):
        self.rt_sum += rt
        self.batch_count += 1

        if self.batch_count >= self.update_interval:
            avg_rt = self.rt_sum / self.batch_count
            adjust = 0.00006 * (avg_rt - self.target_rt) * self.update_interval
            adjust *= batch_size / 32.0

            self.p += adjust
            self.p = max(0.0, min(1.0, self.p))

            self.rt_sum = 0.0
            self.batch_count = 0
