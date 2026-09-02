"""
Defines the on-the-fly data augmentation and normalization transform pipelines.

Augmentation is applied ONLY to the training split. Validation and test splits
are resized + normalized only (no augmentation), so evaluation metrics reflect
real inference conditions rather than an artificially easier/harder distribution.

Normalization defaults to ImageNet mean/std, matching the ImageNet-pretrained
EfficientNet-B3 backbone this project uses for transfer learning.
"""

from typing import Optional, Sequence

from torchvision import transforms

IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)


class AugmentationFactory:
    """Builds torchvision transform pipelines for training and evaluation splits."""

    @staticmethod
    def build_train_transforms(
        image_size: int,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        flip_p: float = 0.5,
        rotation_degrees: int = 15,
        brightness: float = 0.1,
        contrast: float = 0.1,
    ) -> transforms.Compose:
        """Return the augmentation + normalization pipeline used for the training split.

        Args:
            image_size: Target square size (e.g. 300 for EfficientNet-B3).
            mean: Per-channel normalization mean (defaults to ImageNet stats).
            std: Per-channel normalization std (defaults to ImageNet stats).
            flip_p: Probability of a random horizontal flip.
            rotation_degrees: Max absolute degrees for random rotation.
            brightness: ColorJitter brightness factor.
            contrast: ColorJitter contrast factor.
        """
        mean = mean or IMAGENET_MEAN
        std = std or IMAGENET_STD
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.RandomRotation(degrees=rotation_degrees),
            transforms.ColorJitter(brightness=brightness, contrast=contrast),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    @staticmethod
    def build_eval_transforms(
        image_size: int,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ) -> transforms.Compose:
        """Return the resize + normalization-only pipeline used for validation/test splits.

        Args:
            image_size: Target square size (e.g. 300 for EfficientNet-B3).
            mean: Per-channel normalization mean (defaults to ImageNet stats).
            std: Per-channel normalization std (defaults to ImageNet stats).
        """
        mean = mean or IMAGENET_MEAN
        std = std or IMAGENET_STD
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
