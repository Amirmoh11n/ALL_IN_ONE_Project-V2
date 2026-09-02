"""Tests for src/data/augment.py (AugmentationFactory).

Verifies that train transforms include augmentation ops while eval transforms
are resize + normalize only, and that output tensors have the expected shape
and dtype after applying either pipeline to a synthetic PIL image.
"""

import torch
from PIL import Image

from src.data.augment import AugmentationFactory, IMAGENET_MEAN, IMAGENET_STD


def _make_dummy_image(size: int = 64) -> Image.Image:
    """Create a small solid-color RGB image for transform smoke-tests."""
    return Image.new("RGB", (size, size), color=(120, 80, 40))


def test_build_train_transforms_returns_compose_with_augmentation():
    transform = AugmentationFactory.build_train_transforms(image_size=64)
    type_names = [type(t).__name__ for t in transform.transforms]

    assert "Resize" in type_names
    assert "RandomHorizontalFlip" in type_names
    assert "RandomRotation" in type_names
    assert "ColorJitter" in type_names
    assert "ToTensor" in type_names
    assert "Normalize" in type_names


def test_build_eval_transforms_has_no_augmentation():
    transform = AugmentationFactory.build_eval_transforms(image_size=64)
    type_names = [type(t).__name__ for t in transform.transforms]

    assert "Resize" in type_names
    assert "ToTensor" in type_names
    assert "Normalize" in type_names
    # Eval pipeline must never apply stochastic augmentation.
    assert "RandomHorizontalFlip" not in type_names
    assert "RandomRotation" not in type_names
    assert "ColorJitter" not in type_names


def test_train_transform_output_shape_and_dtype():
    transform = AugmentationFactory.build_train_transforms(image_size=48)
    image = _make_dummy_image(size=32)
    tensor = transform(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 48, 48)
    assert tensor.dtype == torch.float32


def test_eval_transform_output_shape_and_dtype():
    transform = AugmentationFactory.build_eval_transforms(image_size=48)
    image = _make_dummy_image(size=32)
    tensor = transform(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 48, 48)
    assert tensor.dtype == torch.float32


def test_custom_mean_std_are_applied():
    custom_mean = [0.1, 0.2, 0.3]
    custom_std = [0.4, 0.5, 0.6]
    transform = AugmentationFactory.build_eval_transforms(
        image_size=32, mean=custom_mean, std=custom_std
    )
    normalize = transform.transforms[-1]
    assert list(normalize.mean) == custom_mean
    assert list(normalize.std) == custom_std


def test_defaults_use_imagenet_stats():
    transform = AugmentationFactory.build_eval_transforms(image_size=32)
    normalize = transform.transforms[-1]
    assert tuple(normalize.mean) == tuple(IMAGENET_MEAN)
    assert tuple(normalize.std) == tuple(IMAGENET_STD)


def test_train_transform_respects_flip_probability_param():
    transform = AugmentationFactory.build_train_transforms(
        image_size=32, flip_p=0.0
    )
    flip = next(t for t in transform.transforms if type(t).__name__ == "RandomHorizontalFlip")
    assert flip.p == 0.0
