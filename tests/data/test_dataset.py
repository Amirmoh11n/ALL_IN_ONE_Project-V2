"""Tests for src/data/dataset.py (BrainTumorDataset)."""

import torch

from src.data.augment import AugmentationFactory
from src.data.classes import TumorClasses
from src.data.dataset import BrainTumorDataset
from src.data.splitter import DatasetSplitter


def test_dataset_from_directory_finds_all_images(synthetic_testing_dir):
    dataset = BrainTumorDataset.from_directory(synthetic_testing_dir)
    # 5 images/class * 4 classes = 20
    assert len(dataset) == 20


def test_dataset_getitem_returns_image_and_label(synthetic_testing_dir):
    transform = AugmentationFactory.build_eval_transforms(image_size=300)
    dataset = BrainTumorDataset.from_directory(synthetic_testing_dir, transform=transform)

    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 300, 300)  # resized + normalized
    assert 0 <= label < TumorClasses.num_classes()


def test_dataset_built_from_splitter_samples(synthetic_training_dir):
    split_result = DatasetSplitter(synthetic_training_dir, val_ratio=0.15, random_seed=42).split()
    train_transform = AugmentationFactory.build_train_transforms(image_size=300)

    train_dataset = BrainTumorDataset(split_result.train_samples, transform=train_transform)
    assert len(train_dataset) == len(split_result.train_samples)

    image, label = train_dataset[0]
    assert image.shape == (3, 300, 300)
    assert isinstance(label, int)


def test_dataset_without_transform_returns_pil_image(synthetic_testing_dir):
    dataset = BrainTumorDataset.from_directory(synthetic_testing_dir, transform=None)
    image, _ = dataset[0]
    assert hasattr(image, "size")  # PIL.Image has .size, torch.Tensor does not
