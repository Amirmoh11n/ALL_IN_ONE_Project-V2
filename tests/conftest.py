"""
Shared pytest fixtures: builds a small synthetic dataset on disk (fake class
folders + tiny generated images) so data-pipeline tests don't need the real,
large Brain Tumor MRI Dataset or a network connection.
"""

import pytest
from PIL import Image

from src.data.classes import TumorClasses


@pytest.fixture
def synthetic_training_dir(tmp_path):
    """Create a fake Training/<class>/*.jpg tree with a few tiny images per class."""
    train_dir = tmp_path / "Training"
    images_per_class = 20
    for class_name in TumorClasses.NAMES:
        class_dir = train_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            img = Image.new("RGB", (32, 32), color=(i % 255, 0, 0))
            img.save(class_dir / f"{class_name}_{i}.jpg")
    return train_dir


@pytest.fixture
def synthetic_testing_dir(tmp_path):
    """Create a fake Testing/<class>/*.jpg tree with a few tiny images per class."""
    test_dir = tmp_path / "Testing"
    images_per_class = 5
    for class_name in TumorClasses.NAMES:
        class_dir = test_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            img = Image.new("RGB", (32, 32), color=(0, i % 255, 0))
            img.save(class_dir / f"{class_name}_{i}.jpg")
    return test_dir
