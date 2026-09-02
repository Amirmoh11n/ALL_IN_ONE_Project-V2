"""
PyTorch Dataset for the Brain Tumor MRI Dataset.

Wraps a list of (image_path, class_index) samples -- produced either by
DatasetSplitter (train/val) or scanned directly from a class-subfoldered
directory such as the Testing folder -- and applies a transform pipeline
(augmentation for train, normalization-only for val/test; see augment.py).
"""

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from src.data.classes import TumorClasses

Sample = Tuple[Path, int]
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class BrainTumorDataset(Dataset):
    """Loads MRI images and their integer class labels for a given split."""

    def __init__(self, samples: List[Sample], transform: Optional[object] = None) -> None:
        """
        Args:
            samples: List of (image_path, class_index) tuples.
            transform: A torchvision transform pipeline (e.g. from AugmentationFactory).
                If None, images are returned as PIL Images (mainly useful for tests/inspection).
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, class_index = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, class_index

    @classmethod
    def from_directory(cls, directory: Path, transform: Optional[object] = None) -> "BrainTumorDataset":
        """Build a dataset directly from a class-subfoldered directory (e.g. the Testing folder).

        Args:
            directory: Root folder containing one subfolder per TumorClasses.NAMES entry.
            transform: Transform pipeline to apply per image.
        """
        directory = Path(directory)
        samples: List[Sample] = []
        for class_name in TumorClasses.NAMES:
            class_dir = directory / class_name
            class_index = TumorClasses.name_to_index(class_name)
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in _IMAGE_SUFFIXES:
                    samples.append((image_path, class_index))
        return cls(samples, transform=transform)
