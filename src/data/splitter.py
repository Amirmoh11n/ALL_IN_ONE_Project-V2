"""
Splits the Training folder into a stratified train / validation file list.

Design decision: patient-level splitting is NOT used. The Brain Tumor MRI
Dataset (Nickparvar) merges three source datasets (figshare, SARTAJ, Br35H)
without patient identifiers, so a true patient-level split is not feasible.
An image-level *stratified* split is used instead (same class proportions in
both train and validation subsets), and this limitation is documented here
rather than silently ignored.
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from src.data.classes import TumorClasses

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

Sample = Tuple[Path, int]


@dataclass
class SplitResult:
    """Container for the outcome of a train/validation split.

    Attributes:
        train_samples: List of (image_path, class_index) for training.
        val_samples: List of (image_path, class_index) for validation / hyperparameter tuning.
    """

    train_samples: List[Sample] = field(default_factory=list)
    val_samples: List[Sample] = field(default_factory=list)


class DatasetSplitter:
    """Performs a stratified image-level split of the Training folder.

    Attributes:
        train_dir: Path to the raw Training folder (contains one subfolder per class).
        val_ratio: Fraction of each class reserved for validation (e.g. 0.15).
        random_seed: Seed for shuffling, so the split is reproducible.
    """

    def __init__(self, train_dir: Path, val_ratio: float, random_seed: int = 42) -> None:
        if not 0.0 < val_ratio < 1.0:
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
        self.train_dir = Path(train_dir)
        self.val_ratio = val_ratio
        self.random_seed = random_seed

    def split(self) -> SplitResult:
        """Perform a per-class shuffle + split, holding out val_ratio for validation.

        Returns:
            A SplitResult with disjoint, shuffled train_samples and val_samples.
        """
        rng = random.Random(self.random_seed)
        samples_by_class = self._collect_samples_by_class()

        result = SplitResult()
        for class_index, paths in samples_by_class.items():
            shuffled = paths.copy()
            rng.shuffle(shuffled)
            n_val = round(len(shuffled) * self.val_ratio)
            result.val_samples.extend((p, class_index) for p in shuffled[:n_val])
            result.train_samples.extend((p, class_index) for p in shuffled[n_val:])

        # Shuffle again across classes so batches aren't ordered class-by-class,
        # which could otherwise bias/confuse the model during training.
        rng.shuffle(result.train_samples)
        rng.shuffle(result.val_samples)

        logger.info(
            "Split complete: %d train samples, %d val samples (val_ratio=%.2f).",
            len(result.train_samples), len(result.val_samples), self.val_ratio,
        )
        return result

    def _collect_samples_by_class(self) -> Dict[int, List[Path]]:
        """Scan train_dir/<class_name>/* and group image paths by class index."""
        samples: Dict[int, List[Path]] = {}
        for class_name in TumorClasses.NAMES:
            class_dir = self.train_dir / class_name
            class_index = TumorClasses.name_to_index(class_name)
            image_paths = sorted(
                p for p in class_dir.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES
            )
            if not image_paths:
                logger.warning("No images found for class '%s' in %s", class_name, class_dir)
            samples[class_index] = image_paths
        return samples
