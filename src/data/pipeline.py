"""End-to-end dataset preparation for train/validation/test DataLoaders."""
import logging
import random
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.augment import AugmentationFactory
from src.data.dataset import BrainTumorDataset
from src.data.downloader import DatasetDownloader
from src.data.splitter import DatasetSplitter
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DataPipeline:
    """Download/prepare data, split Training, and create all DataLoaders."""

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config

    def prepare(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        raw_dir = self.config.resolve_path("data.raw_dir", "data/raw")
        train_dir = raw_dir / self.config.get("data.train_dir_name", "Training")
        test_dir = raw_dir / self.config.get("data.test_dir_name", "Testing")

        DatasetDownloader(
            kaggle_handle=self.config.get(
                "data.kaggle.dataset_slug", "masoudnickparvar/brain-tumor-mri-dataset"
            ),
            train_dir=train_dir,
            test_dir=test_dir,
        ).ensure_dataset()

        seed = int(self.config.get("data.seed", 42))
        seed_everything(seed)

        split_result = DatasetSplitter(
            train_dir=train_dir,
            val_ratio=float(self.config.get("data.val_split", 0.15)),
            random_seed=seed,
        ).split()

        image_size = int(self.config.get("data.image_size", 300))
        mean = self.config.get("data.normalization.mean")
        std = self.config.get("data.normalization.std")
        train_transform = AugmentationFactory.build_train_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            flip_p=float(self.config.get("data.augmentation.random_horizontal_flip_p", 0.5)),
            rotation_degrees=float(self.config.get("data.augmentation.random_rotation_degrees", 15)),
            brightness=float(self.config.get("data.augmentation.color_jitter.brightness", 0.1)),
            contrast=float(self.config.get("data.augmentation.color_jitter.contrast", 0.1)),
        )
        eval_transform = AugmentationFactory.build_eval_transforms(image_size, mean, std)

        train_ds = BrainTumorDataset(split_result.train_samples, train_transform)
        val_ds = BrainTumorDataset(split_result.val_samples, eval_transform)
        test_ds = BrainTumorDataset.from_directory(test_dir, eval_transform)

        batch_size = int(self.config.get("data.dataloader.batch_size", 32))
        num_workers = int(self.config.get("data.dataloader.num_workers", 4))
        pin_memory = bool(self.config.get("data.dataloader.pin_memory", torch.cuda.is_available()))
        common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
        if num_workers > 0:
            common["persistent_workers"] = bool(
                self.config.get("data.dataloader.persistent_workers", True)
            )

        train_loader = DataLoader(
            train_ds,
            shuffle=bool(self.config.get("data.dataloader.shuffle_train", True)),
            **common,
        )
        val_loader = DataLoader(val_ds, shuffle=False, **common)
        test_loader = DataLoader(test_ds, shuffle=False, **common)

        logger.info("DataPipeline ready: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))
        return train_loader, val_loader, test_loader
