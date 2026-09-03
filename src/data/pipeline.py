"""End-to-end dataset preparation for train/validation/test DataLoaders."""

from __future__ import annotations

import json
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
from src.data.quality import ImageQualityFilter
from src.data.splitter import DatasetSplitter, Sample
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
    """Download/prepare data, quality-filter Training, split, and create DataLoaders."""

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config
        self.split_mode: str = "stratified_image_level"

    def prepare(self, seed: int | None = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
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

        used_seed = int(seed if seed is not None else self.config.get("data.seed", 42))
        seed_everything(used_seed)

        splitter = DatasetSplitter(
            train_dir=train_dir,
            val_ratio=float(self.config.get("data.val_split", 0.15)),
            random_seed=used_seed,
            strategy=str(self.config.get("data.split_strategy", "auto")),
        )
        collected = splitter._collect_samples()
        quality = ImageQualityFilter(
            drop_corrupt=bool(self.config.get("data.quality.drop_corrupt", True)),
            drop_duplicates=bool(self.config.get("data.quality.drop_duplicates", True)),
            min_width=int(self.config.get("data.quality.min_width", 32)),
            min_height=int(self.config.get("data.quality.min_height", 32)),
        ).filter(collected)
        splitter.samples = quality.kept
        split_result = splitter.split()
        self.split_mode = split_result.mode

        report_path = self.config.resolve_path(
            "evaluation.output_dir", "artifacts/evaluation"
        ) / "data_quality.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({**quality.to_dict(), "split_mode": self.split_mode, "seed": used_seed}, indent=2),
            encoding="utf-8",
        )

        image_size = int(self.config.get("data.image_size", 380))
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

        batch_size = int(self.config.get("data.dataloader.batch_size", 16))
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

        logger.info(
            "DataPipeline ready: train=%d val=%d test=%d split_mode=%s",
            len(train_ds),
            len(val_ds),
            len(test_ds),
            self.split_mode,
        )
        return train_loader, val_loader, test_loader
