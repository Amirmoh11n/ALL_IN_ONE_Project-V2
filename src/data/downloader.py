"""
Handles checking for the local presence of the Brain Tumor MRI Dataset and
downloading it via kagglehub if it is missing.

This is a new file (not present in the originally proposed tree). It was split
out from splitter.py to keep a single responsibility per module: downloading
is an I/O/network concern, splitting is a pure data-partitioning concern.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class DatasetDownloader:
    """Ensures the raw dataset exists locally, downloading it via kagglehub otherwise.

    Attributes:
        kaggle_handle: Kaggle dataset identifier, e.g.
            "masoudnickparvar/brain-tumor-mri-dataset".
        train_dir: Local path where the Training folder should end up
            (e.g. data/raw/Training).
        test_dir: Local path where the Testing folder should end up
            (e.g. data/raw/Testing).
    """

    def __init__(self, kaggle_handle: str, train_dir: Path, test_dir: Path) -> None:
        self.kaggle_handle = kaggle_handle
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)

    def is_dataset_present(self) -> bool:
        """Return True if both Training and Testing folders exist and contain images."""
        return self._folder_has_images(self.train_dir) and self._folder_has_images(self.test_dir)

    def ensure_dataset(self) -> None:
        """Download the dataset via kagglehub, unless it is already present locally.

        Raises:
            ImportError: If kagglehub is not installed.
            FileNotFoundError: If the downloaded cache does not contain the
                expected Training/Testing subfolders.
        """
        if self.is_dataset_present():
            logger.info(
                "Dataset already present at %s and %s; skipping download.",
                self.train_dir, self.test_dir,
            )
            return

        logger.info("Dataset not found locally. Downloading '%s' via kagglehub...", self.kaggle_handle)
        cache_path = self._download_via_kagglehub()
        self._copy_into_raw(cache_path)

    def _download_via_kagglehub(self) -> Path:
        """Trigger the kagglehub download and return the local cache path."""
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError(
                "kagglehub is required to auto-download the dataset. "
                "Install it with `pip install kagglehub`, or place the dataset manually "
                f"under {self.train_dir} and {self.test_dir}."
            ) from exc

        cache_path = Path(kagglehub.dataset_download(self.kaggle_handle))
        logger.info("Downloaded dataset to kagglehub cache: %s", cache_path)
        return cache_path

    def _copy_into_raw(self, cache_path: Path) -> None:
        """Copy the Training/Testing folders from the kagglehub cache into data/raw/."""
        source_train = self._find_subfolder(cache_path, "Training")
        source_test = self._find_subfolder(cache_path, "Testing")

        if source_train is None or source_test is None:
            raise FileNotFoundError(
                f"Could not locate Training/Testing folders inside kagglehub cache at "
                f"{cache_path}. The dataset layout may have changed and needs manual inspection."
            )

        self.train_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_train, self.train_dir, dirs_exist_ok=True)
        shutil.copytree(source_test, self.test_dir, dirs_exist_ok=True)
        logger.info("Copied dataset into %s and %s.", self.train_dir, self.test_dir)

    @classmethod
    def _folder_has_images(cls, folder: Path) -> bool:
        """Return True when the split contains all expected class folders with images."""
        if not folder.is_dir():
            return False
        expected = {"glioma", "meningioma", "notumor", "pituitary"}
        return all(
            (folder / name).is_dir()
            and any(p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES for p in (folder / name).rglob("*"))
            for name in expected
        )

    @staticmethod
    def _find_subfolder(root: Path, name: str) -> Optional[Path]:
        """Recursively find the first subfolder named `name` under root, if any."""
        matches = [p for p in root.rglob(name) if p.is_dir()]
        return matches[0] if matches else None
