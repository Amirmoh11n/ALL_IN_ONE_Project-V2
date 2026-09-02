"""
Ensures the raw Brain Tumor MRI Dataset is available on local disk before the
rest of the data pipeline runs.

Responsibility (single, per SRP): check whether `data/raw/Training` and
`data/raw/Testing` already contain the expected class subfolders with images;
if not, download the dataset from Kaggle (via `kagglehub`) and place it at
the configured `raw_dir`.

This module does NOT store or manage Kaggle credentials itself. It relies on
the standard Kaggle authentication mechanisms:
    - a `~/.kaggle/kaggle.json` file, or
    - the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables.
See data/README.md for setup instructions.
"""

import logging
import shutil
from pathlib import Path
from typing import List

from src.data.classes import TumorClasses

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class DatasetNotAvailableError(RuntimeError):
    """Raised when the dataset is missing locally and cannot be downloaded
    (e.g. missing Kaggle credentials)."""


class DatasetAcquisition:
    """Guarantees that the raw dataset exists locally at the expected path,
    downloading it from Kaggle on demand if it does not.

    Args:
        raw_dir: root directory where `Training/` and `Testing/` should live
            (e.g. "data/raw").
        train_dir_name: name of the training subfolder (e.g. "Training").
        test_dir_name: name of the testing subfolder (e.g. "Testing").
        kaggle_dataset_slug: Kaggle dataset identifier, e.g.
            "masoudnickparvar/brain-tumor-mri-dataset".
    """

    def __init__(
        self,
        raw_dir: str,
        train_dir_name: str,
        test_dir_name: str,
        kaggle_dataset_slug: str,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.train_dir = self.raw_dir / train_dir_name
        self.test_dir = self.raw_dir / test_dir_name
        self.kaggle_dataset_slug = kaggle_dataset_slug

    def ensure_available(self) -> None:
        """Make sure Training/ and Testing/ exist and are populated.

        Downloads the dataset from Kaggle only if either folder is missing
        or empty of the expected class subfolders/images.
        """
        if self._is_populated(self.train_dir) and self._is_populated(self.test_dir):
            logger.info(
                "Dataset already present at '%s' and '%s'; skipping download.",
                self.train_dir,
                self.test_dir,
            )
            return

        logger.info(
            "Dataset missing or incomplete under '%s'; downloading from Kaggle "
            "dataset '%s'.",
            self.raw_dir,
            self.kaggle_dataset_slug,
        )
        self._download_from_kaggle()

        if not (self._is_populated(self.train_dir) and self._is_populated(self.test_dir)):
            raise DatasetNotAvailableError(
                f"Download completed but expected folders '{self.train_dir}' and "
                f"'{self.test_dir}' are still missing/empty. Check the dataset's "
                "internal folder layout matches 'Training/<class>/*.jpg' and "
                "'Testing/<class>/*.jpg'."
            )

    def _is_populated(self, split_dir: Path) -> bool:
        """A split directory counts as populated if it exists and every
        expected class subfolder contains at least one image file."""
        if not split_dir.is_dir():
            return False
        for class_name in TumorClasses.NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                return False
            if not self._list_images(class_dir):
                return False
        return True

    @staticmethod
    def _list_images(directory: Path) -> List[Path]:
        return [
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def _download_from_kaggle(self) -> None:
        """Downloads the dataset via kagglehub and copies its Training/Testing
        subfolders into `self.raw_dir`.

        Raises:
            DatasetNotAvailableError: if kagglehub is unavailable or Kaggle
                authentication is not configured.
        """
        try:
            import kagglehub
        except ImportError as exc:
            raise DatasetNotAvailableError(
                "The 'kagglehub' package is required to auto-download the dataset. "
                "Install it (e.g. `pip install kagglehub`) or place the dataset "
                f"manually under '{self.raw_dir}'. See data/README.md."
            ) from exc

        try:
            downloaded_path = Path(kagglehub.dataset_download(self.kaggle_dataset_slug))
        except Exception as exc:
            raise DatasetNotAvailableError(
                "Failed to download the dataset from Kaggle. This usually means "
                "Kaggle authentication is not configured. Set up "
                "'~/.kaggle/kaggle.json' or the KAGGLE_USERNAME/KAGGLE_KEY "
                "environment variables — see data/README.md for step-by-step "
                f"instructions. Original error: {exc}"
            ) from exc

        self._copy_split(downloaded_path, self.train_dir.name, self.train_dir)
        self._copy_split(downloaded_path, self.test_dir.name, self.test_dir)

    @staticmethod
    def _copy_split(downloaded_root: Path, split_dir_name: str, destination: Path) -> None:
        """Locates `split_dir_name` somewhere under `downloaded_root` (the
        Kaggle dataset is not always at the top level) and copies it to
        `destination`."""
        candidates = [p for p in downloaded_root.rglob(split_dir_name) if p.is_dir()]
        if not candidates:
            raise DatasetNotAvailableError(
                f"Could not locate a '{split_dir_name}' folder inside the "
                f"downloaded dataset at '{downloaded_root}'."
            )
        source = candidates[0]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        logger.info("Copied '%s' -> '%s'.", source, destination)
