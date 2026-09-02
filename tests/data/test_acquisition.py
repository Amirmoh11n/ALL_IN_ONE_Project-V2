"""Tests for src/data/acquisition.py (DatasetAcquisition).

Only the "dataset already present -> skip download" and population-check paths
are exercised here. The real kagglehub download path requires network access
and credentials, which are out of scope for unit tests.
"""

from pathlib import Path

import pytest

from src.data.acquisition import DatasetAcquisition, DatasetNotAvailableError
from src.data.classes import TumorClasses


def _make_acquisition(tmp_path, train_name="Training", test_name="Testing"):
    return DatasetAcquisition(
        raw_dir=str(tmp_path),
        train_dir_name=train_name,
        test_dir_name=test_name,
        kaggle_dataset_slug="masoudnickparvar/brain-tumor-mri-dataset",
    )


def test_is_populated_true_when_all_class_folders_have_images(synthetic_training_dir):
    acq = _make_acquisition(synthetic_training_dir.parent)
    acq.train_dir = synthetic_training_dir
    assert acq._is_populated(synthetic_training_dir) is True


def test_is_populated_false_when_directory_missing(tmp_path):
    acq = _make_acquisition(tmp_path)
    assert acq._is_populated(tmp_path / "does_not_exist") is False


def test_is_populated_false_when_class_folder_empty(tmp_path):
    train_dir = tmp_path / "Training"
    for name in TumorClasses.NAMES:
        class_dir = train_dir / name
        class_dir.mkdir(parents=True, exist_ok=True)
    acq = _make_acquisition(tmp_path)
    assert acq._is_populated(train_dir) is False


def test_is_populated_false_when_one_class_missing(tmp_path):
    train_dir = tmp_path / "Training"
    for name in TumorClasses.NAMES[:-1]:
        class_dir = train_dir / name
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / "img.jpg").write_bytes(b"fake")
    acq = _make_acquisition(tmp_path)
    assert acq._is_populated(train_dir) is False


def test_ensure_available_skips_download_when_already_present(
    synthetic_training_dir, synthetic_testing_dir, monkeypatch
):
    raw = synthetic_training_dir.parent
    acq = DatasetAcquisition(
        raw_dir=str(raw),
        train_dir_name=synthetic_training_dir.name,
        test_dir_name=synthetic_testing_dir.name,
        kaggle_dataset_slug="masoudnickparvar/brain-tumor-mri-dataset",
    )

    def _fail_if_called():
        raise AssertionError("Download must not run when dataset is already present")

    monkeypatch.setattr(acq, "_download_from_kaggle", _fail_if_called)
    acq.ensure_available()


def test_ensure_available_raises_when_still_empty_after_download(tmp_path, monkeypatch):
    acq = _make_acquisition(tmp_path)

    def _noop_download():
        pass

    monkeypatch.setattr(acq, "_download_from_kaggle", _noop_download)

    with pytest.raises(DatasetNotAvailableError):
        acq.ensure_available()
