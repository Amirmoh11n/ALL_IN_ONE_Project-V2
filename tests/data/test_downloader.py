"""Tests for src/data/downloader.py (DatasetDownloader).

Only the "dataset already present -> skip download" path is tested here.
The actual kagglehub download path requires network access to kaggle.com,
which is out of scope for unit tests (it belongs in a manual/integration check).
"""

from src.data.downloader import DatasetDownloader


def test_is_dataset_present_true_when_images_exist(synthetic_training_dir, synthetic_testing_dir):
    downloader = DatasetDownloader(
        kaggle_handle="masoudnickparvar/brain-tumor-mri-dataset",
        train_dir=synthetic_training_dir,
        test_dir=synthetic_testing_dir,
    )
    assert downloader.is_dataset_present() is True


def test_is_dataset_present_false_when_missing(tmp_path):
    downloader = DatasetDownloader(
        kaggle_handle="masoudnickparvar/brain-tumor-mri-dataset",
        train_dir=tmp_path / "does_not_exist_train",
        test_dir=tmp_path / "does_not_exist_test",
    )
    assert downloader.is_dataset_present() is False


def test_ensure_dataset_skips_download_when_already_present(
    synthetic_training_dir, synthetic_testing_dir, monkeypatch
):
    downloader = DatasetDownloader(
        kaggle_handle="masoudnickparvar/brain-tumor-mri-dataset",
        train_dir=synthetic_training_dir,
        test_dir=synthetic_testing_dir,
    )

    def _fail_if_called():
        raise AssertionError("Download should not be triggered when data already exists")

    monkeypatch.setattr(downloader, "_download_via_kagglehub", lambda: _fail_if_called())
    downloader.ensure_dataset()  # should return early, not call the patched method
