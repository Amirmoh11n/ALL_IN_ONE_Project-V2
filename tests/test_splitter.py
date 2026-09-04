"""Splitter and patient-id extraction tests."""

from pathlib import Path

import pytest

from src.data.splitter import DatasetSplitter, PatientIdExtractor, SplitResult


def test_patient_id_extractor():
    extractor = PatientIdExtractor()
    assert extractor.extract(Path("patient_0123_t1.png")) is not None
    assert extractor.extract(Path("subj_88_flair.jpg")) is not None
    assert extractor.extract(Path("Te-gl_0010.jpg")) is None


def test_image_level_split_stratified(tmp_path):
    samples = []
    for label in range(4):
        for i in range(20):
            samples.append((tmp_path / f"c{label}_{i}.jpg", label))
    splitter = DatasetSplitter(
        train_dir=tmp_path,
        val_ratio=0.15,
        random_seed=42,
        strategy="stratified_image_level",
        samples=samples,
    )
    result = splitter.split()
    assert isinstance(result, SplitResult)
    assert result.mode == "stratified_image_level"
    assert len(result.train_samples) + len(result.val_samples) == 80
    val_labels = [y for _, y in result.val_samples]
    assert len(set(val_labels)) == 4


def test_auto_falls_back_without_patient_ids(tmp_path):
    samples = [(tmp_path / f"Te-gl_{i:04d}.jpg", i % 4) for i in range(40)]
    result = DatasetSplitter(tmp_path, 0.15, 0, "auto", samples).split()
    assert result.mode == "stratified_image_level"


def test_patient_aware_when_ids_group_images(tmp_path):
    samples = []
    for pid in range(20):
        label = pid % 4
        for slice_i in range(3):
            samples.append((tmp_path / f"patient_{pid:03d}_s{slice_i}.png", label))
    result = DatasetSplitter(tmp_path, 0.2, 1, "patient_aware", samples).split()
    assert result.mode == "patient_aware"
    train_ids = {p.stem.split("_s")[0] for p, _ in result.train_samples}
    val_ids = {p.stem.split("_s")[0] for p, _ in result.val_samples}
    assert train_ids.isdisjoint(val_ids)


def test_invalid_ratio_raises(tmp_path):
    with pytest.raises(ValueError):
        DatasetSplitter(tmp_path, 0.0, 0)
