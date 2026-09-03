"""Splitter and patient-id extraction tests."""

from pathlib import Path

from src.data.splitter import DatasetSplitter, PatientIdExtractor, SplitResult


def test_patient_id_extractor():
    extractor = PatientIdExtractor()
    assert extractor.extract(Path("patient_0123_t1.png")) is not None
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
