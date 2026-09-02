"""Tests for src/data/splitter.py (DatasetSplitter)."""

from src.data.classes import TumorClasses
from src.data.splitter import DatasetSplitter


def test_split_ratio_is_approximately_correct(synthetic_training_dir):
    splitter = DatasetSplitter(train_dir=synthetic_training_dir, val_ratio=0.15, random_seed=42)
    result = splitter.split()

    total = len(result.train_samples) + len(result.val_samples)
    # 20 images/class * 4 classes = 80 total; 15% -> 12 val, 68 train (rounded per class)
    assert total == 80
    assert len(result.val_samples) == 12
    assert len(result.train_samples) == 68


def test_split_is_stratified_per_class(synthetic_training_dir):
    splitter = DatasetSplitter(train_dir=synthetic_training_dir, val_ratio=0.15, random_seed=42)
    result = splitter.split()

    val_counts = {idx: 0 for idx in range(TumorClasses.num_classes())}
    for _, class_index in result.val_samples:
        val_counts[class_index] += 1

    # Each class should be represented in validation, not just one class dumped there.
    assert all(count == 3 for count in val_counts.values())  # 15% of 20 = 3


def test_train_and_val_samples_are_disjoint(synthetic_training_dir):
    splitter = DatasetSplitter(train_dir=synthetic_training_dir, val_ratio=0.15, random_seed=42)
    result = splitter.split()

    train_paths = {p for p, _ in result.train_samples}
    val_paths = {p for p, _ in result.val_samples}
    assert train_paths.isdisjoint(val_paths)


def test_split_is_shuffled_not_grouped_by_class(synthetic_training_dir):
    splitter = DatasetSplitter(train_dir=synthetic_training_dir, val_ratio=0.15, random_seed=42)
    result = splitter.split()

    class_sequence = [class_index for _, class_index in result.train_samples]
    # If shuffling worked, the sequence should NOT be sorted by class (all 0s, then all 1s, etc).
    assert class_sequence != sorted(class_sequence)


def test_split_is_reproducible_with_same_seed(synthetic_training_dir):
    result_a = DatasetSplitter(synthetic_training_dir, val_ratio=0.15, random_seed=123).split()
    result_b = DatasetSplitter(synthetic_training_dir, val_ratio=0.15, random_seed=123).split()
    assert result_a.train_samples == result_b.train_samples
    assert result_a.val_samples == result_b.val_samples
