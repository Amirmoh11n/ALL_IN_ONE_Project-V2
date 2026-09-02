"""Tests for src/engine/trainer.py (EarlyStopping, compute_class_weights, Trainer).

Uses a tiny dummy CNN instead of the full EfficientNet-B3 so tests run in a
few seconds on CPU, and a synthetic dataset (no real download needed).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.augment import AugmentationFactory
from src.data.classes import TumorClasses
from src.data.dataset import BrainTumorDataset
from src.data.splitter import DatasetSplitter
from src.engine.trainer import EarlyStopping, Trainer, compute_class_weights
from src.utils.config_loader import ConfigLoader


class TinyDummyModel(nn.Module):
    """A tiny CNN standing in for EfficientNetB3Classifier in fast unit tests."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------- EarlyStopping ----------------

def test_early_stopping_triggers_after_patience_epochs_without_improvement():
    stopper = EarlyStopping(patience=2, mode="min")
    stopper.step(1.0)   # improvement (first score)
    stopper.step(1.1)   # no improvement, counter=1
    assert stopper.should_stop is False
    stopper.step(1.2)   # no improvement, counter=2 -> triggers
    assert stopper.should_stop is True


def test_early_stopping_resets_counter_on_improvement():
    stopper = EarlyStopping(patience=2, mode="min")
    stopper.step(1.0)
    stopper.step(1.1)  # counter=1
    stopper.step(0.5)  # improvement -> counter resets to 0
    assert stopper.counter == 0
    assert stopper.should_stop is False


# ---------------- compute_class_weights ----------------

def test_compute_class_weights_gives_larger_weight_to_minority_class():
    labels = [0] * 80 + [1] * 20  # class 0 majority, class 1 minority
    weights = compute_class_weights(labels, num_classes=2)
    assert weights[1] > weights[0]


def test_compute_class_weights_balanced_classes_are_equal():
    labels = [0] * 25 + [1] * 25 + [2] * 25 + [3] * 25
    weights = compute_class_weights(labels, num_classes=4)
    assert torch.allclose(weights, torch.full((4,), weights[0].item()))


# ---------------- Trainer (integration, tiny model + synthetic data) ----------------

def _build_config(tmp_path, use_class_weights=True, mlflow_enabled=False):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
training:
  epochs: 2
  learning_rate: 0.001
  use_class_weights: {str(use_class_weights).lower()}
  device: "cpu"
  early_stopping:
    patience: 5
    monitor: "val_loss"
  lr_scheduler:
    type: "reduce_on_plateau"
    factor: 0.5
    patience: 2
tracking:
  mlflow:
    enabled: {str(mlflow_enabled).lower()}
    tracking_uri: "sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name: "test_experiment"
""")
    return ConfigLoader(config_path)


def _build_loaders(synthetic_training_dir):
    split_result = DatasetSplitter(synthetic_training_dir, val_ratio=0.15, random_seed=42).split()
    transform = AugmentationFactory.build_eval_transforms(image_size=32)
    train_ds = BrainTumorDataset(split_result.train_samples, transform=transform)
    val_ds = BrainTumorDataset(split_result.val_samples, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    return train_loader, val_loader


def test_trainer_runs_and_returns_history(synthetic_training_dir, tmp_path):
    config = _build_config(tmp_path, use_class_weights=False)
    train_loader, val_loader = _build_loaders(synthetic_training_dir)
    model = TinyDummyModel(num_classes=TumorClasses.num_classes())

    trainer = Trainer(model, train_loader, val_loader, config, checkpoint_dir=tmp_path / "checkpoints")
    history = trainer.fit(num_epochs=2)

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["val_accuracy"]) == 2
    assert all(0.0 <= acc <= 1.0 for acc in history["val_accuracy"])


def test_trainer_saves_best_checkpoint(synthetic_training_dir, tmp_path):
    config = _build_config(tmp_path, use_class_weights=False)
    train_loader, val_loader = _build_loaders(synthetic_training_dir)
    model = TinyDummyModel(num_classes=TumorClasses.num_classes())
    checkpoint_dir = tmp_path / "checkpoints"

    trainer = Trainer(model, train_loader, val_loader, config, checkpoint_dir=checkpoint_dir)
    trainer.fit(num_epochs=2)

    checkpoint_path = checkpoint_dir / "best_model.pt"
    assert checkpoint_path.exists()

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "val_loss" in checkpoint


def test_trainer_with_class_weights_enabled_builds_weighted_loss(synthetic_training_dir, tmp_path):
    config = _build_config(tmp_path, use_class_weights=True)
    train_loader, val_loader = _build_loaders(synthetic_training_dir)
    model = TinyDummyModel(num_classes=TumorClasses.num_classes())

    trainer = Trainer(model, train_loader, val_loader, config, checkpoint_dir=tmp_path / "checkpoints")
    assert trainer.criterion.weight is not None
    assert trainer.criterion.weight.shape == (TumorClasses.num_classes(),)


def test_trainer_with_mlflow_enabled_logs_metrics(synthetic_training_dir, tmp_path):
    config = _build_config(tmp_path, use_class_weights=False, mlflow_enabled=True)
    train_loader, val_loader = _build_loaders(synthetic_training_dir)
    model = TinyDummyModel(num_classes=TumorClasses.num_classes())

    trainer = Trainer(model, train_loader, val_loader, config, checkpoint_dir=tmp_path / "checkpoints")
    trainer.fit(num_epochs=1)

    # mlflow writes a local sqlite DB at the configured tracking_uri
    assert (tmp_path / "mlflow.db").exists()
