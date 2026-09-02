"""Tests for src/utils/checkpoint.py (load_model_checkpoint)."""

import torch
import torch.nn as nn

from src.utils.checkpoint import load_model_checkpoint


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _save_fake_checkpoint(model, path, epoch=3, val_loss=0.1234):
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
        path,
    )


def test_load_model_checkpoint_restores_exact_weights(tmp_path):
    source_model = _TinyModel()
    checkpoint_path = tmp_path / "checkpoint.pt"
    _save_fake_checkpoint(source_model, checkpoint_path)

    target_model = _TinyModel()  # freshly initialized, different weights
    load_model_checkpoint(target_model, checkpoint_path)

    for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
        assert torch.equal(source_param, target_param)


def test_load_model_checkpoint_sets_eval_mode(tmp_path):
    source_model = _TinyModel()
    checkpoint_path = tmp_path / "checkpoint.pt"
    _save_fake_checkpoint(source_model, checkpoint_path)

    target_model = _TinyModel()
    target_model.train()  # explicitly put in train mode first
    load_model_checkpoint(target_model, checkpoint_path)

    assert target_model.training is False
