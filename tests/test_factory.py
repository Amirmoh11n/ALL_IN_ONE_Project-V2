"""Model factory tests (random init, no ImageNet download)."""

from pathlib import Path

import torch

from src.models.factory import build_model
from src.utils.config_loader import ConfigLoader


def test_build_b4_random_init():
    config = ConfigLoader(Path("configs/config.yaml"))
    model = build_model(config, pretrained=False)
    x = torch.randn(2, 3, 32, 32)
    # EfficientNet needs a reasonable size; 32 is small but should still forward.
    y = model(torch.nn.functional.interpolate(x, size=(64, 64)))
    assert y.shape == (2, 4)
    assert model.architecture == "efficientnet_b4"
