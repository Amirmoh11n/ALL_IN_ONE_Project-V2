"""Model factory tests (random init, no ImageNet download)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.models.factory import build_model
from src.utils.config_loader import ConfigLoader


def test_build_b4_random_init():
    config = ConfigLoader(Path("configs/config.yaml"))
    model = build_model(config, pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 4)
    assert model.architecture == "efficientnet_b4"


def test_b3_fallback():
    config = ConfigLoader(Path("configs/config.yaml"))
    config._config.setdefault("model", {})
    config._config["model"]["architecture"] = "efficientnet_b3"
    model = build_model(config, pretrained=False)
    assert model.architecture == "efficientnet_b3"


def test_unsupported_architecture_raises():
    config = ConfigLoader(Path("configs/config.yaml"))
    config._config.setdefault("model", {})
    config._config["model"]["architecture"] = "resnet50"
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_model(config, pretrained=False)
