"""Model factory driven entirely by configuration.

Supports EfficientNet-B4 (V2 default) and EfficientNet-B3 (fallback).
"""

from __future__ import annotations

from src.models.efficientnet import EfficientNetClassifier
from src.utils.config_loader import ConfigLoader


def build_model(config: ConfigLoader, pretrained: bool | None = None):
    """Build a classifier according to ``model.*`` settings in the config.

    Args:
        config: Loaded project configuration.
        pretrained: Optional override for ``model.pretrained``. When None the
            value is read from config (default True). Pass False in unit tests
            to avoid network downloads.

    Returns:
        An ``EfficientNetClassifier`` instance.

    Raises:
        ValueError: If the requested architecture is not supported.
    """
    architecture = str(config.get("model.architecture", "efficientnet_b4")).lower()
    supported = {"efficientnet_b3", "efficientnet_b4"}
    if architecture not in supported:
        raise ValueError(
            f"Unsupported architecture: {architecture}. Supported: {sorted(supported)}"
        )

    if pretrained is None:
        pretrained = bool(config.get("model.pretrained", True))

    num_classes = int(
        config.get(
            "model.num_classes",
            len(config.get("data.class_names", [])) or 4,
        )
    )

    return EfficientNetClassifier(
        num_classes=num_classes,
        architecture=architecture,  # type: ignore[arg-type]
        pretrained=pretrained,
        freeze_backbone=bool(config.get("model.freeze_backbone", False)),
    )
