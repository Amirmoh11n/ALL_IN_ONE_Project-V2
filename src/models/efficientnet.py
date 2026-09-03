"""
EfficientNet classifier definitions (PyTorch / torchvision) for 4-class
brain tumor MRI classification.

Supports EfficientNet-B3 and EfficientNet-B4 (primary for V2). Only the final
Linear layer of the classifier head is replaced; the model's built-in Dropout
is left untouched. The backbone can optionally be frozen for feature-extraction
style training, controlled via configs/config.yaml (model.freeze_backbone).
"""

from __future__ import annotations

import logging
from typing import Literal

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    efficientnet_b3,
    efficientnet_b4,
)

logger = logging.getLogger(__name__)

ArchitectureName = Literal["efficientnet_b3", "efficientnet_b4"]

_ARCHITECTURE_REGISTRY = {
    "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
    "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
}


class EfficientNetClassifier(nn.Module):
    """EfficientNet backbone (B3 or B4) with a replaced classification head.

    Attributes:
        architecture: Name of the selected architecture.
        backbone: The full torchvision EfficientNet module (features + classifier).
    """

    def __init__(
        self,
        num_classes: int,
        architecture: ArchitectureName = "efficientnet_b4",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """
        Args:
            num_classes: Number of output classes (4 for this project).
            architecture: One of ``efficientnet_b3`` or ``efficientnet_b4``.
            pretrained: If True, load ImageNet-pretrained weights (requires
                network access on first run, cached afterwards). If False,
                random init (used in tests to avoid network calls).
            freeze_backbone: If True, freeze all backbone parameters (features)
                and only train the final classifier Linear layer
                (feature-extraction mode). If False (default), the entire
                network is fine-tuned.
        """
        super().__init__()

        architecture = architecture.lower()  # type: ignore[assignment]
        if architecture not in _ARCHITECTURE_REGISTRY:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Supported: {sorted(_ARCHITECTURE_REGISTRY)}"
            )

        builder, weights_enum = _ARCHITECTURE_REGISTRY[architecture]
        weights = weights_enum if pretrained else None
        self.architecture = architecture
        self.backbone = builder(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

        logger.info(
            "Built EfficientNetClassifier(architecture=%s, num_classes=%d, "
            "pretrained=%s, freeze_backbone=%s)",
            architecture,
            num_classes,
            pretrained,
            freeze_backbone,
        )

    def _freeze_backbone(self) -> None:
        """Freeze every parameter except the final classifier Linear layer."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        # classifier[0] is Dropout (no params); classifier[1] is the new Linear head.

    def forward(self, x):
        """Forward pass. Returns raw logits (no softmax — CrossEntropyLoss applies it).

        Args:
            x: Input batch of shape (batch_size, 3, H, W).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        return self.backbone(x)


# Backward-compatible alias used by older imports / tests.
class EfficientNetB3Classifier(EfficientNetClassifier):
    """Deprecated alias for EfficientNetClassifier(architecture='efficientnet_b3')."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            architecture="efficientnet_b3",
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
