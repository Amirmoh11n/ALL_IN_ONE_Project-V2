"""
EfficientNet-B3 model definition (PyTorch/torchvision), adapted for 4-class
brain tumor classification.

Only the final Linear layer of the classifier head is replaced; the model's
built-in Dropout (p=0.3, from torchvision's EfficientNet-B3) is left untouched
per project spec ("follow the model's built-in dropout"). The backbone can
optionally be frozen for feature-extraction-style training, controlled via
configs/config.yaml (model.freeze_backbone), defaulting to full fine-tuning.
"""

import logging

import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

logger = logging.getLogger(__name__)


class EfficientNetB3Classifier(nn.Module):
    """EfficientNet-B3 backbone with a replaced classification head.

    Attributes:
        backbone: The full torchvision EfficientNet-B3 module (features + classifier).
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """
        Args:
            num_classes: Number of output classes (4 for this project).
            pretrained: If True, load ImageNet-pretrained weights (requires network
                access on first run, cached afterwards). If False, random init
                (used in tests to avoid network calls).
            freeze_backbone: If True, freeze all backbone parameters (features)
                and only train the final classifier Linear layer
                (feature-extraction mode). If False (default), the entire
                network is fine-tuned.
        """
        super().__init__()

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b3(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

        logger.info(
            "Built EfficientNetB3Classifier(num_classes=%d, pretrained=%s, freeze_backbone=%s)",
            num_classes, pretrained, freeze_backbone,
        )

    def _freeze_backbone(self) -> None:
        """Freeze every parameter except the final classifier Linear layer."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        # classifier[0] is Dropout (no params); classifier[1] is the new Linear head,
        # which stays trainable.

    def forward(self, x):
        """Forward pass. Returns raw logits (no softmax -- CrossEntropyLoss applies it).

        Args:
            x: Input batch of shape (batch_size, 3, H, W).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        return self.backbone(x)
