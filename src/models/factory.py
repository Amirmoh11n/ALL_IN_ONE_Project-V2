"""Model factory driven entirely by configuration."""
from src.models.efficientnet import EfficientNetB3Classifier
from src.utils.config_loader import ConfigLoader


def build_model(config: ConfigLoader, pretrained=None):
    architecture = str(config.get("model.architecture", "efficientnet_b3")).lower()
    if architecture != "efficientnet_b3":
        raise ValueError(f"Unsupported architecture: {architecture}")
    if pretrained is None:
        pretrained = bool(config.get("model.pretrained", True))
    return EfficientNetB3Classifier(
        num_classes=int(config.get("model.num_classes", len(config.get("data.class_names", [])) or 4)),
        pretrained=pretrained,
        freeze_backbone=bool(config.get("model.freeze_backbone", False)),
    )
