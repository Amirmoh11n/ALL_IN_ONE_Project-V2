# src/models

Model architecture definitions.

- `efficientnet.py` — `EfficientNetClassifier`: torchvision EfficientNet-B3 / B4
  (ImageNet-pretrained) with the final `Linear` layer replaced for 4-class output.
  The model's built-in Dropout is left untouched. `freeze_backbone` (config-driven,
  default `false`) toggles between full fine-tuning and feature-extraction
  (frozen backbone, head-only training).

  V2 default architecture is **EfficientNet-B4**. B3 remains available as a
  config fallback (`model.architecture: efficientnet_b3`).

- `factory.py` — `build_model(config)` constructs the classifier purely from
  configuration; no hard-coded architecture in call sites.
