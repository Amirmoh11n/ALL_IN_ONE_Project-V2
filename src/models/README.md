# src/models

Model architecture definitions.

- `efficientnet.py` — `EfficientNetB3Classifier`: torchvision's EfficientNet-B3
  (ImageNet-pretrained) with its final `Linear` layer replaced for 4-class output.
  The model's built-in `Dropout(p=0.3)` is left untouched (per project spec).
  `freeze_backbone` (config-driven, default `false`) toggles between full
  fine-tuning and feature-extraction (frozen backbone, head-only training).
