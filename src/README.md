# src

Main source package for the Brain Tumor Classification system. Organized by responsibility, one subpackage per
concern (data, models, training, evaluate, export, inference, metrics, utils), following high-cohesion/low-coupling
and single-responsibility principles.

Subpackages:
- `data/` — dataset loading, splitting, augmentation, class definitions.
- `models/` — model architecture definitions (EfficientNet-B3).
- `training/` — training loop, early stopping, LR scheduling, MLflow logging.
- `evaluate/` — evaluation pipeline against the held-out Testing set.
- `metrics/` — individual metric implementations (Confusion Matrix, Recall, F1, Precision, ROC-AUC, Accuracy).
- `export/` — model export to Web/Cloud, Mobile, GPU-optimized, and ONNX formats.
- `inference/` — inference pipeline used by the web application backend.
- `utils/` — shared helpers (config loading, logging setup).
