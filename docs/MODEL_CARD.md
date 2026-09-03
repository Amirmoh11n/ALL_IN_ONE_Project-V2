# Model card — NeuraMRI V2

**Not a medical device.** This model is for research and education only. It must
not be used for diagnosis, triage, or clinical decision-making.

## Intended use

- Classify a **single 2D brain MRI slice** into one of four dataset labels:
  `glioma`, `meningioma`, `notumor`, `pituitary`.
- Users: ML students, researchers, demo operators on localhost / Docker.

## Out of scope

- 3D volumes, longitudinal studies, pediatric protocols not represented in the
  source dataset.
- External multi-hospital generalization (V2.0 roadmap only).
- Web-app Grad-CAM (artifacts-only in V2.0).

## Data

- Source: Brain Tumor MRI Dataset (Masoud Nickparvar), `Training` + `Testing`.
- The original `Testing` folder is never used for training or model selection.
- From `Training`, ~15% stratified hold-out validation.
- Split mode is **config-driven** (`data.split_strategy: auto`):
  patient-aware if reliable IDs can be parsed, otherwise **image-level
  stratified**. Document the mode used in `artifacts/evaluation/data_quality.json`.

## Training

- Architecture: EfficientNet-B4 (ImageNet pretrained), B3 fallback via config.
- Loss: class-weighted CrossEntropy (optional label smoothing).
- Mixup / CutMix default **off**.
- Checkpoint monitor: `val_f1_macro`.
- Multi-seed: 3 seeds (`42, 43, 44`), report mean ± std.

## Metrics (priority)

1. Confusion matrix
2. Recall (sensitivity)
3. F1-score (macro)
4. Precision
5. ROC-AUC (macro / OvR)
6. Accuracy
7. Specificity, PPV/NPV, calibration (temperature scaling + ECE)

Primary artifacts: `artifacts/evaluation/test_metrics.json`,
`confusion_matrix.png`, `roc_curves.png`.

## Limitations

- Dataset merge of public sources without guaranteed patient IDs → possible
  leakage if the same patient appears in train and test as different files.
- Scanner / hospital shift is untested.
- Calibration is estimated on the evaluation split, not an external set.

## Ethical / legal

Apache-2.0 code. Dataset licenses remain those of the original authors.
Do not deploy as a diagnostic product without regulatory review.
