# src/engine

Training loop orchestration (renamed from `training` to `engine`).

- `trainer.py`:
  - `compute_class_weights(labels, num_classes)` — pure function, inverse-frequency
    class weights computed at **runtime** from the actual training split (no manual
    EDA step needed; toggled via `training.use_class_weights` in config).
  - `EarlyStopping` — stops training when the configured monitor metric
    (default `val_f1_macro`) has not improved for
    `training.early_stopping.patience` epochs. Mode (`min`/`max`) is derived
    automatically from the metric name.
  - `Trainer` — orchestrates the full loop: `CrossEntropyLoss` (optionally
    class-weighted + label smoothing), `Adam` optimizer, `ReduceLROnPlateau`
    LR scheduling aligned with the monitor metric, early stopping, best-
    checkpoint saving to `artifacts/checkpoints/best_model.pt`, optional
    gradient clipping, AMP, and MLflow experiment tracking
    (`tracking.mlflow.*` in config).
  - **Per-epoch validation metrics**: each epoch, validation predictions are
    run through the same classes as `src/evaluate/evaluate.py`
    (`AccuracyMetric`, `RecallMetric`, `PrecisionMetric`, `F1ScoreMetric`,
    `ROCAUCMetric`, all macro-averaged) in a single forward pass, printed to
    the log line, returned in `fit()`'s history dict, and logged to MLflow.

All hyperparameters live in `configs/config.yaml` (`training`, `tracking.mlflow`).
