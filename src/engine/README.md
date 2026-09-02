# src/engine

Training loop orchestration (renamed from `training` to `engine`).

- `trainer.py`:
  - `compute_class_weights(labels, num_classes)` — pure function, inverse-frequency
    class weights computed at **runtime** from the actual training split (no manual
    EDA step needed; toggled via `training.use_class_weights` in config).
  - `EarlyStopping` — stops training when `val_loss` hasn't improved for
    `training.early_stopping.patience` epochs.
  - `Trainer` — orchestrates the full loop: `CrossEntropyLoss` (optionally
    class-weighted), `Adam` optimizer, `ReduceLROnPlateau` LR scheduling, early
    stopping, best-checkpoint saving to `artifacts/checkpoints/best_model.pt`,
    and optional MLflow experiment tracking (`tracking.mlflow.*` in config).
  - **Per-epoch validation metrics**: each epoch, validation predictions are
    run through the same classes as `src/evaluate/evaluate.py`
    (`AccuracyMetric`, `RecallMetric`, `PrecisionMetric`, `F1ScoreMetric`,
    `ROCAUCMetric`, all macro-averaged) in a single forward pass, printed to
    the log line, returned in `fit()`'s history dict, and logged to MLflow —
    not just loss/accuracy.

All hyperparameters live in `configs/config.yaml` (`training`, `tracking.mlflow`).
