# artifacts

Generated, non-source artifacts produced by the pipeline. Not hand-written; not meant to hold logic.

- `checkpoints/` — saved model checkpoints (used for inference and resuming training).
- `exports/` — exported model versions (Web/Cloud, Mobile, GPU-optimized, ONNX).
- `mlruns/` — MLflow experiment tracking store.

This folder is dataset/experiment output, not code — contents are gitignored except for structural placeholders.
