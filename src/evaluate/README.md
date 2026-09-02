# src/evaluate

Runs a trained model against a held-out DataLoader (the untouched Testing set)
and computes the full metric suite from `src/metrics/`.

- `evaluate.py`:
  - `ModelEvaluator` — runs inference (collects true labels, predicted labels,
    and predicted probabilities via softmax), then computes Confusion Matrix,
    Accuracy, Recall/Precision/F1 (macro + per-class), and ROC-AUC (macro/OvR).
  - `EvaluationResult` — dataclass holding every computed metric, with a
    `.to_dict()` for JSON/MLflow-friendly export (numpy arrays -> plain lists).

Typical usage:
```python
model = EfficientNetB3Classifier(num_classes=4, pretrained=False)
load_model_checkpoint(model, "artifacts/checkpoints/best_model.pt")
evaluator = ModelEvaluator(model, test_loader, num_classes=4)
result = evaluator.evaluate()
print(result.to_dict())
```
