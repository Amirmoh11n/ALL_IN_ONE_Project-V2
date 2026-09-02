# src/metrics

One file per evaluation metric, in the project's priority order. Each is a thin
class wrapping scikit-learn's implementation (reuse-first: sklearn's metrics are
mature and well-tested; wrapping them keeps a consistent project-internal
interface instead of scattering direct sklearn calls everywhere).

1. `confusion_matrix.py` — `ConfusionMatrixMetric` — full confusion matrix.
2. `recall.py` — `RecallMetric` — `.compute()` (macro-averaged) and `.per_class()`.
3. `f1_score.py` — `F1ScoreMetric` — `.compute()` (macro-averaged) and `.per_class()`.
4. `precision.py` — `PrecisionMetric` — `.compute()` (macro-averaged) and `.per_class()`.
5. `roc_auc.py` — `ROCAUCMetric` — macro, one-vs-rest. **Needs predicted
   probabilities** (e.g. softmax output), not just predicted labels — this is
   the one metric that measures ranking quality across all thresholds, not
   just the final argmax decision.
6. `accuracy.py` — `AccuracyMetric` — overall accuracy.

All averaged metrics default to `average="macro"` (each class weighted equally),
appropriate for a 4-class medical dataset where minority-class performance
matters as much as majority-class performance.
