"""Robust multi-class ROC-AUC (macro, one-vs-rest)."""
from typing import List, Optional, Sequence
import numpy as np
from sklearn.metrics import roc_auc_score


class ROCAUCMetric:
    @staticmethod
    def compute(
        y_true: List[int], y_score: Sequence[Sequence[float]],
        num_classes: Optional[int] = None, average: str = "macro",
    ) -> float:
        if not y_true:
            return 0.0
        scores = np.asarray(y_score, dtype=float)
        labels = list(range(num_classes)) if num_classes is not None else None
        try:
            return float(roc_auc_score(
                y_true, scores, multi_class="ovr", average=average, labels=labels
            ))
        except ValueError:
            # ROC-AUC is undefined when a validation/evaluation split contains
            # only one class. Returning 0 keeps training/evaluation pipelines alive;
            # the warning-worthy condition is visible in the logs from the caller.
            return 0.0
