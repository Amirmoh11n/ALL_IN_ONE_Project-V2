"""Specificity (true-negative rate) per class and macro-averaged."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.metrics.confusion_matrix import ConfusionMatrixMetric


class SpecificityMetric:
    """Computes specificity = TN / (TN + FP) from the confusion matrix."""

    @staticmethod
    def per_class(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
        matrix = ConfusionMatrixMetric.compute(y_true, y_pred, num_classes=num_classes)
        totals = matrix.sum()
        scores = np.zeros(num_classes, dtype=np.float64)
        for i in range(num_classes):
            tp = matrix[i, i]
            fn = matrix[i, :].sum() - tp
            fp = matrix[:, i].sum() - tp
            tn = totals - tp - fn - fp
            denom = tn + fp
            scores[i] = float(tn / denom) if denom else 0.0
        return scores

    @staticmethod
    def compute(
        y_true: List[int],
        y_pred: List[int],
        num_classes: int,
        average: str = "macro",
    ) -> float:
        per_class = SpecificityMetric.per_class(y_true, y_pred, num_classes)
        if average == "macro":
            return float(np.mean(per_class)) if len(per_class) else 0.0
        raise ValueError(f"Unsupported average: {average}")
