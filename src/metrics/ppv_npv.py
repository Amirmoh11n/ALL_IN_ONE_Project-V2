"""Positive / Negative Predictive Value helpers for multi-class reports."""

from __future__ import annotations

from typing import List

import numpy as np

from src.metrics.confusion_matrix import ConfusionMatrixMetric


class PredictiveValueMetric:
    """PPV (precision) and NPV derived from the confusion matrix."""

    @staticmethod
    def ppv_per_class(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
        matrix = ConfusionMatrixMetric.compute(y_true, y_pred, num_classes=num_classes)
        scores = np.zeros(num_classes, dtype=np.float64)
        for i in range(num_classes):
            tp = matrix[i, i]
            fp = matrix[:, i].sum() - tp
            denom = tp + fp
            scores[i] = float(tp / denom) if denom else 0.0
        return scores

    @staticmethod
    def npv_per_class(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
        matrix = ConfusionMatrixMetric.compute(y_true, y_pred, num_classes=num_classes)
        totals = matrix.sum()
        scores = np.zeros(num_classes, dtype=np.float64)
        for i in range(num_classes):
            tp = matrix[i, i]
            fn = matrix[i, :].sum() - tp
            fp = matrix[:, i].sum() - tp
            tn = totals - tp - fn - fp
            denom = tn + fn
            scores[i] = float(tn / denom) if denom else 0.0
        return scores

    @staticmethod
    def ppv_macro(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
        return float(np.mean(PredictiveValueMetric.ppv_per_class(y_true, y_pred, num_classes)))

    @staticmethod
    def npv_macro(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
        return float(np.mean(PredictiveValueMetric.npv_per_class(y_true, y_pred, num_classes)))
