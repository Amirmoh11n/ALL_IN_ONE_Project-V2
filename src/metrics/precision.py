"""
Precision metric (priority #4). Wraps scikit-learn's implementation.
"""

from typing import List, Optional

import numpy as np
from sklearn.metrics import precision_score


class PrecisionMetric:
    """Computes precision: of all predicted positives, how many were correct."""

    @staticmethod
    def compute(y_true: List[int], y_pred: List[int], average: str = "macro") -> float:
        """
        Args:
            y_true: Ground-truth integer class labels.
            y_pred: Predicted integer class labels.
            average: sklearn averaging strategy ("macro", "micro", "weighted").
                "macro" is the project default.

        Returns:
            The averaged precision as a float.
        """
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def per_class(y_true: List[int], y_pred: List[int], num_classes: Optional[int] = None) -> np.ndarray:
        """Return precision for each class individually (no averaging)."""
        labels = list(range(num_classes)) if num_classes is not None else None
        return precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
