"""
Recall / Sensitivity metric (priority #2). Wraps scikit-learn's implementation.
"""

from typing import List, Optional

import numpy as np
from sklearn.metrics import recall_score


class RecallMetric:
    """Computes recall (sensitivity): of all actual positives, how many were caught."""

    @staticmethod
    def compute(y_true: List[int], y_pred: List[int], average: str = "macro") -> float:
        """
        Args:
            y_true: Ground-truth integer class labels.
            y_pred: Predicted integer class labels.
            average: sklearn averaging strategy ("macro", "micro", "weighted").
                "macro" (unweighted mean across classes) is the project default,
                so minority classes count as much as the majority class.

        Returns:
            The averaged recall as a float.
        """
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def per_class(y_true: List[int], y_pred: List[int], num_classes: Optional[int] = None) -> np.ndarray:
        """Return recall for each class individually (no averaging)."""
        labels = list(range(num_classes)) if num_classes is not None else None
        return recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
