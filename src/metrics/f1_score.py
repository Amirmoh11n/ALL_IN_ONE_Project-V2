"""
F1-Score (Macro) metric (priority #3). Wraps scikit-learn's implementation.
"""

from typing import List, Optional

import numpy as np
from sklearn.metrics import f1_score


class F1ScoreMetric:
    """Computes the F1-score: the harmonic mean of precision and recall."""

    @staticmethod
    def compute(y_true: List[int], y_pred: List[int], average: str = "macro") -> float:
        """
        Args:
            y_true: Ground-truth integer class labels.
            y_pred: Predicted integer class labels.
            average: sklearn averaging strategy. "macro" is the project default
                (per the required "F1-Score (Macro)" metric), giving each class
                equal weight regardless of its sample count.

        Returns:
            The averaged F1-score as a float.
        """
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    @staticmethod
    def per_class(y_true: List[int], y_pred: List[int], num_classes: Optional[int] = None) -> np.ndarray:
        """Return F1-score for each class individually (no averaging)."""
        labels = list(range(num_classes)) if num_classes is not None else None
        return f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
