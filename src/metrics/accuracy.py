"""
Accuracy metric (priority #6, lowest priority per project spec -- accuracy
alone is a weak signal on an imbalanced medical dataset, but is still tracked).
Wraps scikit-learn's implementation.
"""

from typing import List

from sklearn.metrics import accuracy_score


class AccuracyMetric:
    """Computes overall accuracy: fraction of correctly classified samples."""

    @staticmethod
    def compute(y_true: List[int], y_pred: List[int]) -> float:
        """
        Args:
            y_true: Ground-truth integer class labels.
            y_pred: Predicted integer class labels.

        Returns:
            The accuracy as a float in [0, 1].
        """
        return accuracy_score(y_true, y_pred)
