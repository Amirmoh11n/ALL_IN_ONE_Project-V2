"""
Confusion Matrix metric (priority #1). Wraps scikit-learn's implementation
behind a small class so the rest of the project depends on this project's
metric interface, not directly on sklearn.
"""

from typing import List, Optional

import numpy as np
from sklearn.metrics import confusion_matrix


class ConfusionMatrixMetric:
    """Computes the confusion matrix for multi-class predictions.

    Rows are true classes, columns are predicted classes (sklearn convention):
    matrix[i, j] = number of samples with true label i predicted as label j.
    """

    @staticmethod
    def compute(y_true: List[int], y_pred: List[int], num_classes: Optional[int] = None) -> np.ndarray:
        """
        Args:
            y_true: Ground-truth integer class labels.
            y_pred: Predicted integer class labels.
            num_classes: If given, forces the matrix to be num_classes x num_classes
                even if some classes are absent from y_true/y_pred.

        Returns:
            A (num_classes, num_classes) numpy array.
        """
        labels = list(range(num_classes)) if num_classes is not None else None
        return confusion_matrix(y_true, y_pred, labels=labels)
