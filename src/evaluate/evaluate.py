"""
Evaluation pipeline: runs a trained model against a held-out DataLoader
(the untouched Testing set) and computes all required metrics -- Confusion
Matrix, Recall, F1-Score (Macro), Precision, ROC-AUC (Macro/OvR), Accuracy --
using the classes in src/metrics/.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.metrics.accuracy import AccuracyMetric
from src.metrics.confusion_matrix import ConfusionMatrixMetric
from src.metrics.f1_score import F1ScoreMetric
from src.metrics.precision import PrecisionMetric
from src.metrics.recall import RecallMetric
from src.metrics.roc_auc import ROCAUCMetric

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for every metric computed by ModelEvaluator, macro-averaged
    plus per-class breakdowns for recall/precision/F1."""

    confusion_matrix: np.ndarray
    accuracy: float
    recall_macro: float
    precision_macro: float
    f1_macro: float
    roc_auc_macro: float
    recall_per_class: np.ndarray
    precision_per_class: np.ndarray
    f1_per_class: np.ndarray

    def to_dict(self) -> Dict:
        """Flat, JSON/MLflow-friendly dict representation (numpy arrays -> lists)."""
        return {
            "accuracy": self.accuracy,
            "recall_macro": self.recall_macro,
            "precision_macro": self.precision_macro,
            "f1_macro": self.f1_macro,
            "roc_auc_macro": self.roc_auc_macro,
            "recall_per_class": self.recall_per_class.tolist(),
            "precision_per_class": self.precision_per_class.tolist(),
            "f1_per_class": self.f1_per_class.tolist(),
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


class ModelEvaluator:
    """Runs inference over a DataLoader and computes the full metric suite.

    Typical usage (against the untouched Testing set):
        model = build_model(config, pretrained=False)  # or EfficientNetClassifier(...)
        load_model_checkpoint(model, "artifacts/checkpoints/best_model.pt")
        evaluator = ModelEvaluator(model, test_loader, num_classes=4)
        result = evaluator.evaluate()
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_classes: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            model: A trained model (weights already loaded), e.g. via
                src/utils/checkpoint.py's load_model_checkpoint.
            data_loader: DataLoader to evaluate against (typically the Testing set).
            num_classes: Total number of classes (4 for this project).
            device: Device to run inference on. Defaults to CUDA if available, else CPU.
        """
        self.model = model
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self) -> EvaluationResult:
        """Run inference over the full data_loader and compute all metrics."""
        y_true, y_pred, y_score = self._run_inference()

        result = EvaluationResult(
            confusion_matrix=ConfusionMatrixMetric.compute(y_true, y_pred, num_classes=self.num_classes),
            accuracy=AccuracyMetric.compute(y_true, y_pred),
            recall_macro=RecallMetric.compute(y_true, y_pred, average="macro"),
            precision_macro=PrecisionMetric.compute(y_true, y_pred, average="macro"),
            f1_macro=F1ScoreMetric.compute(y_true, y_pred, average="macro"),
            roc_auc_macro=ROCAUCMetric.compute(y_true, y_score, num_classes=self.num_classes),
            recall_per_class=RecallMetric.per_class(y_true, y_pred, num_classes=self.num_classes),
            precision_per_class=PrecisionMetric.per_class(y_true, y_pred, num_classes=self.num_classes),
            f1_per_class=F1ScoreMetric.per_class(y_true, y_pred, num_classes=self.num_classes),
        )

        logger.info(
            "Evaluation results: accuracy=%.4f recall_macro=%.4f precision_macro=%.4f "
            "f1_macro=%.4f roc_auc_macro=%.4f",
            result.accuracy, result.recall_macro, result.precision_macro,
            result.f1_macro, result.roc_auc_macro,
        )
        return result

    def _run_inference(self) -> Tuple[List[int], List[int], List[List[float]]]:
        """Run the model over data_loader and collect true labels, predicted
        labels, and predicted probabilities (needed for ROC-AUC)."""
        y_true: List[int] = []
        y_pred: List[int] = []
        y_score: List[List[float]] = []

        with torch.no_grad():
            for images, labels in self.data_loader:
                images = images.to(self.device)
                logits = self.model(images)
                probabilities = F.softmax(logits, dim=1)
                predictions = probabilities.argmax(dim=1)

                y_true.extend(labels.tolist())
                y_pred.extend(predictions.cpu().tolist())
                y_score.extend(probabilities.cpu().tolist())

        return y_true, y_pred, y_score
