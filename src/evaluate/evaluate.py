"""Evaluation pipeline against the untouched Testing set with V2 metrics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.metrics.accuracy import AccuracyMetric
from src.metrics.calibration import ECEMetric, TemperatureScaler
from src.metrics.confusion_matrix import ConfusionMatrixMetric
from src.metrics.f1_score import F1ScoreMetric
from src.metrics.plots import EvaluationPlotter
from src.metrics.ppv_npv import PredictiveValueMetric
from src.metrics.precision import PrecisionMetric
from src.metrics.recall import RecallMetric
from src.metrics.roc_auc import ROCAUCMetric
from src.metrics.specificity import SpecificityMetric

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Full V2 metric suite including calibration and predictive values."""

    confusion_matrix: np.ndarray
    accuracy: float
    recall_macro: float
    precision_macro: float
    f1_macro: float
    roc_auc_macro: float
    specificity_macro: float
    ppv_macro: float
    npv_macro: float
    ece: float
    temperature: float
    recall_per_class: np.ndarray
    precision_per_class: np.ndarray
    f1_per_class: np.ndarray
    specificity_per_class: np.ndarray
    ppv_per_class: np.ndarray
    npv_per_class: np.ndarray
    split_mode: str = "stratified_image_level"
    model_version: str = "2.0.0"
    plot_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "model_version": self.model_version,
            "split_mode": self.split_mode,
            "accuracy": self.accuracy,
            "recall_macro": self.recall_macro,
            "precision_macro": self.precision_macro,
            "f1_macro": self.f1_macro,
            "roc_auc_macro": self.roc_auc_macro,
            "specificity_macro": self.specificity_macro,
            "ppv_macro": self.ppv_macro,
            "npv_macro": self.npv_macro,
            "ece": self.ece,
            "temperature": self.temperature,
            "recall_per_class": self.recall_per_class.tolist(),
            "precision_per_class": self.precision_per_class.tolist(),
            "f1_per_class": self.f1_per_class.tolist(),
            "specificity_per_class": self.specificity_per_class.tolist(),
            "ppv_per_class": self.ppv_per_class.tolist(),
            "npv_per_class": self.npv_per_class.tolist(),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "plots": self.plot_paths,
        }

    def to_markdown_table(self, class_names: List[str]) -> str:
        rows = [
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Accuracy | {self.accuracy:.4f} |",
            f"| Recall (macro) | {self.recall_macro:.4f} |",
            f"| F1 (macro) | {self.f1_macro:.4f} |",
            f"| Precision (macro) | {self.precision_macro:.4f} |",
            f"| ROC-AUC (macro) | {self.roc_auc_macro:.4f} |",
            f"| Specificity (macro) | {self.specificity_macro:.4f} |",
            f"| PPV (macro) | {self.ppv_macro:.4f} |",
            f"| NPV (macro) | {self.npv_macro:.4f} |",
            f"| ECE | {self.ece:.4f} |",
            f"| Temperature | {self.temperature:.3f} |",
        ]
        return "\n".join(rows)


class ModelEvaluator:
    """Runs inference over a DataLoader and computes the V2 metric suite."""

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_classes: int,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        temperature_scaling: bool = True,
        ece_bins: int = 15,
        split_mode: str = "stratified_image_level",
        model_version: str = "2.0.0",
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.output_dir = Path(output_dir) if output_dir else None
        self.temperature_scaling = temperature_scaling
        self.ece_bins = ece_bins
        self.split_mode = split_mode
        self.model_version = model_version
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self) -> EvaluationResult:
        y_true, y_pred, y_score, logits = self._run_inference()
        temperature = 1.0
        if self.temperature_scaling and logits:
            scaler = TemperatureScaler()
            temperature = scaler.fit(
                torch.tensor(logits, dtype=torch.float32),
                torch.tensor(y_true, dtype=torch.long),
            )
            calibrated = F.softmax(torch.tensor(logits) / temperature, dim=1).numpy()
            y_score = calibrated.tolist()
            y_pred = calibrated.argmax(axis=1).tolist()

        result = EvaluationResult(
            confusion_matrix=ConfusionMatrixMetric.compute(y_true, y_pred, num_classes=self.num_classes),
            accuracy=AccuracyMetric.compute(y_true, y_pred),
            recall_macro=RecallMetric.compute(y_true, y_pred, average="macro"),
            precision_macro=PrecisionMetric.compute(y_true, y_pred, average="macro"),
            f1_macro=F1ScoreMetric.compute(y_true, y_pred, average="macro"),
            roc_auc_macro=ROCAUCMetric.compute(y_true, y_score, num_classes=self.num_classes),
            specificity_macro=SpecificityMetric.compute(y_true, y_pred, self.num_classes),
            ppv_macro=PredictiveValueMetric.ppv_macro(y_true, y_pred, self.num_classes),
            npv_macro=PredictiveValueMetric.npv_macro(y_true, y_pred, self.num_classes),
            ece=ECEMetric.compute(y_true, y_score, n_bins=self.ece_bins),
            temperature=temperature,
            recall_per_class=RecallMetric.per_class(y_true, y_pred, num_classes=self.num_classes),
            precision_per_class=PrecisionMetric.per_class(y_true, y_pred, num_classes=self.num_classes),
            f1_per_class=F1ScoreMetric.per_class(y_true, y_pred, num_classes=self.num_classes),
            specificity_per_class=SpecificityMetric.per_class(y_true, y_pred, self.num_classes),
            ppv_per_class=PredictiveValueMetric.ppv_per_class(y_true, y_pred, self.num_classes),
            npv_per_class=PredictiveValueMetric.npv_per_class(y_true, y_pred, self.num_classes),
            split_mode=self.split_mode,
            model_version=self.model_version,
        )

        if self.output_dir is not None:
            plotter = EvaluationPlotter(self.output_dir, self.class_names)
            cm_path = plotter.save_confusion_matrix(result.confusion_matrix)
            roc_path = plotter.save_roc_curves(y_true, y_score)
            result.plot_paths = {"confusion_matrix": str(cm_path), "roc_curves": str(roc_path)}

        logger.info(
            "Evaluation: acc=%.4f recall=%.4f f1=%.4f auc=%.4f spec=%.4f ece=%.4f T=%.3f",
            result.accuracy,
            result.recall_macro,
            result.f1_macro,
            result.roc_auc_macro,
            result.specificity_macro,
            result.ece,
            result.temperature,
        )
        return result

    def _run_inference(self) -> Tuple[List[int], List[int], List[List[float]], List[List[float]]]:
        y_true: List[int] = []
        y_pred: List[int] = []
        y_score: List[List[float]] = []
        logits_out: List[List[float]] = []
        with torch.no_grad():
            for images, labels in self.data_loader:
                images = images.to(self.device)
                logits = self.model(images)
                probabilities = F.softmax(logits, dim=1)
                predictions = probabilities.argmax(dim=1)
                y_true.extend(labels.tolist())
                y_pred.extend(predictions.cpu().tolist())
                y_score.extend(probabilities.cpu().tolist())
                logits_out.extend(logits.cpu().tolist())
        return y_true, y_pred, y_score, logits_out
