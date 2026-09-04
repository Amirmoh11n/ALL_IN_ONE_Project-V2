"""Unit tests for V2 metrics (no network, no dataset)."""

import numpy as np

from src.metrics.accuracy import AccuracyMetric
from src.metrics.calibration import ECEMetric
from src.metrics.confusion_matrix import ConfusionMatrixMetric
from src.metrics.f1_score import F1ScoreMetric
from src.metrics.ppv_npv import PredictiveValueMetric
from src.metrics.precision import PrecisionMetric
from src.metrics.recall import RecallMetric
from src.metrics.roc_auc import ROCAUCMetric
from src.metrics.specificity import SpecificityMetric


def test_perfect_predictions():
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = list(y_true)
    assert AccuracyMetric.compute(y_true, y_pred) == 1.0
    assert RecallMetric.compute(y_true, y_pred, average="macro") == 1.0
    assert PrecisionMetric.compute(y_true, y_pred, average="macro") == 1.0
    assert F1ScoreMetric.compute(y_true, y_pred, average="macro") == 1.0
    assert SpecificityMetric.compute(y_true, y_pred, 4) == 1.0
    assert PredictiveValueMetric.ppv_macro(y_true, y_pred, 4) == 1.0
    assert PredictiveValueMetric.npv_macro(y_true, y_pred, 4) == 1.0
    matrix = ConfusionMatrixMetric.compute(y_true, y_pred, num_classes=4)
    assert matrix.shape == (4, 4)
    assert int(matrix.diagonal().sum()) == 8


def test_per_class_shapes():
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 1, 1, 1, 2, 3, 3, 3]
    assert RecallMetric.per_class(y_true, y_pred, 4).shape == (4,)
    assert PrecisionMetric.per_class(y_true, y_pred, 4).shape == (4,)
    assert F1ScoreMetric.per_class(y_true, y_pred, 4).shape == (4,)
    assert SpecificityMetric.per_class(y_true, y_pred, 4).shape == (4,)


def test_ece_confident_and_correct():
    y_true = [0, 1, 2, 3]
    y_prob = [
        [0.97, 0.01, 0.01, 0.01],
        [0.01, 0.97, 0.01, 0.01],
        [0.01, 0.01, 0.97, 0.01],
        [0.01, 0.01, 0.01, 0.97],
    ]
    ece = ECEMetric.compute(y_true, y_prob, n_bins=10)
    assert ece < 0.05


def test_ece_empty():
    assert ECEMetric.compute([], [], n_bins=5) == 0.0


def test_roc_auc_perfect_and_single_class():
    y_true = [0, 1, 2, 3]
    y_score = np.eye(4).tolist()
    auc = ROCAUCMetric.compute(y_true, y_score, num_classes=4)
    assert auc == 1.0
    assert ROCAUCMetric.compute([0, 0], [[0.9, 0.1], [0.8, 0.2]], num_classes=2) == 0.0
