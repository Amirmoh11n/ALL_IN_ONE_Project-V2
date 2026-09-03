"""Unit tests for V2 metrics (no network, no dataset)."""

from src.metrics.accuracy import AccuracyMetric
from src.metrics.calibration import ECEMetric
from src.metrics.confusion_matrix import ConfusionMatrixMetric
from src.metrics.f1_score import F1ScoreMetric
from src.metrics.ppv_npv import PredictiveValueMetric
from src.metrics.recall import RecallMetric
from src.metrics.specificity import SpecificityMetric


def test_perfect_predictions():
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = list(y_true)
    assert AccuracyMetric.compute(y_true, y_pred) == 1.0
    assert RecallMetric.compute(y_true, y_pred, average="macro") == 1.0
    assert F1ScoreMetric.compute(y_true, y_pred, average="macro") == 1.0
    assert SpecificityMetric.compute(y_true, y_pred, 4) == 1.0
    assert PredictiveValueMetric.ppv_macro(y_true, y_pred, 4) == 1.0
    matrix = ConfusionMatrixMetric.compute(y_true, y_pred, num_classes=4)
    assert matrix.shape == (4, 4)
    assert int(matrix.diagonal().sum()) == 8


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
