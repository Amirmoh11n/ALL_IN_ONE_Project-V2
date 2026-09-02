"""
Tests for src/metrics/*.py, using a small hand-computed 4-class scenario so
expected values are known exactly (not just re-deriving sklearn's own answer).

Scenario (4 classes: 0,1,2,3 -- matching TumorClasses order):
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 1, 1, 1, 2, 3, 3, 3]

Hand-computed confusion matrix (rows=true, cols=pred):
    [[1, 1, 0, 0],
     [0, 2, 0, 0],
     [0, 0, 1, 1],
     [0, 0, 0, 2]]

    Accuracy = 6/8 = 0.75
    Per-class recall    = [0.5, 1.0, 0.5, 1.0]      -> macro = 0.75
    Per-class precision = [1.0, 2/3, 1.0, 2/3]      -> macro = 0.8333...
    Per-class F1        = [2/3, 0.8, 2/3, 0.8]      -> macro = 0.7333...
"""

import numpy as np
import pytest

from src.metrics.accuracy import AccuracyMetric
from src.metrics.confusion_matrix import ConfusionMatrixMetric
from src.metrics.f1_score import F1ScoreMetric
from src.metrics.precision import PrecisionMetric
from src.metrics.recall import RecallMetric
from src.metrics.roc_auc import ROCAUCMetric

Y_TRUE = [0, 0, 1, 1, 2, 2, 3, 3]
Y_PRED = [0, 1, 1, 1, 2, 3, 3, 3]
NUM_CLASSES = 4


def test_confusion_matrix_matches_hand_computed_values():
    expected = np.array([
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 2],
    ])
    result = ConfusionMatrixMetric.compute(Y_TRUE, Y_PRED, num_classes=NUM_CLASSES)
    np.testing.assert_array_equal(result, expected)


def test_accuracy_matches_hand_computed_value():
    assert AccuracyMetric.compute(Y_TRUE, Y_PRED) == pytest.approx(0.75)


def test_recall_macro_matches_hand_computed_value():
    assert RecallMetric.compute(Y_TRUE, Y_PRED, average="macro") == pytest.approx(0.75)


def test_recall_per_class_matches_hand_computed_values():
    result = RecallMetric.per_class(Y_TRUE, Y_PRED, num_classes=NUM_CLASSES)
    np.testing.assert_allclose(result, [0.5, 1.0, 0.5, 1.0])


def test_precision_macro_matches_hand_computed_value():
    assert PrecisionMetric.compute(Y_TRUE, Y_PRED, average="macro") == pytest.approx(5 / 6, rel=1e-3)


def test_precision_per_class_matches_hand_computed_values():
    result = PrecisionMetric.per_class(Y_TRUE, Y_PRED, num_classes=NUM_CLASSES)
    np.testing.assert_allclose(result, [1.0, 2 / 3, 1.0, 2 / 3], rtol=1e-3)


def test_f1_macro_matches_hand_computed_value():
    assert F1ScoreMetric.compute(Y_TRUE, Y_PRED, average="macro") == pytest.approx(11 / 15, rel=1e-3)


def test_f1_per_class_matches_hand_computed_values():
    result = F1ScoreMetric.per_class(Y_TRUE, Y_PRED, num_classes=NUM_CLASSES)
    np.testing.assert_allclose(result, [2 / 3, 0.8, 2 / 3, 0.8], rtol=1e-3)


def test_perfect_predictions_give_perfect_scores_for_all_metrics():
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = list(y_true)

    assert AccuracyMetric.compute(y_true, y_pred) == pytest.approx(1.0)
    assert RecallMetric.compute(y_true, y_pred) == pytest.approx(1.0)
    assert PrecisionMetric.compute(y_true, y_pred) == pytest.approx(1.0)
    assert F1ScoreMetric.compute(y_true, y_pred) == pytest.approx(1.0)


def test_roc_auc_is_perfect_for_perfectly_separated_probabilities():
    y_true = [0, 1, 2, 3]
    # one-hot-like "perfect confidence" probabilities matching the true class
    y_score = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    result = ROCAUCMetric.compute(y_true, y_score, num_classes=4)
    assert result == pytest.approx(1.0)


def test_roc_auc_matches_sklearn_directly():
    """Sanity check: our thin wrapper should return exactly what sklearn returns
    for the same inputs and parameters (tests correct parameter wiring)."""
    from sklearn.metrics import roc_auc_score

    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_score = [
        [0.7, 0.1, 0.1, 0.1],
        [0.2, 0.5, 0.2, 0.1],
        [0.1, 0.1, 0.6, 0.2],
        [0.1, 0.1, 0.2, 0.6],
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.2, 0.2, 0.5, 0.1],
        [0.1, 0.2, 0.2, 0.5],
    ]
    expected = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=[0, 1, 2, 3])
    result = ROCAUCMetric.compute(y_true, y_score, num_classes=4)
    assert result == pytest.approx(expected)


def test_metrics_handle_a_class_with_zero_predictions_without_crashing():
    # class 3 never predicted -> precision for class 3 should be 0, not a crash/NaN
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 2]

    precision_per_class = PrecisionMetric.per_class(y_true, y_pred, num_classes=4)
    assert precision_per_class[3] == pytest.approx(0.0)
