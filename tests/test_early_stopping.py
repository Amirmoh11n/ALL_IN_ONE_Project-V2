"""EarlyStopping and class-weight tests (requires torch only for Trainer module import)."""

import pytest

torch = pytest.importorskip("torch")

from src.engine.trainer import EarlyStopping, compute_class_weights


def test_early_stopping_max_mode():
    stopper = EarlyStopping(patience=2, mode="max", min_delta=0.0)
    stopper.step(0.50)
    stopper.step(0.51)
    assert not stopper.should_stop
    stopper.step(0.50)
    stopper.step(0.49)
    assert stopper.should_stop
    assert stopper.best_score == pytest.approx(0.51)


def test_early_stopping_min_mode():
    stopper = EarlyStopping(patience=1, mode="min")
    stopper.step(1.0)
    stopper.step(1.2)
    assert stopper.should_stop


def test_early_stopping_rejects_bad_args():
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)
    with pytest.raises(ValueError):
        EarlyStopping(mode="auto")


def test_class_weights_inverse_frequency():
    weights = compute_class_weights([0, 0, 0, 1], num_classes=2)
    assert weights.shape == (2,)
    assert weights[1] > weights[0]


def test_class_weights_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_class_weights([], 4)
