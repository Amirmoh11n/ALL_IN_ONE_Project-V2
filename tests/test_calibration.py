"""Temperature scaling tests (torch required)."""

import pytest

torch = pytest.importorskip("torch")

from src.metrics.calibration import ECEMetric, TemperatureScaler


def test_temperature_scaler_learns_finite_t():
    scaler = TemperatureScaler()
    logits = torch.tensor(
        [
            [4.0, 0.1, 0.1, 0.1],
            [0.1, 4.0, 0.1, 0.1],
            [0.1, 0.1, 4.0, 0.1],
            [0.1, 0.1, 0.1, 4.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 3])
    temperature = scaler.fit(logits, labels, max_iter=20)
    assert temperature > 0
    calibrated = scaler(logits)
    assert calibrated.shape == logits.shape


def test_ece_miscalibrated_is_higher_than_perfect():
    y_true = [0, 1, 2, 3]
    sharp = [
        [0.97, 0.01, 0.01, 0.01],
        [0.01, 0.97, 0.01, 0.01],
        [0.01, 0.01, 0.97, 0.01],
        [0.01, 0.01, 0.01, 0.97],
    ]
    noisy = [
        [0.4, 0.2, 0.2, 0.2],
        [0.2, 0.4, 0.2, 0.2],
        [0.2, 0.2, 0.4, 0.2],
        [0.2, 0.2, 0.2, 0.4],
    ]
    assert ECEMetric.compute(y_true, noisy) > ECEMetric.compute(y_true, sharp)
