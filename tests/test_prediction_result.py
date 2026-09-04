"""PredictionResult serialization without loading weights."""

import pytest

pytest.importorskip("torch")

from src.inference.inference import PredictionResult


def test_to_dict_includes_version_and_optional_path():
    result = PredictionResult(
        predicted_class="notumor",
        confidence=0.91,
        probabilities={"glioma": 0.03, "meningioma": 0.03, "notumor": 0.91, "pituitary": 0.03},
        model_version="2.0.0",
        path="slice.png",
    )
    payload = result.to_dict()
    assert payload["model_version"] == "2.0.0"
    assert payload["path"] == "slice.png"
    assert payload["predicted_class"] == "notumor"
