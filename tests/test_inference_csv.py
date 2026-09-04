"""Batch CSV layout for folder inference (no checkpoint)."""

import csv
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.inference.inference import InferencePipeline, PredictionResult


def test_prediction_csv_roundtrip(tmp_path: Path):
    results = [
        PredictionResult(
            predicted_class="glioma",
            confidence=0.7,
            probabilities={"glioma": 0.7, "meningioma": 0.1, "notumor": 0.1, "pituitary": 0.1},
            model_version="2.0.0",
            path="a.png",
        ),
        PredictionResult(
            predicted_class="notumor",
            confidence=0.8,
            probabilities={"glioma": 0.05, "meningioma": 0.05, "notumor": 0.8, "pituitary": 0.1},
            model_version="2.0.0",
            path="b.png",
        ),
    ]
    output = InferencePipeline.write_csv(
        results, tmp_path / "batch.csv", ["glioma", "meningioma", "notumor", "pituitary"]
    )
    with output.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["predicted_class"] == "glioma"
    assert rows[1]["notumor"] == "0.800000"
