"""EvaluationResult serialization without running a network."""

import numpy as np
import pytest

pytest.importorskip("torch")

from src.evaluate.evaluate import EvaluationResult


def _result() -> EvaluationResult:
    return EvaluationResult(
        confusion_matrix=np.eye(4, dtype=int),
        accuracy=1.0,
        recall_macro=1.0,
        precision_macro=1.0,
        f1_macro=0.91,
        roc_auc_macro=0.95,
        specificity_macro=1.0,
        ppv_macro=1.0,
        npv_macro=1.0,
        ece=0.04,
        temperature=1.2,
        recall_per_class=np.ones(4),
        precision_per_class=np.ones(4),
        f1_per_class=np.ones(4),
        specificity_per_class=np.ones(4),
        ppv_per_class=np.ones(4),
        npv_per_class=np.ones(4),
        split_mode="stratified_image_level",
        model_version="2.0.0",
        plot_paths={"confusion_matrix": "cm.png"},
    )


def test_to_dict_is_json_friendly():
    payload = _result().to_dict()
    assert payload["f1_macro"] == 0.91
    assert payload["model_version"] == "2.0.0"
    assert payload["confusion_matrix"] == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert payload["plots"]["confusion_matrix"] == "cm.png"


def test_markdown_table_contains_priority_metrics():
    table = _result().to_markdown_table(["glioma", "meningioma", "notumor", "pituitary"])
    assert "F1 (macro)" in table
    assert "0.9100" in table
    assert "ECE" in table
