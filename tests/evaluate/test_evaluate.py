"""
Tests for src/evaluate/evaluate.py (ModelEvaluator, EvaluationResult).

Correctness of each metric formula is already covered by tests/test_metrics.py;
these tests focus on the orchestration/wiring: correct shapes, correct types,
correct data flow from model output -> metrics.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.augment import AugmentationFactory
from src.data.classes import TumorClasses
from src.data.dataset import BrainTumorDataset
from src.evaluate.evaluate import EvaluationResult, ModelEvaluator


class TinyDummyModel(nn.Module):
    """Small CNN standing in for EfficientNetB3Classifier -- fast on CPU."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture
def test_loader(synthetic_testing_dir):
    transform = AugmentationFactory.build_eval_transforms(image_size=32)
    dataset = BrainTumorDataset.from_directory(synthetic_testing_dir, transform=transform)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def test_evaluate_returns_evaluation_result(test_loader):
    model = TinyDummyModel(num_classes=TumorClasses.num_classes())
    evaluator = ModelEvaluator(model, test_loader, num_classes=TumorClasses.num_classes(), device=torch.device("cpu"))

    result = evaluator.evaluate()

    assert isinstance(result, EvaluationResult)


def test_confusion_matrix_has_correct_shape(test_loader):
    model = TinyDummyModel(num_classes=TumorClasses.num_classes())
    evaluator = ModelEvaluator(model, test_loader, num_classes=TumorClasses.num_classes(), device=torch.device("cpu"))

    result = evaluator.evaluate()

    assert result.confusion_matrix.shape == (4, 4)
    # every test sample must be counted exactly once
    assert result.confusion_matrix.sum() == len(test_loader.dataset)


def test_macro_metrics_are_valid_floats_in_range(test_loader):
    model = TinyDummyModel(num_classes=TumorClasses.num_classes())
    evaluator = ModelEvaluator(model, test_loader, num_classes=TumorClasses.num_classes(), device=torch.device("cpu"))

    result = evaluator.evaluate()

    for value in [result.accuracy, result.recall_macro, result.precision_macro,
                  result.f1_macro, result.roc_auc_macro]:
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_per_class_metrics_have_one_value_per_class(test_loader):
    model = TinyDummyModel(num_classes=TumorClasses.num_classes())
    evaluator = ModelEvaluator(model, test_loader, num_classes=TumorClasses.num_classes(), device=torch.device("cpu"))

    result = evaluator.evaluate()

    assert len(result.recall_per_class) == 4
    assert len(result.precision_per_class) == 4
    assert len(result.f1_per_class) == 4


def test_to_dict_is_json_serializable(test_loader):
    import json

    model = TinyDummyModel(num_classes=TumorClasses.num_classes())
    evaluator = ModelEvaluator(model, test_loader, num_classes=TumorClasses.num_classes(), device=torch.device("cpu"))

    result_dict = evaluator.evaluate().to_dict()

    # numpy arrays must have been converted to plain lists
    assert isinstance(result_dict["confusion_matrix"], list)
    assert isinstance(result_dict["recall_per_class"], list)
    json.dumps(result_dict)  # raises if anything isn't JSON-serializable


def test_evaluator_puts_model_in_eval_mode(test_loader):
    model = TinyDummyModel(num_classes=TumorClasses.num_classes())
    model.train()
    ModelEvaluator(model, test_loader, num_classes=TumorClasses.num_classes(), device=torch.device("cpu"))

    assert model.training is False
