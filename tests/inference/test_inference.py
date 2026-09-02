"""
Tests for src/inference/inference.py (InferencePipeline, PredictionResult).

Uses a tiny model wired in place of EfficientNetB3Classifier's internals is
not practical here (InferencePipeline hardcodes EfficientNetB3Classifier so
the checkpoint format matches training), so a real EfficientNetB3Classifier
is used with pretrained=False (random weights) -- predictions won't be
meaningful, but the pipeline's plumbing (loading, transform, output shape/
format) is fully exercised and verifiable.
"""

import json

import pytest
import torch
from PIL import Image

from src.data.classes import TumorClasses
from src.inference.inference import InferencePipeline, PredictionResult
from src.models.efficientnet import EfficientNetB3Classifier
from src.utils.config_loader import ConfigLoader


@pytest.fixture
def checkpoint_path(tmp_path):
    """Save a randomly-initialized EfficientNetB3Classifier checkpoint, matching
    the exact format Trainer._save_checkpoint produces."""
    model = EfficientNetB3Classifier(num_classes=TumorClasses.num_classes(), pretrained=False)
    path = tmp_path / "best_model.pt"
    torch.save({"epoch": 1, "model_state_dict": model.state_dict(), "val_loss": 0.5}, path)
    return path


@pytest.fixture
def inference_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
model:
  num_classes: 4
data:
  image_size: 64
  normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
""")
    return ConfigLoader(config_path)


@pytest.fixture
def sample_image(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (50, 50), color=(120, 60, 200)).save(image_path)
    return image_path


def test_predict_from_file_path_returns_prediction_result(checkpoint_path, inference_config, sample_image):
    pipeline = InferencePipeline(checkpoint_path, inference_config, device=torch.device("cpu"))
    result = pipeline.predict(sample_image)

    assert isinstance(result, PredictionResult)
    assert result.predicted_class in TumorClasses.NAMES


def test_predict_from_pil_image_returns_prediction_result(checkpoint_path, inference_config, sample_image):
    pipeline = InferencePipeline(checkpoint_path, inference_config, device=torch.device("cpu"))
    pil_image = Image.open(sample_image)

    result = pipeline.predict(pil_image)

    assert isinstance(result, PredictionResult)
    assert result.predicted_class in TumorClasses.NAMES


def test_probabilities_cover_all_classes_and_sum_to_one(checkpoint_path, inference_config, sample_image):
    pipeline = InferencePipeline(checkpoint_path, inference_config, device=torch.device("cpu"))
    result = pipeline.predict(sample_image)

    assert set(result.probabilities.keys()) == set(TumorClasses.NAMES)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_confidence_matches_the_predicted_class_probability(checkpoint_path, inference_config, sample_image):
    pipeline = InferencePipeline(checkpoint_path, inference_config, device=torch.device("cpu"))
    result = pipeline.predict(sample_image)

    assert result.confidence == pytest.approx(result.probabilities[result.predicted_class])
    assert result.confidence == max(result.probabilities.values())


def test_to_dict_is_json_serializable(checkpoint_path, inference_config, sample_image):
    pipeline = InferencePipeline(checkpoint_path, inference_config, device=torch.device("cpu"))
    result_dict = pipeline.predict(sample_image).to_dict()

    json.dumps(result_dict)  # raises if not serializable
    assert set(result_dict.keys()) == {"predicted_class", "confidence", "probabilities"}


def test_grayscale_input_image_is_handled_via_rgb_conversion(checkpoint_path, inference_config, tmp_path):
    grayscale_path = tmp_path / "grayscale.jpg"
    Image.new("L", (50, 50), color=128).save(grayscale_path)

    pipeline = InferencePipeline(checkpoint_path, inference_config, device=torch.device("cpu"))
    result = pipeline.predict(grayscale_path)  # should not raise

    assert result.predicted_class in TumorClasses.NAMES
