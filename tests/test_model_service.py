"""ONNX serving helpers without a trained model file."""

import numpy as np
import pytest

pytest.importorskip("PIL")

from PIL import Image

from webapplication.backend.model_service import ONNXModelService


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0, 0.0]], dtype=np.float32)
    probs = ONNXModelService.softmax(logits)
    assert probs.shape == (1, 4)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert int(probs.argmax()) == 2


def test_preprocess_shape(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "data:\n  class_names: [glioma, meningioma, notumor, pituitary]\n"
        "  image_size: 64\n  normalization:\n    mean: [0,0,0]\n    std: [1,1,1]\n"
        "export:\n  input_size: 64\n  model_version: '2.0.0'\n",
        encoding="utf-8",
    )
    missing = tmp_path / "missing.onnx"
    service = ONNXModelService(missing, cfg)
    assert service.ready is False
    assert service.model_version == "2.0.0"
    image = Image.new("RGB", (32, 40), color=(128, 128, 128))
    tensor = service.preprocess(image)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32
