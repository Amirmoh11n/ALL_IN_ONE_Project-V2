"""
Tests for src/export/export.py (ModelExporter).

Uses a tiny dummy CNN instead of the full EfficientNet-B3 so exports run in
seconds; the export mechanics (tracing, saving, format validity) are identical
regardless of the underlying architecture.
"""

import numpy as np
import torch
import torch.nn as nn

from src.export.export import ModelExporter


class TinyDummyModel(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_export_web_produces_loadable_torchscript_with_matching_output(tmp_path):
    model = TinyDummyModel()
    exporter = ModelExporter(model, input_size=32)

    output_path = exporter.export_web(tmp_path / "model_web.pt")
    assert output_path.exists()

    loaded = torch.jit.load(str(output_path))
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        original_output = model(dummy)
        loaded_output = loaded(dummy)

    torch.testing.assert_close(original_output, loaded_output)


def test_export_gpu_produces_loadable_optimized_torchscript(tmp_path):
    model = TinyDummyModel()
    exporter = ModelExporter(model, input_size=32)

    output_path = exporter.export_gpu(tmp_path / "model_gpu.pt")
    assert output_path.exists()

    loaded = torch.jit.load(str(output_path))
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        original_output = model(dummy)
        loaded_output = loaded(dummy)

    torch.testing.assert_close(original_output, loaded_output, atol=1e-4, rtol=1e-4)


def test_export_mobile_produces_loadable_lite_interpreter_model(tmp_path):
    from torch.jit.mobile import _load_for_lite_interpreter

    model = TinyDummyModel()
    exporter = ModelExporter(model, input_size=32)

    output_path = exporter.export_mobile(tmp_path / "model_mobile.ptl")
    assert output_path.exists()

    loaded = _load_for_lite_interpreter(str(output_path))
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        original_output = model(dummy)
        loaded_output = loaded(dummy)

    torch.testing.assert_close(original_output, loaded_output, atol=1e-4, rtol=1e-4)


def test_export_onnx_produces_valid_model_matching_pytorch_output(tmp_path):
    import onnx
    import onnxruntime as ort

    model = TinyDummyModel()
    exporter = ModelExporter(model, input_size=32)

    output_path = exporter.export_onnx(tmp_path / "model.onnx")
    assert output_path.exists()

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)  # raises if structurally invalid

    dummy = torch.randn(1, 3, 32, 32)
    session = ort.InferenceSession(str(output_path))
    onnx_output = session.run(None, {"input": dummy.numpy()})[0]

    with torch.no_grad():
        torch_output = model(dummy).numpy()

    np.testing.assert_allclose(onnx_output, torch_output, atol=1e-4)


def test_export_onnx_supports_dynamic_batch_size(tmp_path):
    import onnxruntime as ort

    model = TinyDummyModel()
    exporter = ModelExporter(model, input_size=32)
    output_path = exporter.export_onnx(tmp_path / "model.onnx")

    session = ort.InferenceSession(str(output_path))
    batch_of_five = torch.randn(5, 3, 32, 32)
    onnx_output = session.run(None, {"input": batch_of_five.numpy()})[0]

    assert onnx_output.shape == (5, 4)


def test_export_all_produces_all_four_formats(tmp_path):
    model = TinyDummyModel()
    exporter = ModelExporter(model, input_size=32)

    paths = exporter.export_all(tmp_path, base_name="tiny_model")

    assert set(paths.keys()) == {"web", "mobile", "gpu", "onnx"}
    for fmt, path in paths.items():
        assert path.exists(), f"{fmt} export file missing"
