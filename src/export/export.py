"""Validated deployment exports: TorchScript, mobile Lite, GPU TorchScript and ONNX."""
import logging
from pathlib import Path
from typing import Dict
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

logger = logging.getLogger(__name__)


class ModelExporter:
    def __init__(self, model: nn.Module, input_size: int) -> None:
        self.model = model.to("cpu").eval()
        self.input_size = int(input_size)
        self._dummy_input = torch.randn(1, 3, self.input_size, self.input_size)

    def export_web(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traced = torch.jit.trace(self.model, self._dummy_input)
        traced.save(str(output_path))
        loaded = torch.jit.load(str(output_path)).eval()
        self._compare_outputs(loaded)
        return output_path

    def export_gpu(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traced = torch.jit.trace(self.model, self._dummy_input)
        frozen = torch.jit.freeze(traced)
        frozen.save(str(output_path))
        loaded = torch.jit.load(str(output_path)).eval()
        self._compare_outputs(loaded, atol=1e-4, rtol=1e-4)
        return output_path

    def export_mobile(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traced = torch.jit.trace(self.model, self._dummy_input)
        try:
            mobile_module = optimize_for_mobile(traced)
        except (RuntimeError, AttributeError) as exc:
            logger.warning("Mobile optimization unavailable: %s; using traced module.", exc)
            mobile_module = traced
        mobile_module._save_for_lite_interpreter(str(output_path))
        # Round-trip validation using the Lite Interpreter loader.
        from torch.jit.mobile import _load_for_lite_interpreter
        loaded = _load_for_lite_interpreter(str(output_path))
        with torch.no_grad():
            ref = self.model(self._dummy_input)
            out = loaded(self._dummy_input)
        torch.testing.assert_close(ref, out, atol=1e-4, rtol=1e-4)
        return output_path

    def export_onnx(self, output_path: Path, opset_version: int = 18) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = dict(
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=int(opset_version),
        )
        try:
            # Modern PyTorch exporter.
            torch.onnx.export(self.model, self._dummy_input, str(output_path), dynamo=True, **kwargs)
        except (TypeError, RuntimeError, ImportError) as exc:
            # Compatibility path for older PyTorch / missing onnxscript.
            logger.warning("Dynamo ONNX export unavailable (%s); using legacy exporter.", exc)
            torch.onnx.export(self.model, self._dummy_input, str(output_path), **kwargs)
        self._validate_onnx(output_path)
        return output_path

    def _compare_outputs(self, loaded: nn.Module, atol=1e-5, rtol=1e-5) -> None:
        with torch.no_grad():
            ref = self.model(self._dummy_input)
            out = loaded(self._dummy_input)
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def _validate_onnx(self, onnx_path: Path) -> None:
        import onnx
        import onnxruntime as ort
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_output = session.run(None, {"input": self._dummy_input.numpy()})[0]
        with torch.no_grad():
            torch_output = self.model(self._dummy_input).numpy()
        max_diff = float(np.max(np.abs(onnx_output - torch_output)))
        if max_diff > 1e-3:
            raise ValueError(f"ONNX validation failed: max difference={max_diff:.3e}")
        # Confirm dynamic batch axis is actually usable.
        batch5 = np.random.randn(5, 3, self.input_size, self.input_size).astype(np.float32)
        dynamic_out = session.run(None, {"input": batch5})[0]
        if dynamic_out.shape[0] != 5:
            raise ValueError("ONNX dynamic batch validation failed.")

    def export_all(self, output_dir: Path, base_name: str = "model", opset_version: int = 18) -> Dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "web": self.export_web(output_dir / f"{base_name}_web.pt"),
            "mobile": self.export_mobile(output_dir / f"{base_name}_mobile.ptl"),
            "gpu": self.export_gpu(output_dir / f"{base_name}_gpu.pt"),
            "onnx": self.export_onnx(output_dir / f"{base_name}.onnx", opset_version),
        }
        manifest = {
            "input_shape": [1, 3, self.input_size, self.input_size],
            "formats": {k: str(v.name) for k, v in paths.items()},
            "validation": "all exports were round-trip validated; ONNX was checked with onnxruntime",
        }
        (output_dir / f"{base_name}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return paths
