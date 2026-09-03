"""Stable deployment exports. Primary web format: ONNX (legacy exporter by default).

Lite Interpreter / mobile .ptl flows are intentionally removed in V2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export a trained classifier to TorchScript and validated ONNX."""

    def __init__(
        self,
        model: nn.Module,
        input_size: int,
        model_version: str = "2.0.0",
        onnx_atol: float = 1e-3,
        onnx_rtol: float = 1e-3,
        use_dynamo: bool = False,
    ) -> None:
        self.model = model.to("cpu").eval()
        self.input_size = int(input_size)
        self.model_version = model_version
        self.onnx_atol = onnx_atol
        self.onnx_rtol = onnx_rtol
        self.use_dynamo = use_dynamo
        self._dummy_input = torch.randn(1, 3, self.input_size, self.input_size)

    def export_web(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traced = torch.jit.trace(self.model, self._dummy_input)
        traced.save(str(output_path))
        loaded = torch.jit.load(str(output_path)).eval()
        self._compare_outputs(loaded)
        return output_path

    def export_onnx(self, output_path: Path, opset_version: int = 18) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = dict(
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=int(opset_version),
        )
        if self.use_dynamo:
            try:
                torch.onnx.export(
                    self.model, self._dummy_input, str(output_path), dynamo=True, **kwargs
                )
            except (TypeError, RuntimeError, ImportError) as exc:
                logger.warning("Dynamo ONNX export failed (%s); using legacy exporter.", exc)
                torch.onnx.export(self.model, self._dummy_input, str(output_path), **kwargs)
        else:
            torch.onnx.export(self.model, self._dummy_input, str(output_path), **kwargs)
        self._validate_onnx(output_path)
        return output_path

    def export_int8(self, onnx_path: Path, output_path: Path) -> Path:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        output_path = Path(output_path)
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )
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
        if not np.allclose(onnx_output, torch_output, atol=self.onnx_atol, rtol=self.onnx_rtol):
            max_diff = float(np.max(np.abs(onnx_output - torch_output)))
            raise ValueError(
                f"ONNX validation failed: max difference={max_diff:.3e} "
                f"(atol={self.onnx_atol}, rtol={self.onnx_rtol})"
            )
        batch5 = np.random.randn(5, 3, self.input_size, self.input_size).astype(np.float32)
        dynamic_out = session.run(None, {"input": batch5})[0]
        if dynamic_out.shape[0] != 5:
            raise ValueError("ONNX dynamic batch validation failed.")

    def export_all(
        self,
        output_dir: Path,
        base_name: str = "model",
        opset_version: int = 18,
        include_int8: bool = False,
        include_torchscript: bool = True,
    ) -> Dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        versioned = f"{base_name}_v{self.model_version}"
        paths: Dict[str, Path] = {}
        if include_torchscript:
            paths["web"] = self.export_web(output_dir / f"{versioned}_web.pt")
        paths["onnx"] = self.export_onnx(output_dir / f"{versioned}.onnx", opset_version)
        # Keep a stable unversioned filename for the default serving path.
        serving = output_dir / f"{base_name}.onnx"
        serving.write_bytes(paths["onnx"].read_bytes())
        paths["onnx_serving"] = serving
        if include_int8:
            paths["onnx_int8"] = self.export_int8(
                paths["onnx"], output_dir / f"{versioned}_int8.onnx"
            )
        manifest = {
            "model_version": self.model_version,
            "input_shape": [1, 3, self.input_size, self.input_size],
            "formats": {k: str(v.name) for k, v in paths.items()},
            "onnx": {
                "exporter": "legacy" if not self.use_dynamo else "dynamo-or-legacy-fallback",
                "atol": self.onnx_atol,
                "rtol": self.onnx_rtol,
            },
            "validation": "ONNX round-trip checked with onnxruntime; Lite Interpreter removed in V2",
        }
        (output_dir / f"{versioned}_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return paths
