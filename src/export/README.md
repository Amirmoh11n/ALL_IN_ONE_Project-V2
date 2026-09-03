# src/export

V2 serving export is **ONNX** with the stable legacy exporter (`export.onnx.dynamo: false`).
TorchScript web export remains optional. Lite Interpreter / `.ptl` flows were removed.

Validation uses config `atol` / `rtol` plus a dynamic batch check.
Optional INT8: `export.formats.int8: true`.
Filenames include `model_version` (e.g. `brain_tumor_efficientnet_b4_v2.0.0.onnx`).
