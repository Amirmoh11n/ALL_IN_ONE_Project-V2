# src/export

Converts a trained checkpoint into the 4 deployable formats required by the
project.

- `export.py` — `ModelExporter`:
  - `export_web()` — plain TorchScript (`.pt`), self-contained for the FastAPI/AWS backend.
  - `export_gpu()` — frozen TorchScript (`.pt`; constant-folded, e.g. batch-norm
    folding) for GPU serving.
  - `export_mobile()` — PyTorch Lite Interpreter (`.ptl`) for Android.
  - `export_onnx()` — ONNX (`.onnx`, opset 18, dynamic batch axis), validated
    with `onnx.checker` + a real `onnxruntime` inference pass compared against
    the PyTorch output.
  - `export_all()` — runs all four, returns a dict of output paths.

## Known trade-offs / deprecation notes (found while implementing, worth knowing)

- **`torch.jit.optimize_for_inference` was evaluated but is NOT used** for the
  GPU export: in this environment's torch build, its output fails to reload
  via `torch.jit.load` (a save/load round-trip bug). `torch.jit.freeze` alone
  round-trips correctly and still applies real optimizations, so it's used instead.
- **Mobile's `optimize_for_mobile`** requires an XNNPACK-enabled torch build.
  If unavailable, `export_mobile()` automatically falls back to a plain traced
  `.ptl` file (still valid/loadable) and logs a warning. Re-export on an
  XNNPACK-enabled build for the extra optimization.
- **PyTorch Lite Interpreter itself is deprecated upstream** in favor of
  [ExecuTorch](https://docs.pytorch.org/executorch/). It still works and is
  what this exporter produces today; migrating to ExecuTorch would be a
  separate, larger decision.
- More broadly, this torch build emits deprecation warnings for the entire
  `torch.jit.*` (TorchScript) API surface, pointing toward `torch.compile` /
  `torch.export` as the long-term direction. TorchScript still works correctly
  here (tests pass), but this is worth knowing if PyTorch's ecosystem shifts
  further before this project ships.
