# src/inference

The inference pipeline: loads a trained checkpoint and classifies a single MRI
image. This is what `webapplication/backend` calls when a user uploads an image.

- `inference.py`:
  - `InferencePipeline` — loads an `EfficientNetB3Classifier` checkpoint once
    (via `src/utils/checkpoint.py`), builds the eval-only transform (resize +
    ImageNet normalization, matching training), and exposes `.predict(image)`.
  - `PredictionResult` — dataclass: `predicted_class` (str), `confidence`
    (float), `probabilities` (dict of all 4 class names -> probability).
    `.to_dict()` gives a JSON-friendly shape for the FastAPI response.

`predict()` accepts a file path/str or an already-loaded PIL Image, and
auto-converts grayscale/other modes to RGB.
