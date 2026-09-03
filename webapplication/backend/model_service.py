from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from src.utils.config_loader import ConfigLoader


def _download_s3(uri: str, destination: Path) -> None:
    if not uri.startswith("s3://"):
        raise ValueError("MODEL_S3_URI must start with s3://")
    bucket_key = uri[5:].split("/", 1)
    if len(bucket_key) != 2:
        raise ValueError("MODEL_S3_URI must be s3://bucket/key")
    import boto3
    destination.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket_key[0], bucket_key[1], str(destination))


class ONNXModelService:
    """Server-side ONNX inference. The client never loads the model."""

    def __init__(self, model_path: Path, config_path: Path, model_s3_uri: str = ""):
        if model_s3_uri and not model_path.exists():
            _download_s3(model_s3_uri, model_path)
        self.model_path = Path(model_path)
        self.config = ConfigLoader(config_path)
        self.classes = list(self.config.get("data.class_names", ["glioma", "meningioma", "notumor", "pituitary"]))
        self.image_size = int(self.config.get("export.input_size", self.config.get("data.image_size", 380)))
        self.mean = np.asarray(self.config.get("data.normalization.mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.asarray(self.config.get("data.normalization.std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.session = None
        self.input_name = None
        self.output_name = None
        self.provider = "unavailable"
        self._load_error = None
        self._load()

    @property
    def ready(self) -> bool:
        return self.session is not None

    @property
    def load_error(self):
        return self._load_error

    def _load(self) -> None:
        if not self.model_path.exists():
            self._load_error = f"Model file not found: {self.model_path}"
            return
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.provider = self.session.get_providers()[0]
        except Exception as exc:
            self._load_error = str(exc)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        array = np.transpose(array, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(array, dtype=np.float32)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float32)
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def predict(self, image: Image.Image) -> Dict:
        if not self.ready:
            raise RuntimeError("Exported ONNX model is not available. Train and export the model first.")
        tensor = self.preprocess(image)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
        probabilities = self.softmax(logits)[0]
        index = int(np.argmax(probabilities))
        values = {name: float(probabilities[i]) for i, name in enumerate(self.classes)}
        ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
        return {
            "predicted_class": self.classes[index],
            "confidence": float(probabilities[index]),
            "confidence_percentage": float(probabilities[index] * 100.0),
            "probabilities": values,
            "ranked_probabilities": [
                {"class_name": name, "value": value, "percentage": value * 100.0}
                for name, value in ranked
            ],
            "model": self.model_path.name,
            "device": self.provider,
            "warning": "Research/educational use only. This is not a medical diagnosis.",
        }
