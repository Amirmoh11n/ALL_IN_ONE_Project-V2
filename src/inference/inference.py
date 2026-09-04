"""Single-image and batch-folder inference for the trained classifier."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.augment import AugmentationFactory
from src.data.classes import TumorClasses
from src.models.factory import build_model
from src.utils.checkpoint import load_model_checkpoint
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str = "2.0.0"
    path: Optional[str] = None

    def to_dict(self) -> Dict:
        payload = {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "model_version": self.model_version,
        }
        if self.path:
            payload["path"] = self.path
        return payload


class InferencePipeline:
    """Load a checkpoint once and classify many MRI images."""

    def __init__(
        self,
        checkpoint_path: Path,
        config: ConfigLoader,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        requested = str(config.get("training.device", "auto")).lower()
        if device is not None:
            self.device = device
        else:
            self.device = torch.device(
                "cuda" if requested in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
            )
        model = build_model(config, pretrained=False)
        self.model = load_model_checkpoint(model, Path(checkpoint_path), self.device)
        image_size = int(config.get("data.image_size", 380))
        self.transform = AugmentationFactory.build_eval_transforms(
            image_size, config.get("data.normalization.mean"), config.get("data.normalization.std")
        )
        self.class_names = config.get("data.class_names", TumorClasses.NAMES)
        self.model_version = str(config.get("export.model_version", "2.0.0"))
        if len(self.class_names) != int(config.get("model.num_classes", len(self.class_names))):
            raise ValueError("data.class_names and model.num_classes must match.")

    def predict(self, image: Union[str, Path, Image.Image]) -> PredictionResult:
        path_str = str(image) if isinstance(image, (str, Path)) else None
        pil_image = self._to_pil_image(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = F.softmax(self.model(tensor), dim=1)[0]
        index = int(probabilities.argmax().item())
        probability_dict = {
            self.class_names[i]: float(probabilities[i].item())
            for i in range(len(self.class_names))
        }
        return PredictionResult(
            predicted_class=self.class_names[index],
            confidence=float(probabilities[index].item()),
            probabilities=probability_dict,
            model_version=self.model_version,
            path=path_str,
        )

    def predict_folder(self, folder: Path, output_csv: Path) -> List[PredictionResult]:
        folder = Path(folder)
        images = sorted(
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        results = [self.predict(path) for path in images]
        self.write_csv(results, output_csv, self.class_names)
        logger.info("Wrote %d predictions to %s", len(results), output_csv)
        return results

    @staticmethod
    def write_csv(
        results: List[PredictionResult],
        output_csv: Path,
        class_names: List[str],
    ) -> Path:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["path", "predicted_class", "confidence", "model_version", *class_names]
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                row = {
                    "path": result.path,
                    "predicted_class": result.predicted_class,
                    "confidence": f"{result.confidence:.6f}",
                    "model_version": result.model_version,
                }
                row.update({name: f"{value:.6f}" for name, value in result.probabilities.items()})
                writer.writerow(row)
        return output_csv

    @staticmethod
    def _to_pil_image(image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        with Image.open(image) as pil:
            return pil.convert("RGB")
