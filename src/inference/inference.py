"""Single-image inference pipeline for the trained classifier."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.augment import AugmentationFactory
from src.data.classes import TumorClasses
from src.models.factory import build_model
from src.utils.checkpoint import load_model_checkpoint
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class InferencePipeline:
    """Load a checkpoint once and classify many uploaded MRI images."""

    def __init__(
        self, checkpoint_path: Path, config: ConfigLoader,
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
        if len(self.class_names) != int(config.get("model.num_classes", len(self.class_names))):
            raise ValueError("data.class_names and model.num_classes must match.")

    def predict(self, image: Union[str, Path, Image.Image]) -> PredictionResult:
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
        )

    @staticmethod
    def _to_pil_image(image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        with Image.open(image) as pil:
            return pil.convert("RGB")
