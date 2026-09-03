"""Grad-CAM / Grad-CAM++ explanations saved to artifacts (not the web app)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class GradCAMExplainer:
    """Generate Grad-CAM heatmaps for a few samples per class."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str],
        method: str = "gradcam++",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.class_names = class_names
        self.method = method
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._target = self._find_target_layer()
        self._fwd = self._target.register_forward_hook(self._save_activation)
        self._bwd = self._target.register_full_backward_hook(self._save_gradient)

    def close(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def _find_target_layer(self) -> nn.Module:
        backbone = getattr(self.model, "backbone", self.model)
        features = getattr(backbone, "features", None)
        if features is None:
            raise ValueError("Cannot locate convolutional features for Grad-CAM.")
        return features[-1]

    def _save_activation(self, _module, _inputs, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def _cam(self, class_index: int) -> np.ndarray:
        activations = self._activations
        gradients = self._gradients
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")
        if self.method == "gradcam++":
            grads = gradients[0]
            activations_b = activations[0]
            relu_grads = F.relu(grads)
            denom = 2 * grads.pow(2) + (activations_b * grads.pow(3)).sum(dim=(1, 2), keepdim=True)
            denom = torch.where(denom != 0, denom, torch.ones_like(denom))
            alpha = relu_grads.pow(2) / denom
            weights = (alpha * relu_grads).sum(dim=(1, 2))
        else:
            weights = gradients[0].mean(dim=(1, 2))
        cam = (weights[:, None, None] * activations[0]).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def explain_loader(
        self,
        data_loader: DataLoader,
        output_dir: Path,
        samples_per_class: int = 3,
    ) -> List[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        counts = {i: 0 for i in range(len(self.class_names))}
        written: List[Path] = []
        for images, labels in data_loader:
            for i in range(images.size(0)):
                label = int(labels[i].item())
                if counts[label] >= samples_per_class:
                    continue
                tensor = images[i : i + 1].to(self.device)
                self.model.zero_grad(set_to_none=True)
                logits = self.model(tensor)
                score = logits[0, label]
                score.backward()
                cam = self._cam(label)
                path = output_dir / f"{self.class_names[label]}_{counts[label]:02d}.png"
                self._save_overlay(tensor, cam, path)
                written.append(path)
                counts[label] += 1
                if all(v >= samples_per_class for v in counts.values()):
                    logger.info("Wrote %d Grad-CAM images to %s", len(written), output_dir)
                    return written
        logger.info("Wrote %d Grad-CAM images to %s", len(written), output_dir)
        return written

    def _save_overlay(self, tensor: torch.Tensor, cam: np.ndarray, path: Path) -> None:
        import matplotlib.pyplot as plt

        image = tensor[0].detach().cpu()
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)
        image = image.clamp(0, 1).permute(1, 2, 0).numpy()
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR
        )
        cam_np = np.asarray(cam_img) / 255.0
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
        axes[0].imshow(image)
        axes[0].set_title("MRI")
        axes[0].axis("off")
        axes[1].imshow(image)
        axes[1].imshow(cam_np, cmap="jet", alpha=0.45)
        axes[1].set_title(self.method)
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
