"""
Utility for loading a saved model checkpoint (as produced by
src/engine/trainer.py's Trainer._save_checkpoint) into a model instance.

Shared by evaluation and inference so the checkpoint format is defined in
exactly one place.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a checkpoint's model_state_dict into `model`, in-place, and return it.

    Args:
        model: An instantiated model (e.g. EfficientNetB3Classifier) with the
            same architecture as the one the checkpoint was saved from.
        checkpoint_path: Path to a .pt file saved by Trainer._save_checkpoint,
            containing at least a "model_state_dict" key.
        device: Device to move the model to. Defaults to CPU.

    Returns:
        The same `model` instance, with weights loaded, moved to `device`,
        and set to eval() mode.
    """
    device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded checkpoint from %s (epoch=%s, val_loss=%s)",
        checkpoint_path, checkpoint.get("epoch"), checkpoint.get("val_loss"),
    )
    return model
