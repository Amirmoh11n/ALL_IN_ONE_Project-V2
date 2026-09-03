"""Temperature scaling and Expected Calibration Error (ECE)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Learns a single temperature T to calibrate logits on a hold-out set."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-4)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        logits = logits.detach()
        labels = labels.detach()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().item())


class ECEMetric:
    """Expected Calibration Error with equal-width confidence bins."""

    @staticmethod
    def compute(
        y_true: Sequence[int],
        y_prob: Sequence[Sequence[float]],
        n_bins: int = 15,
    ) -> float:
        y_true_arr = np.asarray(y_true)
        probs = np.asarray(y_prob, dtype=np.float64)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == y_true_arr).astype(np.float64)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(y_true_arr)
        if n == 0:
            return 0.0
        for i in range(n_bins):
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
            if not np.any(mask):
                continue
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return float(ece)
