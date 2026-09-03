"""Confusion-matrix and ROC plot writers for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np


class EvaluationPlotter:
    """Save PNG evaluation figures under artifacts/evaluation/."""

    def __init__(self, output_dir: Path, class_names: Sequence[str]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = list(class_names)

    def save_confusion_matrix(self, matrix: np.ndarray) -> Path:
        import matplotlib.pyplot as plt

        path = self.output_dir / "confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = range(len(self.class_names))
        ax.set_xticks(list(ticks))
        ax.set_yticks(list(ticks))
        ax.set_xticklabels(self.class_names, rotation=30, ha="right")
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    def save_roc_curves(self, y_true: List[int], y_score: List[List[float]]) -> Path:
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize
        import matplotlib.pyplot as plt

        path = self.output_dir / "roc_curves.png"
        n_classes = len(self.class_names)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        scores = np.asarray(y_score)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        for i, name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC (one-vs-rest)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path
