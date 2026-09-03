"""Production training engine for brain-tumor MRI classification (EfficientNet-B3/B4)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.metrics.accuracy import AccuracyMetric
from src.metrics.f1_score import F1ScoreMetric
from src.metrics.precision import PrecisionMetric
from src.metrics.recall import RecallMetric
from src.metrics.roc_auc import ROCAUCMetric
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Metrics for which higher is better. Everything else is treated as "min".
_MAX_MODE_METRICS = {
    "val_accuracy",
    "val_f1_macro",
    "val_recall_macro",
    "val_precision_macro",
    "val_roc_auc_macro",
}


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for weighted CrossEntropyLoss."""
    counts = Counter(labels)
    total = len(labels)
    if total == 0:
        raise ValueError("Cannot compute class weights from an empty training set.")
    return torch.tensor(
        [total / (num_classes * counts[i]) if counts.get(i, 0) else 0.0 for i in range(num_classes)],
        dtype=torch.float32,
    )


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_score: float) -> None:
        if self.best_score is None or self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def _is_improvement(self, current_score: float) -> bool:
        if self.mode == "min":
            return current_score < self.best_score - self.min_delta
        return current_score > self.best_score + self.min_delta


class Trainer:
    """Train, validate, checkpoint and optionally track an experiment with MLflow.

    Checkpoint selection and early stopping are driven by
    ``training.early_stopping.monitor`` (default: ``val_f1_macro``).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ConfigLoader,
        num_classes: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.num_classes = int(num_classes or config.get("model.num_classes", 4))

        requested = str(config.get("training.device", "auto")).lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; using CPU.")
            requested = "cpu"
        self.device = torch.device(requested)
        self.model.to(self.device)

        labels = [int(label) for _, label in self.train_loader.dataset.samples]
        weight = None
        if config.get("training.use_class_weights", False):
            weight = compute_class_weights(labels, self.num_classes).to(self.device)
            logger.info("Class weights: %s", weight.detach().cpu().tolist())

        label_smoothing = float(config.get("training.label_smoothing", 0.0))
        self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

        self.optimizer = Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=float(config.get("training.learning_rate", 1e-4)),
            weight_decay=float(config.get("training.weight_decay", 0.0)),
        )

        # Monitor metric (config-driven). Default V2: val_f1_macro.
        self.monitor_metric = str(
            config.get("training.early_stopping.monitor", "val_f1_macro")
        )
        self.monitor_mode = "max" if self.monitor_metric in _MAX_MODE_METRICS else "min"
        logger.info(
            "Checkpoint / early-stopping monitor: %s (mode=%s)",
            self.monitor_metric,
            self.monitor_mode,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=self.monitor_mode,
            factor=float(config.get("training.lr_scheduler.factor", 0.5)),
            patience=int(config.get("training.lr_scheduler.patience", 2)),
        )
        self.early_stopping = EarlyStopping(
            patience=int(config.get("training.early_stopping.patience", 5)),
            mode=self.monitor_mode,
            min_delta=float(config.get("training.early_stopping.min_delta", 0.0)),
        )

        self.checkpoint_dir = Path(
            checkpoint_dir
            or config.resolve_path("artifacts.checkpoint_dir", "artifacts/checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_monitor_score: Optional[float] = None
        self.mlflow_enabled = bool(config.get("tracking.mlflow.enabled", False))
        self.use_amp = (
            bool(config.get("training.amp", self.device.type == "cuda"))
            and self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip_norm = config.get("training.gradient_clip_norm", None)

    def fit(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        epochs = int(num_epochs or self.config.get("training.epochs", 30))
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        with self._mlflow_run_context():
            self._log_mlflow_params(epochs)
            snapshot_dir = self.config.resolve_path(
                "artifacts.config_snapshot_dir", "artifacts/runs"
            )
            self.config.save_snapshot(snapshot_dir / "config.snapshot.yaml")
            for epoch in range(1, epochs + 1):
                train_loss = self._train_one_epoch()
                val_loss, y_true, y_pred, y_score = self._validate_one_epoch()
                metrics = self._compute_validation_metrics(y_true, y_pred, y_score)
                metrics["val_loss"] = val_loss

                monitor_value = self._resolve_monitor_value(val_loss, metrics)
                self.scheduler.step(monitor_value)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                for key, value in metrics.items():
                    history.setdefault(key, []).append(value)

                logger.info(
                    "Epoch %d/%d | train_loss=%.4f val_loss=%.4f val_acc=%.4f "
                    "val_f1=%.4f monitor(%s)=%.4f lr=%.3g",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                    metrics["val_accuracy"],
                    metrics["val_f1_macro"],
                    self.monitor_metric,
                    monitor_value,
                    self.optimizer.param_groups[0]["lr"],
                )
                self._log_mlflow_metrics(epoch, train_loss, val_loss, metrics)

                if self._is_best(monitor_value):
                    self.best_monitor_score = monitor_value
                    self._save_checkpoint(epoch, val_loss, metrics, monitor_value)

                self.early_stopping.step(monitor_value)
                if self.early_stopping.should_stop:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        history_path = self.checkpoint_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return history

    def _resolve_monitor_value(self, val_loss: float, metrics: Dict[str, float]) -> float:
        if self.monitor_metric == "val_loss":
            return val_loss
        if self.monitor_metric not in metrics:
            logger.warning(
                "Monitor metric '%s' not found in computed metrics; falling back to val_loss.",
                self.monitor_metric,
            )
            return val_loss
        return float(metrics[self.monitor_metric])

    def _is_best(self, current: float) -> bool:
        if self.best_monitor_score is None:
            return True
        if self.monitor_mode == "min":
            return current < self.best_monitor_score
        return current > self.best_monitor_score

    def _train_one_epoch(self) -> float:
        self.model.train()
        running = 0.0
        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                mixed, targets_a, targets_b, lam = self._maybe_mix(images, labels)
                logits = self.model(mixed)
                if targets_b is None:
                    loss = self.criterion(logits, targets_a)
                else:
                    loss = lam * self.criterion(logits, targets_a) + (1.0 - lam) * self.criterion(
                        logits, targets_b
                    )
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.grad_clip_norm)
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.grad_clip_norm)
                    )
                self.optimizer.step()
            running += loss.item() * images.size(0)
        return running / len(self.train_loader.dataset)

    def _maybe_mix(self, images: torch.Tensor, labels: torch.Tensor):
        mixup_alpha = float(self.config.get("data.augmentation.mixup_alpha", 0.0) or 0.0)
        cutmix_alpha = float(self.config.get("data.augmentation.cutmix_alpha", 0.0) or 0.0)
        if mixup_alpha <= 0 and cutmix_alpha <= 0:
            return images, labels, None, 1.0
        if cutmix_alpha > 0:
            return self._cutmix(images, labels, cutmix_alpha)
        return self._mixup(images, labels, mixup_alpha)

    def _mixup(self, images: torch.Tensor, labels: torch.Tensor, alpha: float):
        lam = float(np.random.beta(alpha, alpha))
        index = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1.0 - lam) * images[index]
        return mixed, labels, labels[index], lam

    def _cutmix(self, images: torch.Tensor, labels: torch.Tensor, alpha: float):
        lam = float(np.random.beta(alpha, alpha))
        batch, _, height, width = images.shape
        index = torch.randperm(batch, device=images.device)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        cx = int(np.random.randint(0, width))
        cy = int(np.random.randint(0, height))
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
        return mixed, labels, labels[index], lam


    def _validate_one_epoch(
        self,
    ) -> Tuple[float, List[int], List[int], List[List[float]]]:
        self.model.eval()
        running: float = 0.0
        y_true: List[int] = []
        y_pred: List[int] = []
        y_score: List[List[float]] = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(images)
                running += self.criterion(outputs, labels).item() * images.size(0)
                probs = TF.softmax(outputs, dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(probs.argmax(1).cpu().tolist())
                y_score.extend(probs.cpu().tolist())
        return running / len(self.val_loader.dataset), y_true, y_pred, y_score

    def _compute_validation_metrics(
        self, y_true: List[int], y_pred: List[int], y_score: List[List[float]]
    ) -> Dict[str, float]:
        return {
            "val_accuracy": AccuracyMetric.compute(y_true, y_pred),
            "val_recall_macro": RecallMetric.compute(y_true, y_pred, average="macro"),
            "val_precision_macro": PrecisionMetric.compute(y_true, y_pred, average="macro"),
            "val_f1_macro": F1ScoreMetric.compute(y_true, y_pred, average="macro"),
            "val_roc_auc_macro": ROCAUCMetric.compute(
                y_true, y_score, num_classes=self.num_classes
            ),
        }

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        metrics: Dict[str, float],
        monitor_value: float,
    ) -> Path:
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "val_metrics": metrics,
                "monitor_metric": self.monitor_metric,
                "monitor_value": monitor_value,
                "architecture": self.config.get("model.architecture"),
                "class_names": self.config.get("data.class_names"),
                "num_classes": self.num_classes,
                "image_size": self.config.get("data.image_size"),
                "model_version": self.config.get("export.model_version", "2.0.0"),
            },
            path,
        )
        logger.info(
            "Saved best checkpoint -> %s (%s=%.4f)",
            path,
            self.monitor_metric,
            monitor_value,
        )
        return path

    def _mlflow_run_context(self):
        if not self.mlflow_enabled:
            return nullcontext()
        import mlflow

        uri = self.config.get(
            "tracking.mlflow.tracking_uri", "sqlite:///artifacts/mlruns/mlflow.db"
        )
        Path(
            self.config.resolve_path(
                "tracking.mlflow.artifact_location", "artifacts/mlruns"
            )
        ).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(
            self.config.get(
                "tracking.mlflow.experiment_name", "brain_tumor_classification"
            )
        )
        return mlflow.start_run()

    def _log_mlflow_params(self, epochs: int) -> None:
        if not self.mlflow_enabled:
            return
        import mlflow

        mlflow.log_params(
            {
                "learning_rate": self.config.get("training.learning_rate", 1e-4),
                "epochs": epochs,
                "batch_size": self.train_loader.batch_size,
                "num_classes": self.num_classes,
                "device": str(self.device),
                "architecture": self.config.get(
                    "model.architecture", "efficientnet_b4"
                ),
                "monitor_metric": self.monitor_metric,
                "image_size": self.config.get("data.image_size"),
            }
        )

    def _log_mlflow_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, float],
    ) -> None:
        if self.mlflow_enabled:
            import mlflow

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, **metrics},
                step=epoch,
            )
