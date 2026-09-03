"""Multi-seed training: run 3–5 seeds and report mean ± std of primary metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.data.pipeline import DataPipeline, seed_everything
from src.engine.trainer import Trainer
from src.evaluate.evaluate import ModelEvaluator
from src.models.factory import build_model
from src.utils.checkpoint import load_model_checkpoint
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class MultiSeedRunner:
    """Train independent seeds and aggregate test metrics."""

    def __init__(self, config: ConfigLoader) -> None:
        self.config = config

    def run(self, epochs=None) -> Dict:
        seeds = list(self.config.get("training.seeds", [42, 43, 44]))
        records: List[Dict] = []
        for seed in seeds:
            logger.info("=== Multi-seed run seed=%s ===", seed)
            seed_everything(int(seed))
            pipeline = DataPipeline(self.config)
            train_loader, val_loader, test_loader = pipeline.prepare(seed=int(seed))
            model = build_model(self.config)
            checkpoint_dir = self.config.resolve_path(
                "artifacts.checkpoint_dir", "artifacts/checkpoints"
            ) / f"seed_{seed}"
            trainer = Trainer(
                model, train_loader, val_loader, self.config, checkpoint_dir=checkpoint_dir
            )
            trainer.fit(epochs)
            eval_model = build_model(self.config, pretrained=False)
            eval_model = load_model_checkpoint(
                eval_model, checkpoint_dir / "best_model.pt", trainer.device
            )
            result = ModelEvaluator(
                eval_model,
                test_loader,
                int(self.config.get("model.num_classes", 4)),
                device=trainer.device,
                class_names=self.config.get("data.class_names"),
                split_mode=pipeline.split_mode,
                model_version=str(self.config.get("export.model_version", "2.0.0")),
            ).evaluate()
            payload = result.to_dict()
            payload["seed"] = int(seed)
            records.append(payload)

        summary = {"n_seeds": len(records), "seeds": seeds, "metrics": {}}
        keys = [
            "accuracy",
            "recall_macro",
            "f1_macro",
            "precision_macro",
            "roc_auc_macro",
            "specificity_macro",
            "ece",
        ]
        for key in keys:
            values = [float(item[key]) for item in records]
            summary["metrics"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
                "values": values,
            }
        summary["runs"] = records
        output = self.config.resolve_path(
            "evaluation.output_dir", "artifacts/evaluation"
        ) / "multiseed_metrics.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Multi-seed summary written to %s", output)
        return summary
