"""Command-line entry point for the complete ML lifecycle."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from src.data.pipeline import DataPipeline
from src.engine.multiseed import MultiSeedRunner
from src.engine.trainer import Trainer
from src.evaluate.evaluate import ModelEvaluator
from src.explain.gradcam import GradCAMExplainer
from src.export.export import ModelExporter
from src.inference.inference import InferencePipeline
from src.models.factory import build_model
from src.utils.checkpoint import load_model_checkpoint
from src.utils.config_loader import ConfigLoader
from src.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def load_config(path: str) -> ConfigLoader:
    config = ConfigLoader(Path(path))
    configure_logging(config.get("logging.level", "INFO"))
    return config


def _device(config: ConfigLoader) -> torch.device:
    device_name = str(config.get("training.device", "auto"))
    return torch.device(
        "cuda" if device_name in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    )


def train(config: ConfigLoader, epochs=None):
    pipeline = DataPipeline(config)
    train_loader, val_loader, _ = pipeline.prepare()
    model = build_model(config)
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.fit(epochs)
    print(json.dumps({"split_mode": pipeline.split_mode, "history_keys": list(history.keys())}, indent=2))
    return history


def train_multiseed(config: ConfigLoader, epochs=None):
    summary = MultiSeedRunner(config).run(epochs)
    print(json.dumps(summary["metrics"], indent=2))
    return summary


def evaluate(config: ConfigLoader, checkpoint: Path):
    pipeline = DataPipeline(config)
    _, _, test_loader = pipeline.prepare()
    model = build_model(config, pretrained=False)
    device = _device(config)
    model = load_model_checkpoint(model, checkpoint, device)
    output_dir = config.resolve_path("evaluation.output_dir", "artifacts/evaluation")
    result = ModelEvaluator(
        model,
        test_loader,
        int(config.get("model.num_classes", 4)),
        device=device,
        class_names=config.get("data.class_names"),
        output_dir=output_dir,
        temperature_scaling=bool(config.get("evaluation.temperature_scaling", True)),
        ece_bins=int(config.get("evaluation.ece_bins", 15)),
        split_mode=pipeline.split_mode,
        model_version=str(config.get("export.model_version", "2.0.0")),
    ).evaluate()
    output = result.to_dict()
    output_path = config.resolve_path("artifacts.evaluation_file", "artifacts/evaluation/test_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    (output_dir / "metrics_table.md").write_text(
        result.to_markdown_table(config.get("data.class_names")), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))
    return result


def export_models(config: ConfigLoader, checkpoint: Path):
    model = build_model(config, pretrained=False)
    load_model_checkpoint(model, checkpoint, torch.device("cpu"))
    output_dir = config.resolve_path("export.output_dir", "artifacts/exports")
    exporter = ModelExporter(
        model,
        int(config.get("export.input_size", config.get("data.image_size", 380))),
        model_version=str(config.get("export.model_version", "2.0.0")),
        onnx_atol=float(config.get("export.onnx.atol", 1e-3)),
        onnx_rtol=float(config.get("export.onnx.rtol", 1e-3)),
        use_dynamo=bool(config.get("export.onnx.dynamo", False)),
    )
    paths = exporter.export_all(
        output_dir,
        config.get("export.base_name", "brain_tumor_efficientnet_b4"),
        opset_version=int(config.get("export.onnx.opset_version", 18)),
        include_int8=bool(config.get("export.formats.int8", False)),
        include_torchscript=bool(config.get("export.formats.torchscript", True)),
    )
    result = {key: str(value) for key, value in paths.items()}
    print(json.dumps(result, indent=2))
    return paths


def predict(config: ConfigLoader, checkpoint: Path, image: Path):
    result = InferencePipeline(checkpoint, config).predict(image)
    print(json.dumps(result.to_dict(), indent=2))
    return result


def predict_batch(config: ConfigLoader, checkpoint: Path, folder: Path, output: Path):
    results = InferencePipeline(checkpoint, config).predict_folder(folder, output)
    print(json.dumps({"n": len(results), "csv": str(output)}, indent=2))
    return results


def explain(config: ConfigLoader, checkpoint: Path):
    pipeline = DataPipeline(config)
    _, _, test_loader = pipeline.prepare()
    device = _device(config)
    model = load_model_checkpoint(build_model(config, pretrained=False), checkpoint, device)
    explainer = GradCAMExplainer(
        model,
        device,
        class_names=list(config.get("data.class_names")),
        method=str(config.get("explainability.method", "gradcam++")),
    )
    try:
        paths = explainer.explain_loader(
            test_loader,
            config.resolve_path("explainability.output_dir", "artifacts/explainability"),
            samples_per_class=int(config.get("explainability.samples_per_class", 3)),
        )
    finally:
        explainer.close()
    print(json.dumps([str(p) for p in paths], indent=2))
    return paths


def build_parser():
    parser = argparse.ArgumentParser(
        prog="brain-tumor",
        description="Train, evaluate, export and run inference for the Brain Tumor MRI classifier.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Download/prepare data and train the model.")
    p.add_argument("--epochs", type=int, default=None)

    sub.add_parser("train-multiseed", help="Train 3 seeds and report mean ± std.")

    p = sub.add_parser("evaluate", help="Evaluate a checkpoint on the untouched Testing set.")
    p.add_argument("--checkpoint", default=None)

    p = sub.add_parser("export", help="Export a checkpoint to TorchScript + ONNX.")
    p.add_argument("--checkpoint", default=None)

    p = sub.add_parser("predict", help="Classify one MRI image.")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--image", required=True)

    p = sub.add_parser("predict-batch", help="Classify a folder of MRIs and write CSV.")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--folder", required=True)
    p.add_argument("--output", default="artifacts/evaluation/batch_predictions.csv")

    p = sub.add_parser("explain", help="Write Grad-CAM artifacts for a few samples per class.")
    p.add_argument("--checkpoint", default=None)

    return parser


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    default_checkpoint = config.resolve_path("artifacts.checkpoint", "artifacts/checkpoints/best_model.pt")
    checkpoint = Path(getattr(args, "checkpoint", None) or default_checkpoint)

    if args.command == "train":
        train(config, args.epochs)
    elif args.command == "train-multiseed":
        train_multiseed(config, getattr(args, "epochs", None))
    elif args.command == "evaluate":
        evaluate(config, checkpoint)
    elif args.command == "export":
        export_models(config, checkpoint)
    elif args.command == "predict":
        predict(config, checkpoint, Path(args.image))
    elif args.command == "predict-batch":
        predict_batch(config, checkpoint, Path(args.folder), Path(args.output))
    elif args.command == "explain":
        explain(config, checkpoint)


if __name__ == "__main__":
    main()
