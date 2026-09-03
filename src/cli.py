"""Command-line entry point for the complete ML lifecycle."""
import argparse
import json
import logging
from pathlib import Path
import torch

from src.data.pipeline import DataPipeline
from src.engine.trainer import Trainer
from src.evaluate.evaluate import ModelEvaluator
from src.export.export import ModelExporter
from src.inference.inference import InferencePipeline
from src.models.factory import build_model
from src.utils.checkpoint import load_model_checkpoint
from src.utils.config_loader import ConfigLoader
from src.utils.logging_setup import configure_logging


def load_config(path: str) -> ConfigLoader:
    config = ConfigLoader(Path(path))
    configure_logging(config.get("logging.level", "INFO"))
    return config


def train(config: ConfigLoader, epochs=None):
    train_loader, val_loader, _ = DataPipeline(config).prepare()
    model = build_model(config)
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.fit(epochs)
    print(json.dumps(history, indent=2))
    return history


def evaluate(config: ConfigLoader, checkpoint: Path):
    _, _, test_loader = DataPipeline(config).prepare()
    model = build_model(config, pretrained=False)
    device_name = config.get("training.device", "auto")
    device = torch.device("cuda" if device_name in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    model = load_model_checkpoint(model, checkpoint, device)
    result = ModelEvaluator(
        model, test_loader, int(config.get("model.num_classes", 4)), device=device
    ).evaluate()
    output = result.to_dict()
    output_path = config.resolve_path("artifacts.evaluation_file", "artifacts/evaluation/test_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return result


def export_models(config: ConfigLoader, checkpoint: Path):
    model = build_model(config, pretrained=False)
    device = torch.device("cpu")
    load_model_checkpoint(model, checkpoint, device)
    output_dir = config.resolve_path("export.output_dir", "artifacts/exports")
    exporter = ModelExporter(model, int(config.get("export.input_size", config.get("data.image_size", 380))))
    paths = exporter.export_all(output_dir, config.get("export.base_name", "brain_tumor"))
    result = {key: str(value) for key, value in paths.items()}
    print(json.dumps(result, indent=2))
    return paths


def predict(config: ConfigLoader, checkpoint: Path, image: Path):
    result = InferencePipeline(checkpoint, config).predict(image)
    print(json.dumps(result.to_dict(), indent=2))
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        prog="brain-tumor",
        description="Train, evaluate, export and run inference for the Brain Tumor MRI classifier.",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Download/prepare data and train the model.")
    p.add_argument("--epochs", type=int, default=None)

    p = sub.add_parser("evaluate", help="Evaluate a checkpoint on the untouched Testing set.")
    p.add_argument("--checkpoint", default=None)

    p = sub.add_parser("export", help="Export a checkpoint to all configured deployment formats.")
    p.add_argument("--checkpoint", default=None)

    p = sub.add_parser("predict", help="Classify one MRI image.")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--image", required=True)

    return parser


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    default_checkpoint = config.resolve_path("artifacts.checkpoint", "artifacts/checkpoints/best_model.pt")

    if args.command == "train":
        train(config, args.epochs)
    elif args.command == "evaluate":
        evaluate(config, Path(args.checkpoint) if args.checkpoint else default_checkpoint)
    elif args.command == "export":
        export_models(config, Path(args.checkpoint) if args.checkpoint else default_checkpoint)
    elif args.command == "predict":
        predict(config, Path(args.checkpoint) if args.checkpoint else default_checkpoint, Path(args.image))


if __name__ == "__main__":
    main()
