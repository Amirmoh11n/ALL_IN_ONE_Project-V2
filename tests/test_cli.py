"""CLI parser coverage without running training."""

import pytest

pytest.importorskip("torch")

from src.cli import build_parser


def test_parser_requires_command():
    parser = build_parser()
    assert parser.parse_args(["train"]).command == "train"
    assert parser.parse_args(["train-multiseed"]).command == "train-multiseed"
    assert parser.parse_args(["evaluate"]).command == "evaluate"
    assert parser.parse_args(["export"]).command == "export"
    assert parser.parse_args(["explain"]).command == "explain"


def test_predict_requires_image():
    parser = build_parser()
    args = parser.parse_args(["predict", "--image", "slice.png"])
    assert args.image == "slice.png"


def test_predict_batch_defaults():
    parser = build_parser()
    args = parser.parse_args(["predict-batch", "--folder", "data/raw/Testing"])
    assert args.folder == "data/raw/Testing"
    assert args.output.endswith("batch_predictions.csv")
