"""ConfigLoader dotted-key and snapshot tests."""

from pathlib import Path

import yaml

from src.utils.config_loader import ConfigLoader


def test_get_nested_and_default(tmp_path: Path):
    path = tmp_path / "configs" / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("model:\n  architecture: efficientnet_b4\nlogging:\n  level: INFO\n", encoding="utf-8")
    config = ConfigLoader(path)
    assert config.get("model.architecture") == "efficientnet_b4"
    assert config.get("missing.key", "fallback") == "fallback"


def test_resolve_relative_to_project_root(tmp_path: Path):
    path = tmp_path / "configs" / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("artifacts:\n  checkpoint: artifacts/checkpoints/best_model.pt\n", encoding="utf-8")
    config = ConfigLoader(path)
    resolved = config.resolve_path("artifacts.checkpoint", "x.pt")
    assert resolved == tmp_path / "artifacts/checkpoints/best_model.pt"


def test_save_snapshot_roundtrip(tmp_path: Path):
    path = tmp_path / "configs" / "config.yaml"
    path.parent.mkdir(parents=True)
    payload = {"training": {"epochs": 30, "seeds": [42, 43, 44]}}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = ConfigLoader(path)
    snapshot = tmp_path / "run" / "config.snapshot.yaml"
    config.save_snapshot(snapshot)
    loaded = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert loaded["training"]["seeds"] == [42, 43, 44]
