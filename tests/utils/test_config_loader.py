"""Tests for src/utils/config_loader.py (ConfigLoader)."""

from src.utils.config_loader import ConfigLoader


def test_dotted_get_reads_nested_value(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "data:\n"
        "  val_split: 0.15\n"
        "  normalization:\n"
        "    mean: [0.485, 0.456, 0.406]\n"
    )
    config = ConfigLoader(config_file)

    assert config.get("data.val_split") == 0.15
    assert config.get("data.normalization.mean") == [0.485, 0.456, 0.406]


def test_dotted_get_returns_default_for_missing_key(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("data:\n  val_split: 0.15\n")
    config = ConfigLoader(config_file)

    assert config.get("data.not_a_real_key", "fallback") == "fallback"
    assert config.get("not_a_real_section.foo") is None


def test_raw_returns_full_dict(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("a: 1\nb: 2\n")
    config = ConfigLoader(config_file)

    assert config.raw == {"a": 1, "b": 2}
