"""Configuration loading and project-root-aware path resolution."""
from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigLoader:
    """Loads YAML configuration and exposes dotted-path access."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.project_root = self.config_path.parent.parent
        self._config: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def raw(self) -> Dict[str, Any]:
        return self._config

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._config
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve_path(self, dotted_key: str, default: str) -> Path:
        """Resolve a config path relative to the project root."""
        value = self.get(dotted_key, default)
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.project_root / path

    def save_snapshot(self, output_path: Path) -> None:
        """Save the exact active configuration for experiment reproducibility."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self._config, f, sort_keys=False)
