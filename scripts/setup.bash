#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
}

mkdir -p data/raw artifacts/checkpoints artifacts/exports artifacts/evaluation artifacts/mlruns

uv venv --python 3.12
uv sync --extra dev

echo
echo "Setup complete."
echo "Run tests:       uv run pytest tests -v"
echo "Train:           uv run brain-tumor train"
echo "Evaluate:        uv run brain-tumor evaluate"
echo "Export:          uv run brain-tumor export"
echo "Predict:         uv run brain-tumor predict --image /path/to/mri.jpg"
echo "MLflow UI:       ./scripts/run.bash mlflow"
