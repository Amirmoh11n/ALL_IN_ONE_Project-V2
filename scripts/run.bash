#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-}"
shift || true

case "$COMMAND" in
  train|evaluate|export|predict)
    uv run brain-tumor "$COMMAND" "$@"
    ;;
  test)
    uv run pytest tests -v
    ;;
  mlflow)
    uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlruns/mlflow.db --host 127.0.0.1 --port 5000
    ;;
  *)
    echo "Usage:"
    echo "  ./scripts/run.bash train [--epochs N]"
    echo "  ./scripts/run.bash evaluate [--checkpoint PATH]"
    echo "  ./scripts/run.bash export [--checkpoint PATH]"
    echo "  ./scripts/run.bash predict --image PATH [--checkpoint PATH]"
    echo "  ./scripts/run.bash test"
    echo "  ./scripts/run.bash mlflow"
    exit 1
    ;;
esac
