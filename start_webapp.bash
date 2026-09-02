#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ ! -f artifacts/exports/brain_tumor_efficientnet_b3.onnx ]]; then
  echo "ONNX model not found. Run: uv run brain-tumor train && uv run brain-tumor export"
  exit 1
fi
uv run uvicorn webapplication.backend.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
sleep 2
python - <<'PY'
import webbrowser
webbrowser.open("http://127.0.0.1:8000")
PY
wait $PID
