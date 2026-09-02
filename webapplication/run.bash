#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run uvicorn webapplication.backend.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
