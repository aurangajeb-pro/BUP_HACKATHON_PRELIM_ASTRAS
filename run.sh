#!/usr/bin/env bash
# Start the API after the model has been configured as described in README.md.
set -euo pipefail
cd "$(dirname "$0")"
GRIDWISE_PYTHON="${PYTHON:-$(command -v python3.12 || command -v python3.11 || command -v python3)}"
if [ ! -d .venv ]; then
  "$GRIDWISE_PYTHON" -m venv .venv
fi
# Git Bash on Windows uses Scripts, while Linux/macOS use bin.
GRIDWISE_VENV_BIN=.venv/bin
if [ -d .venv/Scripts ]; then GRIDWISE_VENV_BIN=.venv/Scripts; fi
"$GRIDWISE_VENV_BIN/python" -m pip install -q -r requirements.txt
GRIDWISE_ENV_ARGS=()
if [ -f .env ]; then GRIDWISE_ENV_ARGS=(--env-file .env); fi
exec "$GRIDWISE_VENV_BIN/python" -m uvicorn app.main:app \
  --host 0.0.0.0 --port 8000 --workers 1 "${GRIDWISE_ENV_ARGS[@]}"
