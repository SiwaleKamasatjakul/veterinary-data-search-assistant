#!/usr/bin/env bash
# Start the API from anywhere — resolves to this script's directory.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec uvicorn src.LLM_Process_API:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --reload
