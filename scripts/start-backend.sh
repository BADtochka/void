#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Python virtual environment not found: %s\n' "$PYTHON" >&2
  printf 'Create it and install voice_core/requirements.txt first.\n' >&2
  exit 1
fi

cd "$PROJECT_DIR"

exec "$PYTHON" -m uvicorn voice_core.app:app \
  --app-dir voice_core \
  --host "${VOICE_CORE_HOST:-127.0.0.1}" \
  --port "${VOICE_CORE_PORT:-8765}" \
  "$@"
