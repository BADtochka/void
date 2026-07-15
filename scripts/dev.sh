#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BACKEND_PID=""
BOT_PID=""

cleanup() {
  trap - EXIT INT TERM
  for pid in "$BOT_PID" "$BACKEND_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$BOT_PID" "$BACKEND_PID"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

cd "$PROJECT_DIR"

"$SCRIPT_DIR/start-backend.sh" \
  --reload \
  --reload-dir "$PROJECT_DIR/voice_core/voice_core" &
BACKEND_PID=$!

bun --watch src/index.ts &
BOT_PID=$!

set +e
wait -n "$BACKEND_PID" "$BOT_PID"
status=$?
set -e
exit "$status"
