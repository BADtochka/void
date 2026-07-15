#!/usr/bin/env bash
set -u

failed=0

check_command() {
  local command_name="$1"
  local hint="$2"
  if command -v "$command_name" >/dev/null 2>&1; then
    printf 'ok   %s: %s\n' "$command_name" "$(command -v "$command_name")"
  else
    printf 'miss %s: %s\n' "$command_name" "$hint"
    failed=1
  fi
}

check_command bun "install Bun 1.2 or newer"
check_command python3 "install Python 3.11-3.13"

if [[ ! -f .env ]]; then
  printf 'miss .env: copy .env.example to .env and fill in Discord credentials\n'
  failed=1
else
  printf 'ok   .env\n'
fi

if [[ -x .venv/bin/python ]]; then
  printf 'ok   Python virtual environment\n'
else
  printf 'miss .venv: run python3 -m venv .venv and install voice_core/requirements.txt\n'
  failed=1
fi

if curl --silent --fail --max-time 2 http://127.0.0.1:1234/v1/models >/dev/null; then
  printf 'ok   LM Studio API\n'
else
  printf 'miss LM Studio API: start the local server on http://127.0.0.1:1234\n'
  failed=1
fi

exit "$failed"
