#!/usr/bin/env bash
# One-shot installer: Ollama (where applicable), model pulls, Python venv, dependencies.
# Usage: from this directory, run:  bash install.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ ! -f "setup_local.sh" ]]; then
  echo "install.sh must live next to setup_local.sh (the secondBrainAI app folder)." >&2
  exit 1
fi
exec bash setup_local.sh
