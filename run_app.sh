#!/usr/bin/env bash
# Launch the Streamlit UI after install.sh / setup_local.sh has created secondbrain_env.
# Usage: bash run_app.sh   OR   chmod +x run_app.sh && ./run_app.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ ! -d "secondbrain_env" ]]; then
  echo "Virtual env not found. Run:  bash install.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "secondbrain_env/bin/activate"
exec streamlit run app.py "$@"
