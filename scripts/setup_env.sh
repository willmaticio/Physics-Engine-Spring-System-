#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-$ROOT_DIR/.venv}"

python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e "$ROOT_DIR"

echo "Virtual environment created at: $VENV_PATH"
echo "Activate it with: source \"$VENV_PATH/bin/activate\""
