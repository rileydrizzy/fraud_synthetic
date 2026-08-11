#!/usr/bin/env bash
#
# setup_env.sh -- create .venv and install requirements.txt into it.
#
# Usage:
#   ./setup_env.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$HERE/requirements.txt"

echo
echo "Done. Activate with:"
echo "    source .venv/bin/activate"
