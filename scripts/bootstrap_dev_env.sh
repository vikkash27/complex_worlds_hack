#!/usr/bin/env bash
# Resolves: "No matching distribution found for openreward>=0.1.89" on environments with
# an outdated pip (PyPI index metadata stops at e.g. 0.1.33).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PREFERRED="${1:-python3.11}"
if ! command -v "$PREFERRED" >/dev/null 2>&1; then
  echo "Error: $PREFERRED not found. This project needs Python 3.11+ (see pyproject requires-python >=3.11)." >&2
  echo "On Ubuntu, install then re-run, e.g.:" >&2
  echo "  sudo apt-get update && sudo apt-get install -y ${PREFERRED} ${PREFERRED}-venv" >&2
  echo "  rm -rf .venv" >&2
  echo "  $0 ${PREFERRED}" >&2
  exit 1
fi
BASE_PY="$PREFERRED"
"$BASE_PY" -c 'import sys; v=sys.version_info; assert (v.major, v.minor) >= (3, 11), f"Need Python 3.11+, got {v.major}.{v.minor}"'

if [ -d .venv ] && ! .venv/bin/python -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
  echo "Removing .venv: it was not created with Python 3.11+ (required for robocerebra-rl / pyproject)." >&2
  rm -rf .venv
fi

"$BASE_PY" -m pip install -U "pip>=24" setuptools wheel
"$BASE_PY" -m venv .venv
.venv/bin/python -m pip install -U "pip>=24" setuptools wheel
.venv/bin/pip install -e ".[dev]"

echo "OK. Example: .venv/bin/python -m pytest tests/"
echo "Optional PyPI smoke test: PYPI_CHECK=1 .venv/bin/python -m pytest tests/test_openreward_packaging.py -v"
