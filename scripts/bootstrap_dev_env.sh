#!/usr/bin/env bash
# Resolves: "No matching distribution found for openreward>=0.1.89" on environments with
# an outdated pip (PyPI index metadata stops at e.g. 0.1.33).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PREFERRED="${1:-python3.11}"
if command -v "$PREFERRED" >/dev/null 2>&1; then
  BASE_PY="$PREFERRED"
else
  echo "No $PREFERRED; using python3. Install Python 3.11+ (pyproject requires-python >=3.11)." >&2
  BASE_PY="python3"
fi

"$BASE_PY" -c 'import sys; v=sys.version_info; assert (v.major, v.minor) >= (3, 11), f"Need Python 3.11+, got {v.major}.{v.minor}"'

"$BASE_PY" -m pip install -U "pip>=24" setuptools wheel
"$BASE_PY" -m venv .venv
.venv/bin/python -m pip install -U "pip>=24" setuptools wheel
.venv/bin/pip install -e ".[dev]"

echo "OK. Example: .venv/bin/python -m pytest tests/"
echo "Optional PyPI smoke test: PYPI_CHECK=1 .venv/bin/python -m pytest tests/test_openreward_packaging.py -v"
