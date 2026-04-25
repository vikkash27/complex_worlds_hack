#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ ! -f artifacts/traces/humanoid_baseline_long_horizon.jsonl ] || [ ! -f artifacts/traces/humanoid_trained_long_horizon.jsonl ]; then
  .venv/bin/python scripts/run_demo.py
fi

.venv/bin/python scripts/mujoco/fetch_menagerie_g1.py
.venv/bin/python scripts/mujoco/render_g1_showcase.py --backend auto
