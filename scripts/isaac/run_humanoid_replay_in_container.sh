#!/usr/bin/env bash
# Run inside the Isaac Sim container (paths match hackathon_demo_runbook.md).
# Do not replace this script with "..." placeholders in your shell.
set -euo pipefail
cd /workspace/complex_worlds_hack
exec /isaac-sim/python.sh -u scripts/isaac/replay_breakfast_tray.py \
  --baseline-trace artifacts/traces/humanoid_baseline_long_horizon.jsonl \
  --trained-trace artifacts/traces/humanoid_trained_long_horizon.jsonl \
  --output-dir artifacts/isaac \
  --humanoid-showcase
