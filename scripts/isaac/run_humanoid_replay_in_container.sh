#!/usr/bin/env bash
# Run inside the Isaac Sim container (paths match hackathon_demo_runbook.md).
# Do not replace this script with "..." placeholders in your shell.
set -euo pipefail
cd /workspace/complex_worlds_hack
G1_USD="$(
  /isaac-sim/python.sh scripts/isaac/validate_unitree_g1_asset.py \
    --target-dir artifacts/isaac/vendor/unitree
)"
export ROBOCEREBRA_HUMANOID_USD="$G1_USD"
echo "[replay] validated local Unitree G1 asset: $ROBOCEREBRA_HUMANOID_USD"
exec /isaac-sim/python.sh -u scripts/isaac/replay_breakfast_tray.py \
  --baseline-trace artifacts/traces/humanoid_baseline_long_horizon.jsonl \
  --trained-trace artifacts/traces/humanoid_trained_long_horizon.jsonl \
  --output-dir artifacts/isaac \
  --humanoid-showcase
