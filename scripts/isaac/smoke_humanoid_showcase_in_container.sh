#!/usr/bin/env bash
# Run inside the Isaac Sim container after Unitree assets and traces are present.
set -euo pipefail
cd /workspace/complex_worlds_hack

G1_USD="$(
  /isaac-sim/python.sh scripts/isaac/validate_unitree_g1_asset.py \
    --target-dir artifacts/isaac/vendor/unitree
)"
export ROBOCEREBRA_HUMANOID_USD="$G1_USD"
echo "[smoke] validated local Unitree G1 asset: $ROBOCEREBRA_HUMANOID_USD"

/isaac-sim/python.sh -u scripts/isaac/replay_breakfast_tray.py \
  --baseline-trace artifacts/traces/humanoid_baseline_long_horizon.jsonl \
  --trained-trace artifacts/traces/humanoid_trained_long_horizon.jsonl \
  --output-dir artifacts/isaac \
  --humanoid-showcase

/isaac-sim/python.sh - <<'PY'
from pathlib import Path
import json
import os

root = Path("/workspace/complex_worlds_hack")
summary_path = root / "artifacts/isaac/isaac_replay_summary.json"
usd_path = root / "artifacts/isaac/humanoid_openreward_showcase.usda"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
tool_calls = int(summary["tool_trace_summary"]["total_tool_calls"])
asset = os.environ.get("ROBOCEREBRA_HUMANOID_USD", "")
usd_text = usd_path.read_text(encoding="utf-8", errors="ignore")

assert usd_path.is_file(), usd_path
assert tool_calls >= 100, tool_calls
assert asset and asset in usd_text, f"Expected local G1 reference {asset!r} in {usd_path}"
assert "robocerebra_motion" in usd_text, "Expected G1 link-level motion overlays in exported USD"
assert "xformOp:rotateXYZ:robocerebra" in usd_text, "Expected animated G1 link rotations in exported USD"
print(f"[smoke] humanoid USD: {usd_path}")
print(f"[smoke] tool calls: {tool_calls}")
print(f"[smoke] local G1 reference: {asset}")
print("[smoke] G1 link motion overlays: present")
PY
