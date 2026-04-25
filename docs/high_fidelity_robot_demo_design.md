# High-Fidelity RoboCerebra Demo Design

## Demo Claim

This demo showcases RoboCerebra as a long-horizon mobile service robot benchmark with OpenReward-compatible macro-skills, dense reward learning, and optional Gemini vision scoring. It does not claim low-level Isaac RL training. Isaac Sim is used for a high-fidelity visual replay of the same tool traces and task outcomes measured by the benchmark.

## Visual Direction

The Isaac stage should look like a service robot benchmark rather than a toy block scene:

- A mobile service robot lane for the baseline policy and one for the improved policy.
- Three visible task stations: breakfast tray, spill recovery, and countertop cleanup.
- Colored materials, labels, lights, camera framing, progress markers, and action-driven animation.
- Hybrid assets: use built-in Isaac/Omniverse robot and prop assets when available, with colored proxy geometry as deterministic fallback.

## Evaluation Direction

The metrics story has two layers:

- Deterministic macro-policy metrics: success rate, mean reward, tool calls, progress, and disturbance recovery.
- Optional Gemini vision metrics: confidence, progress agreement, and rationales using rendered observation images when `GEMINI_API_KEY` and `GEMINI_MODEL` are configured.

All OpenReward numbers must state that they measure macro-policy tool use through OpenReward sessions, not direct physics training in Isaac.

## Artifacts

Expected outputs:

- `artifacts/isaac/breakfast_tray_side_by_side.usda`
- `artifacts/isaac/isaac_replay_summary.json`
- `artifacts/openreward/*_results.json`
- `artifacts/metrics/leaderboard.json`
- `artifacts/replays/side_by_side_before_after.gif`

## Acceptance Checks

- The local test suite passes without network or Gemini credentials.
- The Isaac USD opens in the streamed WebRTC client and displays colored, labeled, multi-task lanes.
- OpenReward benchmark output includes a baseline/improved comparison with explicit claim boundaries.
- Gemini vision scoring remains opt-in and falls back deterministically when credentials or images are unavailable.
