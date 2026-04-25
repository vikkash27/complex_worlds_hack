# Brev / Isaac Stretch Runbook

The core submission does not depend on Isaac Sim. Use this only after
`python scripts/run_demo.py` has generated metrics and replays.

## Fail-Fast Gate

Stop this stretch and return to the OpenReward demo if any step takes more than
45 minutes.

## Recommended Instance

Use a Brev instance with an RTX-compatible GPU for Isaac rendering, such as L4,
A10, A10G, L40, or another RTX/RT-core GPU exposed by Brev. Avoid A100/H100 for
Isaac rendering demos because they are compute-focused and can be problematic
for Kit streaming.

## Setup

1. Open NVIDIA Brev in the browser and create an Isaac Sim or Isaac Lab
   launchable if available.
2. Stop the instance whenever idle to preserve the $50 credit budget.
3. Clone or upload this repository folder to the instance.
4. Run the local benchmark first:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/run_demo.py
```

## Stretch Demo Goal

Do not train in Isaac. Record or stream a short visual companion that mirrors
the benchmark story:

- Baseline replay: random macro-policy stalls or accumulates errors.
- Dense-reward replay: trained macro-policy completes all subgoals and recovers
  from the disturbance.
- Show `artifacts/metrics/leaderboard.json` beside the replay.

## Submission Framing

If Isaac setup works, describe it as a visualization stretch. The verifiable
benchmark remains the OpenReward environment and its generated metrics.
