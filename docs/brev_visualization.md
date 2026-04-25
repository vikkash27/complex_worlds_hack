# Brev Visualization Runbook

Use Brev for two visual paths:

1. A fast browser report from generated metrics and replay artifacts.
2. The Isaac Sim core visual replay using the same JSONL traces.

## 1. Verify Brev CLI

```bash
brev --version
brev login
brev ls
```

If the CLI asks for the first-time tour, answer `No` once so future commands run
non-interactively.

## 2. Recommended Isaac-Capable Instance

Dry-run search found an L40S option suitable for Isaac Sim:

```text
type: l40s-48gb.1x
provider: crusoe
vram: 48 GB
flex ports: true
price: about $1.74/hour
```

Create it only when you are ready to spend credits:

```bash
brev create complex-worlds-isaac --type l40s-48gb.1x --flex-ports --stoppable
brev ls
brev shell complex-worlds-isaac
```

Expose Isaac livestream ports `49100` and `47998` only to your IP.

## 3. Fast Static Report

For the static visual report, a CPU or low-end GPU instance is enough.

Clone the repo on the instance:

```bash
cd /home/ubuntu/workspace
git clone https://github.com/vikkash27/complex_worlds_hack.git
cd complex_worlds_hack
```

Install and regenerate artifacts:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/run_demo.py
.venv/bin/python scripts/build_visual_report.py
.venv/bin/python scripts/build_side_by_side.py
```

Serve the report:

```bash
cd artifacts/visual_report
python3 -m http.server 7860
```

Then open/forward port `7860` from Brev. The page shows:

- headline metric lifts,
- random baseline replay,
- dense-reward trained replay,
- training curve,
- raw leaderboard JSON.

## 4. Isaac Sim Core Replay

After the static report is working, run:

```bash
bash scripts/brev_setup_isaac.sh
```

Inside the Isaac Sim container, the core replay command is:

```bash
cd /workspace/complex_worlds_hack
/isaac-sim/python.sh scripts/isaac/replay_breakfast_tray.py \
  --baseline-trace artifacts/traces/baseline_fixed_script.jsonl \
  --trained-trace artifacts/traces/dense_trained.jsonl \
  --output-dir artifacts/isaac
```

This produces an Isaac USD replay scene:

```text
artifacts/isaac/breakfast_tray_side_by_side.usda
artifacts/isaac/isaac_replay_summary.json
```

Open the USD in Isaac Sim or stream the Brev viewport for the before/after demo.
