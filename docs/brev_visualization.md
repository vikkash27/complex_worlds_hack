# Brev Visualization Runbook

Use Brev for a cloud-hosted visual demo of the generated metrics and replay
artifacts. This does not require Isaac Sim and is the safest visualization path
for the hackathon.

## 1. Verify Brev CLI

```bash
brev --version
brev login
brev ls
```

If the CLI asks for the first-time tour, answer `No` once so future commands run
non-interactively.

## 2. Create A Low-Cost Instance

For the static visual report, a CPU or low-end GPU instance is enough. Save RTX
credits for Isaac only if the core demo is already complete.

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

## 3. Optional Isaac Stretch

Only attempt Isaac after the static report is working. Pick an RTX-capable Brev
instance such as L4, A10, A10G, or L40. Do not make Isaac part of the core
submission; use it as a visual companion if setup is quick.
