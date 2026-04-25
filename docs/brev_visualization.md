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

Install and regenerate artifacts. **Ubuntu’s default `python3` is often 3.10**; you must have **`python3.11`** on the PATH, then run the bootstrap (it will remove a stale 3.10 `.venv` automatically):

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv
bash scripts/bootstrap_dev_env.sh python3.11
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

## 5. Headless livestream (WebRTC)

After `bash scripts/brev_setup_isaac.sh`, expose **TCP 49100** and **UDP 47998** from the Brev instance to your machine, then start headless streaming with the printed `runheadless.sh` command (use the VM’s public IP for `publicEndpointAddress`).

**Viewing:** use NVIDIA’s **Isaac Sim WebRTC Streaming Client** (download from the [Isaac Sim 5.0 livestream clients](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/manual_livestream_clients.html) documentation). In the client, connect to **`<PUBLIC_IP>:49100`** (replace with your instance address). This is a small desktop app; the default path is not a generic browser URL. Some newer docs describe an optional **browser** viewer via a separate web stack (for example on port 8210 with Docker); that is not what `brev_setup_isaac.sh` prints by default.
