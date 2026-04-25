# Hackathon Demo Runbook

## 1. Generate local metrics and traces

```bash
cd ~/complex_worlds_hack
git pull origin main
bash scripts/bootstrap_dev_env.sh python3.11
.venv/bin/python scripts/run_demo.py
.venv/bin/python scripts/build_side_by_side.py
```

Expected artifacts:

- `artifacts/metrics/leaderboard.json`
- `artifacts/traces/baseline_fixed_script.jsonl`
- `artifacts/traces/dense_trained.jsonl`
- `artifacts/traces/humanoid_baseline_long_horizon.jsonl`
- `artifacts/traces/humanoid_trained_long_horizon.jsonl`
- `artifacts/replays/side_by_side_before_after.gif`

## 2. Start Isaac Sim streaming on Brev

Expose **TCP 49100** and **UDP 47998** in Brev, then:

```bash
cd ~/complex_worlds_hack
bash scripts/brev_setup_isaac.sh
docker pull nvcr.io/nvidia/isaac-sim:5.0.0
docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all --rm --network=host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v "$PWD":/workspace/complex_worlds_hack:rw \
  -v "$HOME/docker/isaac-sim/cache/kit":/isaac-sim/kit/cache:rw \
  -v "$HOME/docker/isaac-sim/cache/ov":/root/.cache/ov:rw \
  -v "$HOME/docker/isaac-sim/cache/pip":/root/.cache/pip:rw \
  -v "$HOME/docker/isaac-sim/cache/glcache":/root/.cache/nvidia/GLCache:rw \
  -v "$HOME/docker/isaac-sim/cache/computecache":/root/.nv/ComputeCache:rw \
  -v "$HOME/docker/isaac-sim/logs":/root/.nvidia-omniverse/logs:rw \
  -v "$HOME/docker/isaac-sim/data":/root/.local/share/ov/data:rw \
  -v "$HOME/docker/isaac-sim/documents":/root/Documents:rw \
  nvcr.io/nvidia/isaac-sim:5.0.0
```

Inside the container:

```bash
cd /isaac-sim
export PUBLIC_IP="<brev-public-ip>"
./runheadless.sh \
  --/app/livestream/publicEndpointAddress="${PUBLIC_IP}" \
  --/app/livestream/port=49100
```

Open the macOS Isaac Sim WebRTC Streaming Client and connect to `<brev-public-ip>:49100`.

## 3. Generate the high-fidelity USD replay

In a second Brev terminal:

```bash
docker exec -it isaac-sim bash -lc 'cd /workspace/complex_worlds_hack && /isaac-sim/python.sh scripts/isaac/replay_breakfast_tray.py \
  --baseline-trace artifacts/traces/baseline_fixed_script.jsonl \
  --trained-trace artifacts/traces/dense_trained.jsonl \
  --output-dir artifacts/isaac'
```

Then in the streamed Isaac UI:

1. File -> Open.
2. Open `/workspace/complex_worlds_hack/artifacts/isaac/breakfast_tray_side_by_side.usda`.
3. Press Play and scrub the timeline.

## 3b. Generate the humanoid long-horizon showcase USD

The humanoid showcase uses the same OpenReward/RoboCerebra trace contract, but
with a 100+ event long-horizon hospitality task. It uses a **local official
Unitree G1 asset**; do not rely on locked `/Isaac/...` browser folders for the
main demo.

First fetch and validate Unitree assets on the Brev host:

```bash
cd ~/complex_worlds_hack
.venv/bin/python scripts/isaac/fetch_unitree_g1_assets.py
```

Then generate the USD inside the Isaac container:

```bash
docker exec -it isaac-sim bash -lc 'bash /workspace/complex_worlds_hack/scripts/isaac/run_humanoid_replay_in_container.sh'
```

Or the same command written out in full (do **not** replace any line with `...`):

```bash
docker exec -it isaac-sim bash -lc 'cd /workspace/complex_worlds_hack && /isaac-sim/python.sh -u scripts/isaac/replay_breakfast_tray.py \
  --baseline-trace artifacts/traces/humanoid_baseline_long_horizon.jsonl \
  --trained-trace artifacts/traces/humanoid_trained_long_horizon.jsonl \
  --output-dir artifacts/isaac \
  --humanoid-showcase'
```

Run the smoke check before opening WebRTC:

```bash
docker exec -it isaac-sim bash -lc 'bash /workspace/complex_worlds_hack/scripts/isaac/smoke_humanoid_showcase_in_container.sh'
```

The smoke check verifies:

- `artifacts/isaac/humanoid_openreward_showcase.usda` exists.
- the USD references the validated local Unitree G1 asset.
- the replay summary contains at least 100 tool calls.

Optional articulation smoke test:

```bash
docker exec -it isaac-sim bash -lc 'cd /workspace/complex_worlds_hack && /isaac-sim/python.sh -u scripts/isaac/run_g1_articulation_policy.py'
```

This command validates whether the fetched G1 USD exposes discoverable hip,
knee, ankle, shoulder, elbow, or waist joints and writes
`artifacts/isaac/g1_articulation_policy.json`. If it fails, the asset is likely
mesh-only or URDF-only and should not be used for the physics-policy claim.

Then in the streamed Isaac UI:

1. File -> Open.
2. Open `/workspace/complex_worlds_hack/artifacts/isaac/humanoid_openreward_showcase.usda`.
3. Press Play and scrub the timeline.

Truthful claim: OpenReward/RoboCerebra optimizes the high-level planning and
tool policy. Isaac Sim shows a high-fidelity replay of those decisions using a
humanoid asset/fallback, not low-level humanoid RL training.

## 4. Run OpenReward comparison metrics

Local server:

```bash
.venv/bin/python -m robocerebra_rl.env
```

Second terminal:

```bash
.venv/bin/python scripts/benchmark_openreward.py \
  --base-url http://127.0.0.1:8080 \
  --environment robocerebra_reward_lab \
  --split test \
  --episodes 4 \
  --policy reactive_script \
  --compare-policy expert \
  --score-progress \
  --output artifacts/openreward/local_reactive_vs_expert_results.json
```

This reports per-shift `mean_tool_calls`, `p50_tool_calls`, `max_tool_calls`,
`mean_events_handled`, `mean_memory_recalls`, `mean_inventory_restocks`, and
`mean_tool_diversity`, plus comparison aggregates. The expected result on
`test`:

| Policy | success_rate | median tool_calls | events_handled | memory_recalls |
|--------|-------------:|-------------------:|----------------:|----------------:|
| `expert` | **1.0** | **~1660** | 9 | ≥1 |
| `reactive_script` | 0.0 | ~1130 | 0 | 0 |
| `random` | 0.0 | <10 | 0 | 0 |

(The reactive baseline still emits ~1130 calls per shift but fails because
it never acknowledges scheduled events or summarizes memory.)

Hosted OpenReward:

```bash
set -a && source .env && set +a
.venv/bin/python scripts/benchmark_openreward.py \
  --environment vikkash/complex_worlds_hack \
  --split test \
  --episodes 12 \
  --policy reactive_script \
  --compare-policy expert \
  --score-progress \
  --output artifacts/openreward/hosted_reactive_vs_expert_results.json
```

For **live** Gemini on `score_progress`, set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and
optionally `GEMINI_MODEL` (default `gemini-2.5-flash`). Set `ROBOCEREBRA_FORCE_SYMBOLIC_VLM=1`
to force the deterministic symbolic scorer. Without a key, scoring uses the symbolic
fallback (still exercised by `--score-progress` for metrics shape).

When you pass `--compare-policy`, the script also emits
`artifacts/openreward/submission_benchmark_summary.json` with both policies'
metrics plus the lift block—use it as a single attachment for write-ups or
leaderboard submissions.

## 5. MuJoCo G1 offline showcase

Use this if you need a reliable moving-joint artifact instead of a live Isaac
USD replay. It uses MuJoCo Menagerie Unitree G1 assets and the same
RoboCerebra/OpenReward traces:

```bash
cd ~/complex_worlds_hack
bash scripts/mujoco/run_g1_showcase.sh
```

Outputs:

- `artifacts/mujoco/g1_openreward_showcase.gif`
- `artifacts/mujoco/g1_openreward_showcase_summary.json`

The summary reports baseline/optimized execute calls, failures, recoveries, and
total tool calls. If MuJoCo is not installed, the renderer falls back to a
deterministic storyboard GIF that still shows the baseline wait/recovery chain
versus the optimized smooth chain. For real MuJoCo rendering:

```bash
.venv/bin/python -m pip install mujoco
.venv/bin/python scripts/mujoco/render_g1_showcase.py --backend mujoco
```

### 5.1 Interactive 3D demo (not a GIF)

The GIF pipeline uses kinematic keyframes. For a **live MuJoCo Simulate window** with
`mj_step`, position (PD) actuators toward the trace, **measured** weld relpose when grasp
latches, and a pinned floating base so the robot does not drift away, run from the same repo:

```bash
cd /path/to/complex_worlds
.venv/bin/python scripts/mujoco/view_dynamic_grasp.py \
  --trace artifacts/traces/humanoid_trained_long_horizon.jsonl
```

This opens the default macOS/Windows **MuJoCo viewer** (same stack as
`mujoco.viewer.launch_passive`). On **macOS**, MuJoCo requires `mjpython` for the viewer; the
script re-invokes `.venv/bin/mjpython` automatically, or you can run
`.venv/bin/mjpython scripts/mujoco/view_dynamic_grasp.py` yourself. No WebRTC: this runs
**on the machine** where the command executes. Use `--lane baseline` for the stall/recovery
policy. Headless / CI smoke (no window):

```bash
.venv/bin/python scripts/mujoco/view_dynamic_grasp.py --headless
```

For Brev, you typically need a virtual display (e.g. Xvfb) or run headless, then download a
**GIF** from `render_g1_showcase.py` for presentation video instead.
