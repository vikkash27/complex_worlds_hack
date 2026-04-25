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
with a 100+ event long-horizon hospitality task. It references built-in Isaac
humanoid assets first (`Unitree/H1` and `IsaacSim/Humanoid`) and keeps an
animated fallback skeleton if those assets are unavailable.

```bash
docker exec -it isaac-sim bash -lc 'cd /workspace/complex_worlds_hack && /isaac-sim/python.sh scripts/isaac/replay_breakfast_tray.py \
  --baseline-trace artifacts/traces/humanoid_baseline_long_horizon.jsonl \
  --trained-trace artifacts/traces/humanoid_trained_long_horizon.jsonl \
  --output-dir artifacts/isaac \
  --humanoid-showcase'
```

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
  --episodes 12 \
  --policy reactive_script \
  --compare-policy expert \
  --score-progress \
  --output artifacts/openreward/local_reactive_vs_expert_results.json
```

This reports per-policy `mean_tool_calls`, `total_tool_calls`, and comparison
`aggregate_tool_calls`. For the long-horizon single-episode claim, use the
humanoid trace metrics in `artifacts/metrics/replay_rollouts.json`.

To back the 100+ call claim through OpenReward rather than only local traces,
run a filtered single-episode humanoid benchmark:

```bash
.venv/bin/python scripts/benchmark_openreward.py \
  --base-url http://127.0.0.1:8080 \
  --environment robocerebra_reward_lab \
  --split test \
  --episodes 1 \
  --task-name humanoid_hospitality \
  --policy expert \
  --score-progress \
  --output artifacts/openreward/local_humanoid_100_call_episode.json
```

With `--score-progress`, the benchmark records `observe`, `choose_subgoal`,
`execute_skill`, and `score_progress` around each humanoid micro-step, producing
100+ OpenReward tool calls in one long-horizon episode.

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

Set `ROBOCEREBRA_USE_GEMINI_VISION=1` and `GEMINI_MODEL=<available-model-id>` only when you want paid/nondeterministic Gemini vision scoring. Without those env vars, the same benchmark uses deterministic fallback scoring.
