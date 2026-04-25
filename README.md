# RoboCerebra Reward Lab

RoboCerebra Reward Lab is a same-day hackathon benchmark slice for long-horizon
physical-AI planning. It wraps a RoboCerebra/LIBERO-style household manipulation
workflow as an OpenReward environment and demonstrates that dense subgoal rewards
make a macro-policy learn faster than sparse success-only feedback.

## Winning Claim

Frontier agents struggle when physical workflows require memory, recovery, and
long-horizon credit assignment. This project turns a 1000-tick breakfast-tray
manipulation workflow into an OpenReward tool-call environment with dense
Gemini-style reward scoring at semantic subgoal boundaries.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/run_demo.py
```

The demo writes:

- `artifacts/metrics/leaderboard.json`
- `artifacts/plots/training_curve.png`
- `artifacts/replays/baseline_random.gif`
- `artifacts/replays/dense_trained.gif`

Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` to use Gemini for reward scoring. Without
an API key, the project uses a deterministic symbolic fallback and records that in
the reward rationale so the demo remains runnable.

## OpenReward Environment

The environment class is `robocerebra_rl.env.RoboCerebraRewardLabEnv`.

Run a local OpenReward-compatible server:

```bash
.venv/bin/python -m robocerebra_rl.env
```

Core tools:

- `observe`: returns current task state plus a rendered frame.
- `choose_subgoal`: records the agent's intended semantic subgoal.
- `execute_skill`: advances the world by one macro-action and returns dense reward.
- `score_progress`: returns cached Gemini-style progress scoring.
- `submit_done`: terminates the episode with final success reward.

## Benchmark Task

The workflow is a 7-stage breakfast-tray task:

1. Locate items.
2. Clear workspace.
3. Pick mug.
4. Fill drink.
5. Place snack.
6. Recover from a tray disturbance.
7. Deliver tray.

Each macro-action advances the simulator by enough internal ticks to produce a
500-2000 tick long-horizon episode while keeping live tool calls manageable.

## Metrics

`scripts/run_demo.py` evaluates:

- Random macro-action baseline.
- Dense-reward tabular macro-policy.
- Expert oracle ceiling.

The leaderboard JSON reports success rate, mean progress, dense reward, mean
ticks, disturbance recovery rate, and headline lift over the random baseline.

## Two-Minute Pitch

“Post-SWE frontier agents need environments that test long-horizon physical
reasoning, not just code tasks. RoboCerebra Reward Lab exposes a manipulation
workflow as an OpenReward environment with semantic tools, dense VLM-style
progress rewards, and verifiable rollouts. Dense rewards improve sample
efficiency over sparse success-only feedback, producing metric curves and replay
videos that show recovery from non-stationary disturbances.”

## Stretch Path

Use Brev/Isaac only after the core metrics are generated. The safe stretch is a
visual replay, not training. If Isaac setup takes more than 45 minutes, preserve
the benchmark story and submit the OpenReward environment plus metrics.

## Limitations

This MVP is a macro-level physical-AI benchmark slice, not full continuous robot
control. The purpose is to demonstrate a verifiable long-horizon evaluation and
dense reward strategy that can later be connected to real RoboCerebra/LIBERO
rollouts.
