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

Build a browser report from those artifacts:

```bash
.venv/bin/python scripts/build_visual_report.py
.venv/bin/python scripts/build_side_by_side.py
open artifacts/visual_report/index.html
```

Optional **live** Gemini vision for `score_progress`: set **`ROBOCEREBRA_USE_GEMINI_VISION=1`**
plus **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`**, and optionally **`GEMINI_MODEL`**
(default `gemini-2.5-flash`). Without that flag (or without a key), `score_progress`
uses a deterministic symbolic fallback and states that in the rationale so demos
stay reproducible offline.

## OpenReward Environment

The environment class is `robocerebra_rl.env.RoboCerebraRewardLabEnv`.

Run a local OpenReward-compatible server:

```bash
.venv/bin/python -m robocerebra_rl.env
```

This is an API server, not a browser UI. See
`docs/openreward_deploy.md` for the correct session-header flow and hosted
OpenReward deployment commands.

Core tools:

- `observe`: returns current task state plus a rendered frame.
- `choose_subgoal`: records the agent's intended semantic subgoal.
- `execute_skill`: advances the world by one macro-action and returns dense reward.
- `score_progress`: returns cached Gemini-style progress scoring.
- `submit_done`: terminates the episode with final success reward.

## Benchmark Task

The OpenReward splits ship **76 train, 16 validation, and 16 test** tasks (**108**
total: four task families × curated seeds). Each
split cycles four embodied families (breakfast tray, spill recovery, countertop
cleanup, and a 30-stage humanoid hospitality chain) over fixed seed grids so
hosted sessions, local training, and `scripts/benchmark_openreward.py` share the
same `scene`, `horizon_ticks`, and task-aware `max_macro_steps` budgets.

The flagship workflow is a 7-stage breakfast-tray task:

1. Locate items.
2. Clear workspace.
3. Pick mug.
4. Fill drink.
5. Place snack.
6. Recover from a tray disturbance.
7. Deliver tray.

Each macro-action advances the simulator by enough internal ticks to produce a
long-horizon episode (typically 1000–1500 ticks) while keeping live tool calls
manageable. The humanoid task stretches the same tool API to 30 sequential
subgoals for 100+ OpenReward tool-call demos when `observe`, `choose_subgoal`,
`execute_skill`, and `score_progress` are chained (see
`docs/hackathon_demo_runbook.md`).

## Metrics

`scripts/run_demo.py` evaluates:

- Random macro-action baseline.
- Fixed-script and reactive-script baselines on randomized held-out scenes.
- Sparse-trained and dense-reward tabular macro-policies.
- Expert oracle ceiling.

The leaderboard JSON reports success rate, mean progress, dense reward, mean
ticks, disturbance recovery rate, confidence intervals, tool-call counts, and
headline lift over the stronger reactive-script baseline. The old deterministic
100% result is kept only under `deterministic_smoke_test`.

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

For the cloud visualization path and Isaac Sim core replay, see
`docs/brev_visualization.md`. The Isaac script consumes the same JSONL traces
used by the OpenReward evaluation so the 3D replay is tied to actual tool calls.

## Limitations

This MVP is a macro-level physical-AI benchmark slice, not full continuous robot
control. The purpose is to demonstrate a verifiable long-horizon evaluation and
dense reward strategy that can later be connected to real RoboCerebra/LIBERO
rollouts.

## License

Released under the [MIT License](LICENSE).

## Hosted benchmark bundle

When comparing two scripted policies through OpenReward, pass
`--compare-policy` to `scripts/benchmark_openreward.py`. Besides the per-policy
JSON files, it also writes `artifacts/openreward/submission_benchmark_summary.json`
with both metric dicts and the lift summary for judge-facing write-ups.
