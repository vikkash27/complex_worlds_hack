# RoboCerebra Reward Lab

[OpenReward](https://openreward.ai) benchmark for **long-horizon physical-AI
planning** with dense, VLM-style subgoal rewards. The repo implements two
sides of the same story: a **single-task breakfast-tray** slice with symbolic
dense reward and tabular baselines (`scripts/run_demo.py`), and **Shift
Mode** — full **hospitality shifts** on the host where each episode is a
deterministic chain of 12–30 manipulation jobs (inventory, memory, clock,
tickets, scheduled non-stationary events) wired through 18 OpenReward tools.
On `test` shifts, a capable agent burns **~1660 tool calls** to win; the
strongest hand-written baseline burns ~1130 and still fails.

**Paper (methods, splits, leaderboard schema):** [RoboCerebra Reward Lab on OSF](https://osf.io/h8rnv/overview)

## Winning claim

Short-horizon benchmarks reward one good guess. This environment targets the
opposite: success requires *long-horizon planning, credit assignment across
macro tool calls, and (in Shift Mode) memory recall, inventory tracking, event
handling, and eventual self-summarization* — for shifts, that means composing
**18** OpenReward tools over hundreds to thousands of calls before the
terminal success bit. The OSF paper above spells out reward decomposition,
splits (`train` / `validation` / `test`), and the evaluation schema for
[judge-style write-ups and leaderboard entries](https://openreward.ai).

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/run_demo.py
```

The demo writes:

- `artifacts/metrics/leaderboard.json` — breakfast-tray **randomized held-out** results, smoke baselines, and headline lifts (see **Metrics**).
- `artifacts/metrics/randomized_policy_report.json` — per-policy breakdown for the 80-episode held-out run.
- `artifacts/plots/training_curve.png`
- `artifacts/replays/baseline_random.gif`
- `artifacts/replays/dense_trained.gif`

Build a browser report from those artifacts:

```bash
.venv/bin/python scripts/build_visual_report.py
.venv/bin/python scripts/build_side_by_side.py
open artifacts/visual_report/index.html
```

**Live** Gemini for `score_progress`: set **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`**, and
optionally **`GEMINI_MODEL`** (default `gemini-2.5-flash`). The OpenReward server and
`run_demo` sample VLM call use the API whenever a key is present. Set
**`ROBOCEREBRA_FORCE_SYMBOLIC_VLM=1`** to force the deterministic symbolic scorer (e.g. CI, offline
repro) even if a key exists.

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

## Benchmark Task — Long-Horizon Shift Mode

Each OpenReward task is a full **hospitality shift**: a chain of 12–30 jobs
(breakfast tray, spill recovery, countertop cleanup) running on persistent
shift state — **inventory, memory, clock, ticket queue, deterministic
non-stationary events** — until the agent can submit a verified shift
summary. This is what gives us *hundreds-to-thousands of tool calls per
episode*.

| Split | Shifts | Jobs / shift | Events / shift | Median expert tool calls |
|-------|------:|-------------:|---------------:|--------------------------:|
| `train` | 76 | 12 | 3 | **~634** |
| `validation` | 16 | 22 | 6 | **~1189** |
| `test` | 16 | 30 | 9 | **~1660** |

(`test` shifts can grow past 1700 calls when scheduled `spill` events insert
extra recovery jobs.)

**Tool surface (18 tools)**:

- Per-job loop: `observe`, `choose_subgoal`, `execute_skill`, `score_progress`.
- Plan & memory: `read_ticket`, `plan_create`, `plan_revise`,
  `memory_write`, `memory_read`, `memory_search`, `memory_summarize`.
- Resource & time: `inventory_check`, `inventory_consume`, `inventory_restock`,
  `clock_get`.
- Disturbance lifecycle: `acknowledge_event`, `log_job`, `submit_done`.

**Why this is hard but tractable**:

- *Long horizon*: 1500–1700 expert tool calls on `test`; reactive baseline
  pushes ~1130 calls and still fails because it never acknowledges events
  or summarizes memory.
- *Capability tangent*: success requires real planning, recall via
  `memory_search`, inventory restocking on stockouts, and `plan_revise` on
  VIP / time-pressure events.
- *Solvable*: the deterministic per-seed expert oracle solves every shift
  with **100% success** within budget, proving tractability.

## Metrics

There are two evaluation surfaces; use the one that matches what you are reporting.

**1. Single-task breakfast tray (80-episode randomized held-out).** Produced by
`scripts/run_demo.py`. Read `artifacts/metrics/leaderboard.json` under
`randomized_heldout` and `headline`, or `artifacts/metrics/randomized_policy_report.json`.
Report (per policy): `success_rate` (with `success_rate_ci95` if you show intervals),
`mean_progress`, `disturbance_recovery_rate` (with `disturbance_recovery_ci95`),
`mean_reward` (symbolic dense reward), `mean_ticks`, `mean_tool_calls`, and
`episodes` (80). Regime label: `randomized_heldout` (80 episodes per policy).

**2. Shift Mode (hosted OpenReward environment).** Full hospitality shifts with
the 18-tool surface. Use `scripts/benchmark_openreward.py` against
`vikkash/complex_worlds_hack` (or your fork). Output JSON includes `split`,
`episodes`, `success_rate`, `success_rate_ci95`, `mean_reward`, `mean_tool_calls`,
and related shift aggregates. Headline on `test`: **expert 100% success @ ~1660
median calls** vs **reactive 0% success @ ~1130 calls** — same long-horizon work,
only the capability-rich policy wins.

`artifacts/metrics/leaderboard.json` produced by `run_demo` is **breakfast-only**
(the structure above). Shift-mode numbers are **not** in that file; they come
from `scripts/benchmark_openreward.py` (see **Hosted benchmark bundle**), with
per-policy `expert`, `reactive_script`, and `random` including:

- `success_rate` — gated on completing every job, acknowledging every
  scheduled event, and a passing `memory_summarize`.
- `mean_tool_calls`, `p50_tool_calls`, `min_tool_calls` / `max_tool_calls`
  (hosted JSON from `benchmark_openreward.py`).
- `mean_events_handled`, `mean_memory_recalls`, `mean_inventory_restocks`,
  `mean_score_progress_calls`, `mean_tool_diversity`.

### OpenReward leaderboard (site)

When you **add a leaderboard entry** on [openreward.ai](https://openreward.ai) for this environment:

- **Split** — Use `test` for hosted `benchmark_openreward.py` results. For the
  80-episode breakfast held-out numbers from `run_demo`, the platform may only
  offer `overall`; that is fine if you name the **regime** in the metrics (below).
- **Metrics** — Add one name/value row per scalar. Prefer these names:
  - Always: `success_rate`, `mean_reward`, `episodes`, `mean_tool_calls`.
  - Breakfast held-out: also `mean_progress`, `mean_ticks`, `disturbance_recovery_rate`, and
    optional `success_rate_ci95_low` / `success_rate_ci95_high` (and the same
    for recovery) if the form accepts them.
  - Optional context row: `regime` = `randomized_heldout` for breakfast numbers
    from `run_demo`; for hosted shift runs, put `split=test` (or your split) in
    the form’s Split field and `episodes` in metrics.
- **Provenance** — Link the OSF preprint ([`osf.io/h8rnv`](https://osf.io/h8rnv/overview)) in
  the form’s **Paper** field where available, and note commit SHA, whether
  `score_progress` used live Gemini or symbolic fallback, and the exact command.

Do not use a generic `Accuracy` field unless the UI requires a single primary
column; use `success_rate` for the main success metric.

## Two-minute pitch

“Post-SWE frontier agents need environments that test long-horizon physical
reasoning, not just code tasks. RoboCerebra Reward Lab exposes a manipulation
workflow as an OpenReward environment with semantic tools, dense VLM-style
progress rewards, and verifiable rollouts. Dense rewards improve sample
efficiency over sparse success-only feedback, producing metric curves and replay
videos that show recovery from non-stationary disturbances.” Full setup,
headline tables, and reporting conventions are in the [OSF paper](https://osf.io/h8rnv/overview).

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
with both metric dicts and the lift summary for judge-facing write-ups. Copy
values from the `metrics` object in that file (or the per-policy output JSON)
into the site leaderboard form as described under **OpenReward leaderboard (site)**
above.
