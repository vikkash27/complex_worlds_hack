# OpenReward Deployment Runbook

This repo already runs locally as an OpenReward/ORS environment. Hosted
OpenReward deployment works by connecting the GitHub repo to an OpenReward
environment.

Task counts in the OpenReward UI come from the deployed image’s
`list_tasks` implementation. **Shift mode** exposes **76 train / 16 validation /
16 test** tasks (**108** total): each task is one full multi-job shift (not a
single breakfast-tray episode). After you change splits, seeds, or tools,
**push to the linked GitHub default branch** and wait for the deployment build
to finish (`orwd logs <env> --build`); until then the hosted API may still
serve an older image.

## Local API Smoke Test

Start the server:

```bash
.venv/bin/python -m robocerebra_rl.env
```

In another terminal, create a session and environment instance:

```bash
SESSION_ID=$(curl -s -N -X POST http://127.0.0.1:8080/create_session | \
  .venv/bin/python -c "import sys,re; d=sys.stdin.read(); m=re.search(r'data: ([0-9a-f\\-]+)', d); print(m.group(1) if m else '')")

curl -s -X POST http://127.0.0.1:8080/create \
  -H "content-type: application/json" \
  -H "X-Session-ID: $SESSION_ID" \
  -d '{"env_name":"robocerebra_reward_lab","task_spec":{"split":"test","seed":4001}}'
```

Fetch prompt and tools:

```bash
curl -s http://127.0.0.1:8080/robocerebra_reward_lab/prompt \
  -H "X-Session-ID: $SESSION_ID"

curl -s http://127.0.0.1:8080/robocerebra_reward_lab/tools \
  -H "X-Session-ID: $SESSION_ID"
```

Call the first shift tool (read the active ticket before `execute_skill`):

```bash
curl -s -X POST http://127.0.0.1:8080/robocerebra_reward_lab/call \
  -H "content-type: application/json" \
  -H "X-Session-ID: $SESSION_ID" \
  -d '{"tool_name":"read_ticket","input":{}}'
```

Then create a plan and execute (example continues in
`scripts/benchmark_openreward.py`, which replays full expert sequences).

The browser route `/robocerebra_reward_lab/` is expected to return `404` because
this is an API server, not a web UI.

## Hosted OpenReward Deployment

Make sure `.env` has:

```bash
OPENREWARD_API_KEY="..."
GEMINI_API_KEY="..."
```

Create and link the hosted environment:

```bash
set -a && source .env && set +a

orwd create complex_worlds_hack \
  --description "Long-horizon physical-AI benchmark with dense Gemini rewards"

orwd link vikkash/complex_worlds_hack vikkash27/complex_worlds_hack \
  --cpu-memory 1:4 \
  --max-scale 2 \
  --concurrency 100
```

OpenReward will build from GitHub and deploy the Docker image defined by this
repo's `Dockerfile`.

If `orwd link` prints “You need to install the OpenReward GitHub App”, complete
the browser authorization for `vikkash27/complex_worlds_hack`, then rerun the
same command.

Monitor deployment:

```bash
orwd deployments vikkash/complex_worlds_hack
orwd logs vikkash/complex_worlds_hack --build
orwd logs vikkash/complex_worlds_hack
```

View in browser:

```text
https://openreward.ai/vikkash/complex_worlds_hack
```

## Recording Rollouts

After deployment, use OpenReward's Runs/Rollouts tab to inspect uploaded
trajectories. For the hackathon demo, the local artifacts under `artifacts/`
remain the fastest evidence: leaderboard JSON, training plot, and replay GIFs.

Run hosted benchmark calls (shift mode — expect **~1.5k–1.7k tool calls** per
`test` episode for `expert` when `--score-progress` is on):

```bash
set -a && source .env && set +a

.venv/bin/python scripts/benchmark_openreward.py \
  --environment vikkash/complex_worlds_hack \
  --split test \
  --episodes 1 \
  --policy expert \
  --score-progress \
  --output artifacts/openreward/hosted_shift_expert_smoke.json
```

Add `--compare-policy reactive_script` to also write
`artifacts/openreward/submission_benchmark_summary.json`, which bundles both
policies' metrics and the lift summary for judges.

After a **git push** to the linked repo, OpenReward rebuilds the Docker image
from this repo’s `Dockerfile`. There is no separate `orwd redeploy` step for
GitHub-linked envs—the new build starts from the push (watch
`orwd deployments` / `orwd logs … --build`).

Before hosted deployment is linked, test the same script against the local
server by passing `--base-url http://127.0.0.1:8080` and `--environment
robocerebra_reward_lab`.

## If the Tasks table still shows only a handful of rows

The dashboard counts should match what the **hosted** API returns from
`list_tasks`. Your local checkout should print **76 / 16 / 16** (108 total):

```bash
.venv/bin/python -c "from robocerebra_rl.env import RoboCerebraRewardLabEnv as E; \
print('train', len(E.list_tasks('train')), 'val', len(E.list_tasks('validation')), \
'test', len(E.list_tasks('test')))"
# Expect: train 76, val 16, test 16 (108 shift tasks).
```

Compare with **hosted** (same `OPENREWARD_API_KEY` as `orwd`; replace the env id):

```bash
set -a && source .env && set +a
.venv/bin/python -c "
from openreward import OpenReward
name = 'vikkash/complex_worlds_hack'  # your namespace/env
env = OpenReward().environments.get(name=name)
for split in ('train', 'validation', 'test'):
    n = len(list(env.list_tasks(split=split)))
    print(split, n)
"
```

- If **local** is 76/16/16 but **hosted** is still wrong, the linked GitHub
  default branch does not contain your latest commit, or the deployment build
  failed—check `orwd logs <env> --build` and GitHub.
- Confirm the browser page is the **same** environment slug you linked (not
  another project such as a different hackathon env).

### Why “Task Images” is empty on a deployment

The **Task Images** tab is for **Harbor** environments: OpenReward can build
extra **per-task (sandbox) images** in that mode. The CLI describes
`orwd task-builds` as listing those **harbor task image** builds.

If you created the environment **without** Harbor (`orwd create` without
`--harbor`, which is the usual case for a single-Dockerfile server like this
repo), **no task-image builds are expected**. Your tasks still come from
`list_tasks` inside the main container at runtime. An empty Task Images tab is
normal and **not** a sign that `list_tasks` failed.

### Deploy “Completed” but the Overview Tasks table still looks wrong

Use the **hosted** `num_tasks` / `list_tasks` check in the section above. If
those return 76 / 16 / 16 but the web table does not, treat it as a UI or cache
issue (hard refresh, private window, **Redeploy**).

Also confirm GitHub’s **default branch** at the deployment commit contains
`SHIFT_SPLIT_SEEDS` and `list_openreward_shift_tasks_for_split` in
`src/robocerebra_rl/world.py`, plus `src/robocerebra_rl/shift.py`. If that
commit exists only on a side branch, the hosted image will not match your
local checkout.
