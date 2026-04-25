# OpenReward Deployment Runbook

This repo already runs locally as an OpenReward/ORS environment. Hosted
OpenReward deployment works by connecting the GitHub repo to an OpenReward
environment.

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
  -d '{"env_name":"robocerebra_reward_lab","task_spec":{"seed":1,"horizon_ticks":1000}}'
```

Fetch prompt and tools:

```bash
curl -s http://127.0.0.1:8080/robocerebra_reward_lab/prompt \
  -H "X-Session-ID: $SESSION_ID"

curl -s http://127.0.0.1:8080/robocerebra_reward_lab/tools \
  -H "X-Session-ID: $SESSION_ID"
```

Call the first macro-skill:

```bash
curl -s -X POST http://127.0.0.1:8080/robocerebra_reward_lab/call \
  -H "content-type: application/json" \
  -H "X-Session-ID: $SESSION_ID" \
  -d '{"tool_name":"execute_skill","input":{"action":"locate_items"}}'
```

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
