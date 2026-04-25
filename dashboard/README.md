# RoboCerebra Reward Lab — Dashboard

Vite + React + TS + Tailwind v3 + shadcn structure. Aceternity Aurora hero +
Aceternity Bento Grid + ECharts viz.

## Run

```bash
cd dashboard
npm install
npm run dev
```

Open http://localhost:5173.

## Stack

- `vite` + `react` 18 + TS strict
- `tailwindcss` v3, dark-mode class on `<html>` (set in `index.html`)
- `framer-motion` (hero entrance, live counter)
- `echarts` + `echarts-for-react` (Gantt, heatmap, graph, sankey)
- `lucide-react` (icons)
- `@xyflow/react` (reserved for future React Flow upgrade of DAG)

## Layout

1. **Aurora hero** — `components/ui/aurora-background.tsx` with stat tiles.
2. **Bento grid** — `components/ui/bento-grid.tsx`.
3. **Charts** — `components/charts/{timeline,dag,sankey}.tsx`.
4. **Live pulse** — `components/live-pulse.tsx`. Currently fakes counter;
   swap for real `EventSource('/stream')` when `serve_dashboard.py` is wired.

## Live wiring (next step)

Backend: add `scripts/serve_dashboard.py` (FastAPI + SSE). Browser:
swap the `setInterval` in `live-pulse.tsx` for `new EventSource('/stream')`
and pipe events into `trace-data.ts` buffers. Vite proxies `/stream` to
the backend port via `vite.config.ts`.
