# Frontend — Multi-Agent Coding Task System

React + TypeScript + Vite UI for the orchestration API.

## Pages

- `Dashboard` — runs, status, live progress
- `NewTask` — register a project and launch an orchestration
- `Execution` — DAG waves, critical path, per-task status
- `Agents` — per-task agent activity
- `Inspect` — changes, diffs, test results, final report

## Running it

```bash
cd frontend
npm install
npm run dev      # UI on :5173, proxies /api to localhost:8000
```

Production (served by nginx, same-origin `/api` → backend):

```bash
npm run build
```

Docker Compose serves the built UI on `:3000` with nginx proxying `/api`
to the backend, so there is no CORS in the browser path.

## Config

- `vite.config.ts` — dev server + `/api` proxy
- `src/api/client.ts` — typed API client (projects, orchestrations, DAG, agents, logs, changes, test results, SSE stream)
- `tailwind.config.js` / `src/index.css` — Tailwind styling
