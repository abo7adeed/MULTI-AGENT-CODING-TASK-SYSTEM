# Multi-Agent Coding Task System

Give it a software engineering request. It inspects the repository, decomposes
the work into a dependency graph, dispatches specialised coding agents into
isolated Git worktrees, merges their branches — resolving conflicts with a merge
agent rather than picking a winner — runs the project's real test suite, and
reports what actually happened.

The design rule throughout: **the LLM proposes, Python disposes.** Anything that
must be reproducible — graph validity, scheduling, retries, worktree isolation,
merging, testing — is ordinary Python with a test. The model is asked to
understand a request and to propose file contents; its output is validated
before it is allowed to change anything.

---

## Architecture

```
                     POST /orchestrations
                              │
                              ▼
                    ┌───────────────────┐
                    │   Orchestrator    │  owns the pipeline, nothing else
                    └───────────────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                        ▼
┌─────────────┐        ┌──────────────┐         ┌──────────────┐
│ Repository  │        │ TaskAnalyzer │         │  DAGEngine   │ pure graph
│  Analyzer   │───────►│  (heuristic  │────────►│  algorithms  │
│ (measured)  │        │   + LLM)     │         └──────────────┘
└─────────────┘        └──────────────┘                │
     │                        │                        ▼
     └────────────────────────┴──────────────►  ┌──────────────┐
                                                │  Scheduler   │ async policy:
                                                │ waves, retry │ retries, timeout
                                                └──────────────┘
                                                          │
                             ┌────────────────────────────┤
                             ▼                            ▼
                    ┌─────────────────┐         ┌──────────────────┐
                    │  AgentExecutor  │         │  AgentRegistry   │ 16 roles
                    │ worktree per    │────────►│  + selector      │
                    │ task, isolated  │         └──────────────────┘
                    └─────────────────┘
                             │  one branch + one commit per task
                             ▼
                    ┌─────────────────┐
                    │   Integrator    │ merge → resolve → test → review
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  IntegrationReport│ merged / conflicts / skipped /
                    └─────────────────┘   regressions / review findings
```

### The modules

| Package | Responsibility |
|---|---|
| `app/brain/` | Understand the request and the repository. `RepositoryAnalyzer` measures; the LLM only interprets the measurements. `TaskDecomposer` builds a validated DAG from blueprints. `ContextManager` assembles the smallest useful context per task. |
| `app/engine/` | `DAGEngine` (pure algorithms: waves, cycles, critical path), `Scheduler` (async policy: concurrency, retries, timeout, cancellation), `ExecutionManager` (run lifetime, pause/resume/cancel, persistence). |
| `app/agents/` | 16-role roster, capability-based selection, the fenced file patch protocol, `LLMCoderAgent` / `LLMReviewAgent` with bounded repair rounds, and `AgentExecutor`, which gives every task its own worktree. |
| `app/git/` | One async wrapper around the git CLI. Worktree lifecycle, three-way merge, conflict preview and resolution. |
| `app/integrator/` | Test detection and execution, change collection, regression detection, conflict resolution, final review. |
| `app/llm/` | Provider abstraction: OpenCode CLI, Ollama (local or Cloud), any OpenAI-compatible API, and Gemini. |
| `app/sandbox/` | Docker isolation with a locked-down policy; explicit `NoSandbox` when disabled. |
| `app/api/` | FastAPI routes, SSE event stream, container wiring. |
| `app/orchestrator/` | The seam between all of the above. |

---

## Running it

### Locally

```bash
pip install -r requirements.txt
python -m uvicorn app.api.main:app --reload      # API on :8000, docs at /docs
cd frontend && npm install && npm run dev        # UI on :5173
```

Point a project at a local repository, then start a run:

```bash
curl -X POST localhost:8000/projects \
  -H 'content-type: application/json' \
  -d '{"name": "my-app", "local_path": "/path/to/repo"}'

curl -X POST localhost:8000/orchestrations \
  -H 'content-type: application/json' \
  -d '{"project_id": "<id>", "user_request": "Add JWT auth to the API", "wait": true}'
```

`wait: true` blocks until the run finishes. Otherwise the run continues in the
background and you follow it live on `GET /orchestrations/{id}/stream` (SSE).

A repository can only have one live run. Agents commit into one shared `.git`
and the integrator merges, resets and checks out branches in it, so a second
concurrent run against the same path would corrupt both: `POST /orchestrations`
answers `409` and names the run that is holding the repository.

### With Docker

```bash
mkdir -p repos && cp -r /path/to/your/project repos/my-app
docker compose up --build            # UI on :3000, API on :8000
docker compose --profile ollama up   # add a local Ollama
```

The UI is served same-origin: nginx proxies `/api` to the backend, so there is
no CORS in the browser path.

---

## Configuration

Everything is environment-driven (`app/config.py`); nothing hard-codes a model.

### LLM provider

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `opencode` | `opencode`, `ollama`, `api`, `gemini` |
| `LLM_MODEL` | *(empty)* | Overrides the provider default when set |
| `OPENCODE_MODEL` | `nemotron-3.5-lightning-free` | Requires the CLI on `PATH` |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / see below | Local server *or* Ollama Cloud |
| `OLLAMA_API_KEY` | *(empty)* | Required by `https://ollama.com`, ignored locally |
| `OLLAMA_NUM_PREDICT` | `8192` | Output ceiling per completion |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | — / `gemini-2.5-flash` | Google's OpenAI-compatible endpoint |
| `API_BASE_URL` / `API_MODEL` / `API_KEY` | OpenAI / `gpt-4o-mini` / — | Any OpenAI-compatible endpoint |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | `300` / `2` | Retries use capped exponential backoff |

Set `LLM_PROVIDER` to a real model backend before running a task. The test
suite runs fully offline against isolated fixtures; production runs require
one of the providers above.

### Ollama: local server, or Ollama Cloud

One provider serves both. The base URL decides which, and the API key is simply
ignored by a local server.

**Local** — nothing to sign up for, nothing leaves the machine:

```bash
ollama serve                 # or start the desktop app
ollama pull qwen2.5-coder:7b

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b
```

**Ollama Cloud** — no local model, no GPU, needs a key from
<https://ollama.com/settings/keys>:

```bash
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=<your key>
OLLAMA_MODEL=gpt-oss:120b
```

Two things bite people here, and both are reported clearly rather than
silently retried:

* **Model names differ between the two.** The API takes the exact name that
  `GET /api/tags` returns; the Ollama app and CLI show the same cloud models
  with a `:cloud` suffix. Asking the API for `gpt-oss:120b-cloud` is a 404.
  List the API names with:
  `curl -H "Authorization: Bearer $OLLAMA_API_KEY" https://ollama.com/api/tags`
* **A plan covers a subset of the catalogue.** A model that exists but is
  outside your plan answers with HTTP 402; add usage credits or pick another
  model. `/system/models` lists every hosted model and says so, and the startup
  probe reports whether the configured model is reachable before any run starts.

When the model is missing or the key is wrong, the API logs it at startup and
reports it on `GET /health` (`provider_reachable`, `provider_model_available`,
`provider_detail`) and in the Settings page — no run is needed to find out.

### Scheduling, storage, integration

| Variable | Default |
|---|---|
| `MAX_PARALLEL_TASKS` | `4` |
| `MAX_TASK_RETRIES` | `3` |
| `RETRY_BACKOFF_SECONDS` / `RETRY_BACKOFF_MULTIPLIER` | `1.0` / `2.0` |
| `TASK_TIMEOUT_SECONDS` / `ORCHESTRATOR_TIMEOUT_SECONDS` | `1800` / `7200` |
| `STATE_DB_PATH` | `./state/orchestrations.db` |
| `WORKSPACE_ROOT` | `./workspaces` |
| `TEST_COMMAND` / `LINT_COMMAND` / `TYPECHECK_COMMAND` | *(empty = auto-detect)* |
| `TEST_TIMEOUT_SECONDS` | `900` |
| `MAX_CONFLICT_RESOLUTION_ROUNDS` | `2` |

### Sandbox

Agents execute code, so isolation is a correctness property rather than a
nicety. `SANDBOX_ENABLED=false` (the default) runs commands directly and is
meant for local experimentation; turn it on anywhere else.

| Variable | Default |
|---|---|
| `SANDBOX_ENABLED` | `false` |
| `SANDBOX_IMAGE` | `python:3.11-slim` |
| `SANDBOX_MEMORY` / `SANDBOX_CPUS` / `SANDBOX_PIDS_LIMIT` | `2g` / `2.0` / `256` |
| `SANDBOX_NETWORK` | `none` |
| `SANDBOX_TIMEOUT_SECONDS` | `600` |

Containers are created with no network, capped memory/CPU/PIDs, a read-only root
filesystem plus a tmpfs, all capabilities dropped, `no-new-privileges`, a
non-root user, and an environment built from an allowlist. The only host path
mounted is the task workspace, and a path outside the configured roots is
refused rather than clamped.

---

## The invariants

These are the properties the system is built to keep, and the test suite exists
to hold them in place.

1. **A valid DAG or no DAG.** Whatever the request or the model proposes,
   `decompose` returns a graph that passes validation, or repairs it first.
2. **Parallel agents never share a directory.** Every task gets its own git
   worktree on its own branch. This is the entire isolation story.
3. **A conflict never silently picks a winner.** Either a merge agent
   reconciles both intents, or the branch is left unmerged and reported.
4. **A failed integration rolls back.** A merge that breaks the suite is
   reverted to the pre-merge snapshot rather than left for the user to untangle.
5. **"All tasks succeeded" is not success.** A run that merged nothing, or that
   recorded unresolved conflicts or regressions, is reported as a failure even
   when every individual task said SUCCESS.
6. **Honest self-reporting.** A read-only agent cannot claim to have written
   files, and a writing agent that produced no file changes reports failure
   rather than an empty success.
7. **Secrets do not leak into agent code.** Sandboxed environments are built
   from an allowlist; test runs are scrubbed of ambient credentials.
8. **One live run per repository.** Worktrees are per task, but the repository
   they belong to is shared, so a second concurrent run against the same path is
   refused instead of being allowed to corrupt the first one's merges.
9. **A failure is reported as itself.** A rejected key, a model outside the
   plan, and an answer cut off at the output limit are three different problems
   with three different fixes, and none of them is reported as "the agent
   changed nothing".

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, active provider, agent count, sandbox mode |
| `GET` | `/system/info` | Providers, scheduling limits, roster |
| `GET` | `/system/models` | Which models the configured provider can reach |
| `POST` `GET` `DELETE` | `/projects[/{id}]` | Register a repository |
| `POST` | `/orchestrations` | Plan and launch a run |
| `GET` | `/orchestrations[/{id}]` | List runs, inspect one |
| `GET` | `/orchestrations/{id}/dag` | Nodes with live status, waves, critical path |
| `GET` | `/orchestrations/{id}/agents` | Per-task agent activity |
| `POST` | `/orchestrations/{id}/pause` `resume` `cancel` | Run control |
| `GET` | `/orchestrations/{id}/stream` | SSE: task, agent, conflict and test events |
| `GET` | `/orchestrations/{id}/logs?after=` | Durable event history |
| `GET` | `/orchestrations/{id}/changes` `diff` | What each agent contributed |
| `GET` | `/orchestrations/{id}/test-results` `report` | Verification and the final report |
| `POST` | `/agents/{task_id}/retry` | Re-queue a failed or blocked task |

---

## Tests

```bash
python -m pytest app/tests/ -q
```

The suite is offline and deterministic by construction: isolated fixtures
throughout, every repository built in a temp directory, and real `git`
so worktree and merge behaviour is genuinely exercised rather than faked.

| Module | Covers |
|---|---|
| `test_domain.py` | Models, status transitions, invariants |
| `test_dag.py` | Graph algorithms: waves, cycles, critical path, parallelism |
| `test_scheduler.py` | Concurrency, retries, timeout, cancellation, blocking |
| `test_agents.py` | Roster, selection, patch protocol, coder/review agents, executor |
| `test_llm.py` | Provider contract, retries, error classification, factory |
| `test_git.py` | Worktrees, commits, diffs, merges, conflicts, recovery |
| `test_brain.py` | Repository analysis, task analysis, context building, decomposition |
| `test_integrator.py` | Test detection/execution, regressions, conflict resolution |
| `test_orchestrator.py` | Planning, execution, binding, integration gating, verdicts |
| `test_api.py` | Every endpoint, validation, error codes, SSE |
| `test_sandbox.py` | Policy, command construction, mount refusal, env scrubbing |
| `test_e2e.py` | The whole pipeline through HTTP against a real repository |

Tests requiring a Docker daemon skip themselves when one is not running.
