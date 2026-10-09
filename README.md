# Travel Assistant Multi-Agent System

The project uses a supervisor-led LangGraph workflow with destination, travel planner, and itinerary specialists. Only the destination specialist can access the separate `travel-tools-mcp` service, through the application's fixed `search_destination` capability. The initial MCP tool uses deterministic local data. All model calls go through LangChain's `ChatOpenAI` client to the LiteLLM gateway, which provides the configured `travel_general` and `travel_local` aliases.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Docker with Docker Compose
- A Gemini API key for real chat requests

## Local development

1. Copy `.env.example` to `.env`.
2. Set `GEMINI_API_KEY` and generate one strong random key for both `LITELLM_MASTER_KEY` and `LITELLM_API_KEY`. The shared admin-level credential is for local/internal development only. Keep `.env` out of Git.
3. Install locked dependencies with `uv sync --locked`.
4. Start the database, gateway, Redis, API, MCP service, and worker with `docker compose up --build`.

LiteLLM is DB-less. Its host port is bound to `127.0.0.1:4000`; the API reaches it through the private Compose network. The Gemini key and LiteLLM master key are passed as runtime environment variables and are not baked into the application image or LiteLLM config file.

For host-side API development, set `DATABASE_URL` to a reachable PostgreSQL instance and `LITELLM_BASE_URL` to `http://localhost:4000/v1`, then run:

```sh
uv run uvicorn app.main:app --reload
```

On native Windows, Psycopg's async driver requires a selector event loop. Start Uvicorn with that loop policy:

```powershell
uv run python -c "import asyncio, uvicorn; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()); uvicorn.run('app.main:app')"
```

## API

- `GET /api/v1/health/live` reports process liveness independently of dependencies.
- `GET /api/v1/health/ready` checks PostgreSQL and LiteLLM's readiness endpoint without generating model output.
- `POST /api/v1/chat` invokes `DEFAULT_MODEL` (default: `travel_general`) unless a model alias is supplied in the request. The selected model's graph state is persisted by thread ID.

Example request:

```json
{
  "message": "Xin chào",
  "model": "travel_local",
  "thread_id": "9de6a955-31ee-4a0c-b877-34123cd130d5"
}
```

`model` and `thread_id` are optional. If `model` is omitted, the API uses `DEFAULT_MODEL`; the alias is passed through LiteLLM without automatic fallback. The API returns a generated UUID when `thread_id` is omitted. The supervisor dynamically routes to allowed specialists and validates each proposed route before graph execution. PostgreSQL checkpoints conversation messages and structured travel state by thread ID. Tests use fake chat models and do not call Gemini or Ollama.

Planning requests pause after the planner creates a structured travel plan and before an itinerary is generated. The response is additive and includes `status: "interrupted"` plus `pending_approval` with the `interrupt_id`, current `travel_plan`, `plan_revision_count`, and `max_plan_revisions`.

Resume the same thread with `POST /api/v1/chat/{thread_id}/resume` and the pending `interrupt_id`. Choose `approve`, `modify`, or `reject`. A `modify` decision requires a non-empty `changes` object containing only TravelPlan fields; approve and reject do not accept changes. The API permits up to three modifications. Each modification returns a new pending approval and interrupt ID. An approved plan continues to itinerary generation; a rejected plan ends the workflow. While approval is pending, `/api/v1/chat` returns 409 for that thread. Completed chat responses retain the original `{ "thread_id", "reply" }` shape.

Example modification:

```json
{
  "interrupt_id": "<pending interrupt id>",
  "decision": "modify",
  "changes": { "budget": "moderate", "preferences": ["food"] }
}
```

### Asynchronous jobs

The synchronous endpoints above remain available with their existing response contract. For callers that should not hold an HTTP request open during graph execution, Phase 6 adds:

- `POST /api/v1/chat/jobs` — accepts the same chat request and returns `202` with `job_id`, `thread_id`, and `status: "queued"`.
- `POST /api/v1/chat/{thread_id}/resume/jobs` — accepts the existing resume decision and returns a queued job. The URL thread ID is authoritative, and resume does not add a user message.
- `GET /api/v1/jobs/{job_id}` — returns operational job status and reconciles workflow details from the PostgreSQL checkpoint.

Redis Streams and a separate worker deliver jobs at least once. Redis job statuses (`queued`, `running`, `succeeded`, `failed`) describe queue execution only; PostgreSQL/LangGraph checkpoints remain authoritative for workflow progress, results, and pending approval. The Redis service uses AOF with `appendfsync everysec` and a named volume; this accepts a non-zero disaster recovery point objective. A worker recovery can repeat the deterministic, read-only destination lookup if it crashes during that MCP node before its graph checkpoint is written; exactly-once MCP execution is not promised.

The worker heartbeats its pending stream entry every 30 seconds and claims entries idle for 300 seconds. LangGraph 1.2.13 recovery was exercised with deterministic in-memory graphs: active interrupts re-enter the interrupted node on resume, checkpointed preceding nodes are not replayed after a later node failure, and a failed node is re-entered. The heartbeat keeps an active delivery from reaching the claim idle threshold during long graph calls; the PostgreSQL per-thread advisory lock prevents concurrent graph execution for one thread.

## Checks

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
docker compose config
```

## Phase 8 – verification status

- **Phase 8.3** – edge‑case worker tests were previously reported **PASS**.
- **Phase 8.4** – four targeted tests were reported **PASS**: the two HITL tests, `tests/test_phase4_e2e.py`, and the worker redelivery test.
- **PostgreSQL checkpoint persistence across two independent runtime instances** – **DEFERRED / UNVERIFIED** (dedicated test not run).

This local-only gateway setup has no user authentication or LiteLLM database-backed virtual keys. Do not expose it to untrusted networks or reuse its shared admin-level key in production.
