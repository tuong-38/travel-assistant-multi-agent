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
4. Start the database, gateway, and API with `docker compose up --build`.

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

## Checks

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
docker compose config
```

This local-only gateway setup has no user authentication or LiteLLM database-backed virtual keys. Do not expose it to untrusted networks or reuse its shared admin-level key in production.
