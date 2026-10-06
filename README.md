# Travel Assistant Multi-Agent System

Phase 0 provides a small Python API foundation: FastAPI, a deterministic one-node LangGraph workflow, and PostgreSQL-backed LangGraph checkpoints. It does not call an LLM or any travel service.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Docker with Docker Compose

## Local development

1. Copy `.env.example` to `.env` and replace the local database password if desired. Keep `.env` out of Git.
2. Install locked dependencies and create the environment:

   ```sh
   uv sync --locked
   ```

3. Start PostgreSQL:

   ```sh
   docker compose up -d db
   ```

4. Start the API:

   ```sh
   uv run uvicorn app.main:app --reload
   ```

   On native Windows, Psycopg's async driver requires a selector event loop. Start Uvicorn with that loop policy:

   ```powershell
   uv run python -c "import asyncio, uvicorn; asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()); uvicorn.run('app.main:app')"
   ```

Or start both services in containers with `docker compose up --build`.

The Compose database password fallback is for local development only. Set `POSTGRES_PASSWORD` in the environment or `.env` before using a shared environment. The API uses the Compose-internal database URL when run in Compose; for host-side development, set `DATABASE_URL` in `.env` to `postgresql://travel_assistant:<password>@localhost:5432/travel_assistant`.

## API

- `GET /api/v1/health/live` — process liveness.
- `GET /api/v1/health/ready` — database readiness.
- `POST /api/v1/chat` — deterministic reply, persisted by LangGraph thread ID.

Example request:

```json
{
  "message": "Hello",
  "thread_id": "9de6a955-31ee-4a0c-b877-34123cd130d5"
}
```

`thread_id` is optional; the API returns a generated UUID when omitted. The response contains `thread_id` and `reply`. The sample workflow replies with `You said: <message>` and stores graph state in PostgreSQL checkpoints.

## Checks

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
docker compose config
```

The initial API has no authentication and is intended only for local or otherwise protected environments.
