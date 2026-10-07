# Repository guidance

- Keep the project small: FastAPI, the approved supervisor-led LangGraph travel workflow, LiteLLM gateway, PostgreSQL checkpoints, and the approved travel-tools-mcp service.
- Do not add extra infrastructure or travel APIs. The bounded MCP capability is destination-only `search_destination`; planner and itinerary remain MCP-free. Do not add other orchestration without an explicit request.
- Read configuration from environment variables through `app.core.config.Settings`.
- Never commit real credentials. Update `.env.example` only with safe placeholders.
- Keep gateway credentials in runtime environment variables; the shared LiteLLM master/API key is approved for local/internal development only.
- Keep API routes under `/api/v1` and add focused pytest coverage for behavior changes.
- Use `uv` and keep `uv.lock` committed and synchronized with `pyproject.toml`.

