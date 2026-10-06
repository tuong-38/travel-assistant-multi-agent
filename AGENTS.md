# Repository guidance

- Keep Phase 0 small: FastAPI, one deterministic LangGraph chat node, and PostgreSQL checkpoints.
- Do not add model providers, travel APIs, extra infrastructure, or agent orchestration without an explicit request.
- Read configuration from environment variables through `app.core.config.Settings`.
- Never commit real credentials. Update `.env.example` only with safe placeholders.
- Keep API routes under `/api/v1` and add focused pytest coverage for behavior changes.
- Use `uv` and keep `uv.lock` committed and synchronized with `pyproject.toml`.

