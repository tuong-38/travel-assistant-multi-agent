from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> Any:
    pool = request.app.state.db_pool
    try:
        async with pool.connection() as connection:
            await connection.execute("SELECT 1")
        settings = get_settings()
        base_url = settings.litellm_base_url.removesuffix("/v1").rstrip("/")
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{base_url}/health/readiness")
            response.raise_for_status()
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ok"}
