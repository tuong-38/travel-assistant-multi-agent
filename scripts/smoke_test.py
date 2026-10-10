#!/usr/bin/env python3
"""Smoke test for API health endpoints.

Makes real HTTP requests to `/api/v1/health/live` and `/api/v1/health/ready`
using httpx. Exits with code 0 if both endpoints return status 200 and a
JSON payload containing `{"status": "ok"}` (the API's health responses).
Any network error, non‑200 status, JSON parse error, or unexpected payload
results in a non‑zero exit code and prints an error message to stderr.
"""

import sys
import argparse
import asyncio
import httpx

async def check(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                print(f"FAIL {url} returned status {resp.status_code}", file=sys.stderr)
                return False
            data = resp.json()
            # Expect a field "status" with value "ok" (common pattern in this project)
            if data.get("status") != "ok":
                print(f"FAIL {url} payload unexpected: {data}", file=sys.stderr)
                return False
            print(f"OK {url} ok")
            return True
    except Exception as e:
        print(f"FAIL {url} error: {e}", file=sys.stderr)
        return False

async def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke‑test API health endpoints")
    parser.add_argument("--host", default="127.0.0.1", help="API host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}/api/v1/health"
    live_ok = await check(f"{base}/live")
    ready_ok = await check(f"{base}/ready")
    sys.exit(0 if live_ok and ready_ok else 1)

if __name__ == "__main__":
    asyncio.run(main())
