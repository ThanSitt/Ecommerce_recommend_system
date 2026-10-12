"""Core recommendation service.

Run locally from the repo root:
    uvicorn core.api.main:app --port 8000 --reload
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Recommendation core")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
