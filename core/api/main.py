"""Core recommendation service.

Run locally from the repo root:
    uvicorn core.api.main:app --port 8000 --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from core.api.routes import events

app = FastAPI(title="Recommendation core")
app.include_router(events.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
