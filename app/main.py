"""FastAPI application entry-point."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="CXR Diagnostics API",
    description="REST API for chest X-ray disease classification.",
    version="0.1.0",
)

app.include_router(router)
