"""
FastAPI application entry point.

Mounts the query router. Run with: uvicorn backend.api.main:app --reload
"""

import logging

from fastapi import FastAPI

from backend.api.query import router as query_router

# Reduce noise from third-party loggers when running the API
for name in ("httpx", "huggingface_hub", "sentence_transformers", "transformers", "urllib3", "openai"):
    logging.getLogger(name).setLevel(logging.WARNING)

app = FastAPI(
    title="Hybrid Memory Augmented LLM",
    description="Query pipeline with rule-based routing (vectordb / kg / hybrid / none).",
    version="0.1.0",
)

app.include_router(query_router)


@app.get("/health")
def health() -> dict:
    """Health check for deployment."""
    return {"status": "ok"}
