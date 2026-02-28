"""
Query pipeline router.

POST /query accepts query and test_mode, runs the orchestration pipeline,
and returns the response DTO. Thin handler: delegates to services.
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.dto import QueryRequest, QueryResponse
from backend.services import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def post_query(body: QueryRequest) -> QueryResponse:
    """
    Run the query pipeline for the given query and test_mode.

    Routes to vector DB, KG (stub), or both based on test_mode; builds
    prompt and returns the Test LLM (stub) answer with optional context_used.
    """
    try:
        return run_pipeline(query=body.query, test_mode=body.test_mode)
    except Exception as e:
        logger.exception("Pipeline failed for test_mode=%s: %s", body.test_mode, e)
        raise HTTPException(
            status_code=500,
            detail="PIPELINE_ERROR: The query pipeline failed. Please try again.",
        ) from e
