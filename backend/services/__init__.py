"""
Business logic and orchestration.

Exposes run_pipeline for the API layer. No FastAPI or transport details here.
"""

from .orchestration import run_pipeline

__all__ = ["run_pipeline"]
