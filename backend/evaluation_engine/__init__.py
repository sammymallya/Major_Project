"""
Evaluation Engine package.

This package provides comprehensive evaluation of the hybrid memory augmented LLM pipeline,
including semantic matching, F1 score, and hallucination rate metrics.
"""

from .main import evaluate_pipeline

__all__ = ["evaluate_pipeline"]