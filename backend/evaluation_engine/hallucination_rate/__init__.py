"""
Hallucination Rate module.

Detects hallucinations by verifying claims against the memory layer (vector DB and KG).
"""

from .main import compute_hallucination_rate

__all__ = ["compute_hallucination_rate"]