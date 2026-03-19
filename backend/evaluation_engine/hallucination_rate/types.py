"""
Typed representations for the Hallucination Rate component.
"""

from dataclasses import dataclass


@dataclass
class HallucinationRateResult:
    """
    Result of hallucination rate computation.

    Attributes:
        rate: Hallucination rate (0.0 to 1.0, fraction of unverified claims).
        total_claims: Total number of claims extracted.
        verified_claims: Number of claims verified against memory layer.
        unverified_claims: Number of claims not verified.
    """

    rate: float
    total_claims: int
    verified_claims: int
    unverified_claims: int