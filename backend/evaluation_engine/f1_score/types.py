"""
Typed representations for the F1 Score component.
"""

from dataclasses import dataclass


@dataclass
class F1ScoreResult:
    """
    Result of F1 score computation.

    Attributes:
        f1: F1 score (0.0 to 1.0).
        precision: Precision score.
        recall: Recall score.
        rouge_type: ROUGE type used.
    """

    f1: float
    precision: float
    recall: float
    rouge_type: str