"""
Public interface for the cross-encoder re-ranking component.

This module exposes a small, stable API that scores vector DB candidates with
a CrossEncoder and returns the best match (or top-N matches).
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

from sentence_transformers import CrossEncoder  # type: ignore[import]

from backend.vector_db.types import VectorResult

from .config import RerankerSettings, get_reranker_settings
from .types import RerankDebugInfo, RerankedVectorResult

logger = logging.getLogger(__name__)


class CrossEncoderRerankerClient:
    """
    Thin wrapper around a sentence-transformers CrossEncoder.

    Responsibilities:
      - Loading the configured CrossEncoder model.
      - Scoring (query, passage) pairs.
      - Selecting the highest scoring candidate deterministically.
    """

    def __init__(self, settings: RerankerSettings | None = None) -> None:
        self._settings = settings or get_reranker_settings()

        logger.info(
            "Loading cross-encoder model '%s' (device=%s, batch_size=%d)",
            self._settings.cross_encoder_model_name,
            self._settings.device,
            self._settings.batch_size,
        )

        init_kwargs: dict = {}
        if self._settings.device:
            init_kwargs["device"] = self._settings.device
        self._model = CrossEncoder(self._settings.cross_encoder_model_name, **init_kwargs)

    @property
    def settings(self) -> RerankerSettings:
        """
        Return the active settings for this client instance.
        """

        return self._settings

    def _make_pairs(self, query: str, passages: Iterable[str]) -> list[list[str]]:
        """
        Build CrossEncoder input pairs in the expected [[q, p], ...] format.
        """

        return [[query, p] for p in passages]

    def score_candidates(self, query: str, candidates: Sequence[VectorResult]) -> list[float]:
        """
        Score all candidates using the cross-encoder.

        Args:
            query: Natural language query.
            candidates: Candidate passages from vector retrieval.

        Returns:
            List of float scores aligned with the `candidates` order.
        """

        if not candidates:
            return []

        pairs = self._make_pairs(query, (c.text for c in candidates))
        scores = self._model.predict(pairs, batch_size=self._settings.batch_size)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: Sequence[VectorResult],
        *,
        top_n: int = 3,
        include_debug: bool = False,
    ) -> tuple[List[RerankedVectorResult], RerankDebugInfo | None]:
        """
        Re-rank vector candidates using the cross-encoder and return top-N.

        Deterministic tie-break: higher cross_score, then higher vector_score,
        then lexicographically smaller id.
        """

        if top_n <= 0:
            logger.warning("Requested non-positive top_n=%s for rerank()", top_n)
            return [], None

        if not candidates:
            logger.info("No candidates to rerank; returning empty list")
            debug = None
            if include_debug:
                debug = RerankDebugInfo(
                    model_name=self._settings.cross_encoder_model_name,
                    device=self._settings.device,
                    input_count=0,
                    top_cross_scores=[],
                    selected_id=None,
                )
            return [], debug

        scores = self.score_candidates(query=query, candidates=candidates)

        reranked: list[RerankedVectorResult] = []
        for cand, s in zip(candidates, scores, strict=False):
            reranked.append(
                RerankedVectorResult(
                    id=cand.id,
                    text=cand.text,
                    vector_score=float(cand.score),
                    cross_score=float(s),
                    metadata=cand.metadata,
                )
            )

        reranked.sort(
            key=lambda r: (-r.cross_score, -r.vector_score, str(r.id)),
        )

        selected = reranked[: min(top_n, len(reranked))]
        logger.info(
            "Cross-encoder rerank selected %d/%d candidate(s) (top cross score=%.4f)",
            len(selected),
            len(reranked),
            selected[0].cross_score if selected else float("nan"),
        )

        debug_info: RerankDebugInfo | None = None
        if include_debug:
            top_cross_scores = [r.cross_score for r in reranked[: min(10, len(reranked))]]
            debug_info = RerankDebugInfo(
                model_name=self._settings.cross_encoder_model_name,
                device=self._settings.device,
                input_count=len(candidates),
                top_cross_scores=top_cross_scores,
                selected_id=selected[0].id if selected else None,
            )

        return selected, debug_info


_client_singleton: CrossEncoderRerankerClient | None = None


def _get_client() -> CrossEncoderRerankerClient:
    """
    Lazily create and return a process-wide `CrossEncoderRerankerClient` instance.
    """

    global _client_singleton
    if _client_singleton is None:
        _client_singleton = CrossEncoderRerankerClient()
    return _client_singleton


def rerank_top_cross_encoder(
    query: str,
    candidates: Sequence[VectorResult],
    *,
    top_n: int = 1,
    include_debug: bool = False,
) -> tuple[List[RerankedVectorResult], RerankDebugInfo | None]:
    """
    Convenience wrapper to re-rank vector DB candidates using a cross-encoder.

    Args:
        query: Natural language query.
        candidates: Results returned by `fetch_top_vectordb`.
        top_n: Number of best matches to return after re-ranking.
        include_debug: When True, return additional debug information.

    Returns:
        A tuple `(reranked_results, debug_info)`.
    """

    client = _get_client()
    return client.rerank(
        query=query,
        candidates=candidates,
        top_n=top_n,
        include_debug=include_debug,
    )

