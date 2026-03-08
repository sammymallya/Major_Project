"""
Public interface for the KG reranker (cross-encoder) component.

This module provides a cross-encoder client that scores and re-ranks KG triples
by their relevance to a user query.
"""

from __future__ import annotations

import logging
from typing import Sequence

from sentence_transformers import CrossEncoder  # type: ignore[import]

from backend.kg.types import KgTriple

from .config import KgRerankerSettings, get_kg_reranker_settings
from .types import RerankedKgTriple, KgRerankDebugInfo

logger = logging.getLogger(__name__)


class KgRerankerClient:
    """
    Thin wrapper around a sentence-transformers CrossEncoder for KG triples.

    Responsibilities:
      - Loading the configured CrossEncoder model.
      - Scoring (query, triple) pairs.
      - Selecting the highest scoring triple(s) deterministically.
    """

    def __init__(self, settings: KgRerankerSettings | None = None) -> None:
        self._settings = settings or get_kg_reranker_settings()

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
    def settings(self) -> KgRerankerSettings:
        """Return the active settings for this client instance."""
        return self._settings

    def _make_pairs(self, query: str, triples: Sequence[KgTriple]) -> list[list[str]]:
        """Build CrossEncoder input pairs from query and triples."""
        pairs = []
        for triple in triples:
            # Format triple as "subject predicate object"
            triple_text = f"{triple.subject} {triple.predicate} {triple.object}"
            pairs.append([query, triple_text])
        return pairs

    def score_triples(self, query: str, triples: Sequence[KgTriple]) -> list[float]:
        """
        Score all KG triples using the cross-encoder.

        Args:
            query: Natural language query.
            triples: KG triples from fetch_kg.

        Returns:
            List of float scores aligned with the `triples` order.
        """

        if not triples:
            return []

        pairs = self._make_pairs(query, triples)
        scores = self._model.predict(pairs, batch_size=self._settings.batch_size)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        triples: Sequence[KgTriple],
        *,
        top_n: int = 1,
        include_debug: bool = False,
    ) -> tuple[list[RerankedKgTriple], KgRerankDebugInfo | None]:
        """
        Re-rank KG triples using the cross-encoder and return top-N.

        Deterministic tie-break: higher cross_score, then lexicographically
        smaller subject.
        """

        if top_n <= 0:
            logger.warning("Requested non-positive top_n=%s for rerank()", top_n)
            return [], None

        if not triples:
            logger.info("No triples to rerank; returning empty list")
            debug = None
            if include_debug:
                debug = KgRerankDebugInfo(
                    model_name=self._settings.cross_encoder_model_name,
                    device=self._settings.device,
                    input_count=0,
                    top_cross_scores=[],
                    selected_subject=None,
                )
            return [], debug

        scores = self.score_triples(query=query, triples=triples)

        reranked: list[RerankedKgTriple] = []
        for triple, s in zip(triples, scores, strict=False):
            reranked.append(
                RerankedKgTriple(
                    subject=triple.subject,
                    predicate=triple.predicate,
                    object=triple.object,
                    cross_score=float(s),
                )
            )

        reranked.sort(
            key=lambda r: (-r.cross_score, r.subject),
        )

        selected = reranked[: min(top_n, len(reranked))]
        logger.info(
            "KG rerank selected %d/%d triple(s) (top cross score=%.4f)",
            len(selected),
            len(reranked),
            selected[0].cross_score if selected else float("nan"),
        )

        debug_info: KgRerankDebugInfo | None = None
        if include_debug:
            top_cross_scores = [r.cross_score for r in reranked[: min(10, len(reranked))]]
            debug_info = KgRerankDebugInfo(
                model_name=self._settings.cross_encoder_model_name,
                device=self._settings.device,
                input_count=len(triples),
                top_cross_scores=top_cross_scores,
                selected_subject=selected[0].subject if selected else None,
            )

        return selected, debug_info


_client_singleton: KgRerankerClient | None = None


def _get_client() -> KgRerankerClient:
    """
    Lazily create and return a process-wide `KgRerankerClient` instance.
    """

    global _client_singleton
    if _client_singleton is None:
        _client_singleton = KgRerankerClient()
    return _client_singleton


def rerank_kg_triples(
    query: str,
    triples: Sequence[KgTriple],
    *,
    top_n: int = 1,
    include_debug: bool = False,
) -> tuple[list[RerankedKgTriple], KgRerankDebugInfo | None]:
    """
    Convenience wrapper to re-rank KG triples using a cross-encoder.

    Args:
        query: Natural language query.
        triples: Results returned by `fetch_kg`.
        top_n: Number of best matches to return after re-ranking.
        include_debug: When True, return additional debug information.

    Returns:
        A tuple `(reranked_triples, debug_info)`.
    """

    client = _get_client()
    return client.rerank(
        query=query,
        triples=triples,
        top_n=top_n,
        include_debug=include_debug,
    )
