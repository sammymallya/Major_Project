"""
Public interface for the vector database (Pinecone) component.

This module exposes a small, stable API that the rest of the system can use
to interact with the vector database without knowing any Pinecone internals.

Key entry point:
    - `fetch_top_vectordb(n, query)`:
        Send the query to Pinecone's serverless embedding (llama-text-embed-v2)
        and return the top `n` most similar results.
"""

from __future__ import annotations

import logging
from typing import List

from pinecone import Pinecone  # type: ignore[import]

from .config import get_vectordb_settings, VectorDBSettings
from .types import VectorQueryDebugInfo, VectorResult

logger = logging.getLogger(__name__)


class VectorDBClient:
    """
    Thin wrapper around Pinecone with serverless embedding support.

    This class is responsible for:
      - Initializing the Pinecone client and index.
      - Delegating query embedding to Pinecone's serverless llama-text-embed-v2 model.
      - Executing similarity queries and mapping raw results to `VectorResult`.
    """

    def __init__(self, settings: VectorDBSettings | None = None) -> None:
        self._settings = settings or get_vectordb_settings()

        logger.info(
            "Initialising VectorDBClient with index '%s' (environment hint: '%s')",
            self._settings.pinecone_index_name,
            self._settings.pinecone_environment,
        )

        # Pinecone v3 client: create a Pinecone instance and then obtain an Index handle.
        self._pc = Pinecone(api_key=self._settings.pinecone_api_key)
        self._index = self._pc.Index(self._settings.pinecone_index_name)
        
        logger.info(
            "Using Pinecone serverless embedding (llama-text-embed-v2) for index '%s'",
            self._settings.pinecone_index_name,
        )

    @property
    def settings(self) -> VectorDBSettings:
        """
        Return the active settings for this client instance.
        """

        return self._settings

    def fetch_top(
        self,
        n: int,
        query: str,
        *,
        include_debug: bool = False,
    ) -> tuple[List[VectorResult], VectorQueryDebugInfo | None]:
        """
        Retrieve the top `n` most similar records from Pinecone for the query.

        The query is embedded server-side using Pinecone's llama-text-embed-v2 model,
        ensuring consistency with the data embeddings.

        Args:
            n: Number of top results to return.
            query: Natural language search query.
            include_debug: When True, also return a `VectorQueryDebugInfo`
                instance with additional scoring details.

        Returns:
            A tuple `(results, debug_info)` where:
              - `results` is a list of `VectorResult`.
              - `debug_info` is either `VectorQueryDebugInfo` or `None`.
        """

        if n <= 0:
            logger.warning("Requested non-positive number of results: %s", n)
            return [], None

        query_kwargs: dict = {
            "query_text": query,
            "top_k": n,
            "include_metadata": True,
        }
        if self._settings.pinecone_namespace:
            query_kwargs["namespace"] = self._settings.pinecone_namespace

        logger.info(
            "Querying Pinecone index '%s' for top %d results (namespace=%s) using serverless embedding",
            self._settings.pinecone_index_name,
            n,
            self._settings.pinecone_namespace,
        )
        response = self._index.query(**query_kwargs)

        matches = response.get("matches", []) or []
        results: list[VectorResult] = []
        top_scores: list[float] = []

        for match in matches:
            score = float(match.get("score", 0.0))
            top_scores.append(score)

            metadata = match.get("metadata") or {}
            text = metadata.get("text") or ""

            results.append(
                VectorResult(
                    id=str(match.get("id")),
                    text=text,
                    score=score,
                    metadata=metadata,
                )
            )

        logger.info("Vector search returned %d result(s)", len(results))

        debug_info: VectorQueryDebugInfo | None = None
        if include_debug:
            debug_info = VectorQueryDebugInfo(
                model_name="llama-text-embed-v2 (Pinecone serverless)",
                top_scores=top_scores,
                namespace=self._settings.pinecone_namespace,
            )

        return results, debug_info


_client_singleton: VectorDBClient | None = None


def _get_client() -> VectorDBClient:
    """
    Lazily create and return a process-wide `VectorDBClient` instance.
    """

    global _client_singleton
    if _client_singleton is None:
        _client_singleton = VectorDBClient()
    return _client_singleton


def fetch_top_vectordb(
    n: int,
    query: str,
    *,
    include_debug: bool = False,
) -> tuple[List[VectorResult], VectorQueryDebugInfo | None]:
    """
    Convenience wrapper used by the rest of the system.

    This function hides the underlying client implementation so callers only
    need to import a single symbol from `vector_db.main`.

    Args:
        n: Number of top results to retrieve.
        query: Natural language search query.
        include_debug: When True, also return debug information alongside
            the results.

    Returns:
        A tuple `(results, debug_info)` described in `VectorDBClient.fetch_top`.
    """

    client = _get_client()
    return client.fetch_top(n=n, query=query, include_debug=include_debug)

