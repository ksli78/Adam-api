"""
Reranker using IBM's Granite cross‑encoder.

The retriever (dense + lexical search) produces an initial ranking of
documents.  A cross‑encoder reranker jointly encodes the query and each
candidate document to compute a more precise relevance score.  IBM's
``granite‑embedding‑reranker‑english‑r2`` model is designed for this task
and achieves strong performance on long document retrieval benchmarks【449332267749196†L65-L70】.

This module provides a simple wrapper that loads the model via
``sentence_transformers.CrossEncoder`` and exposes a ``rank`` method.
"""

from __future__ import annotations

import logging
from typing import List

from sentence_transformers import CrossEncoder

import config
from .index_store import RetrievalResult

logger = logging.getLogger(__name__)


_CROSS_ENCODER: CrossEncoder | None = None


def _load_reranker() -> CrossEncoder:
    """Load the cross‑encoder model if not already loaded."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        logger.info("Loading reranker model %s", config.RERANKER_MODEL_NAME)
        _CROSS_ENCODER = CrossEncoder(config.RERANKER_MODEL_NAME)
    return _CROSS_ENCODER


class Reranker:
    """Wrapper around a cross‑encoder reranker model."""

    def __init__(self) -> None:
        self.model = _load_reranker()

    def rank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        *,
        top_k: int = config.TOP_K_RERANKED,
    ) -> List[RetrievalResult]:
        """Reorder the candidate results using the cross‑encoder.

        Parameters
        ----------
        query:
            The search query.
        candidates:
            List of retrieval results to rerank.
        top_k:
            Return at most this many results after reranking.

        Returns
        -------
        list of :class:`RetrievalResult`
            The reranked results sorted by descending relevance score.
        """
        if not candidates:
            return []
        # Prepare inputs for cross encoder: list of [query, document] pairs
        pairs = [[query, res.chunk.text] for res in candidates]
        # Compute relevance scores.  Higher scores indicate greater relevance.
        scores = self.model.predict(pairs)
        # Attach new scores to results
        for res, score in zip(candidates, scores):
            res.score = float(score)
        # Sort by score descending
        ranked = sorted(candidates, key=lambda r: r.score, reverse=True)
        return ranked[:top_k]
