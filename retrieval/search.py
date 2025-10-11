"""
High‑level search orchestration.

This module defines a convenience function that performs a two‑stage search
over the hybrid index: first retrieving candidate chunks using the hybrid
dense/lexical similarity metric, then applying the cross‑encoder reranker to
produce the final ranking.
"""

from __future__ import annotations

from typing import List

import config
from .index_store import HybridIndex, RetrievalResult
from .reranker import Reranker


def search_documents(index: HybridIndex, query: str, *, top_k: int | None = None) -> List[RetrievalResult]:
    """Search the hybrid index and rerank the results.

    Parameters
    ----------
    index:
        The hybrid index to search.
    query:
        The user query.
    top_k:
        Number of results to return.  Defaults to :data:`config.TOP_K_RERANKED`.

    Returns
    -------
    list of :class:`RetrievalResult`
        The reranked search results.
    """
    top_k = top_k or config.TOP_K_RERANKED
    # Stage 1: dense + lexical retrieval
    ranked_indices = index.search(query, top_k=config.TOP_K, alpha=config.ALPHA)
    candidates = index.get_results(ranked_indices)
    # Stage 2: rerank
    reranker = Reranker()
    reranked = reranker.rank(query, candidates, top_k=top_k)
    return reranked
