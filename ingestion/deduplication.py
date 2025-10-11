"""
Duplicate detection and removal utilities.

Ingestion pipelines often generate duplicated or near‑duplicated chunks when
documents contain repeated sections (e.g. definitions or boilerplate
language).  This module provides simple heuristics to detect and filter
duplicate chunks based on normalised token overlap.  Removing such
redundancies improves retrieval performance and reduces the storage
footprint【849888487782910†L62-L90】.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Set

from .chunker import Chunk

logger = logging.getLogger(__name__)


def _normalise_text(text: str) -> Set[str]:
    """Normalise text by lowercasing and splitting on whitespace and simple
    punctuation.  Returns a set of tokens for Jaccard similarity."""
    # Replace punctuation with spaces and split
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    tokens = {t for t in cleaned.split() if t}
    return tokens


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Compute the Jaccard similarity between two sets of tokens."""
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)


def deduplicate_chunks(chunks: List[Chunk], *, threshold: float = 0.85) -> List[Chunk]:
    """Remove duplicate or near‑duplicate chunks based on Jaccard similarity.

    Parameters
    ----------
    chunks:
        List of chunks to deduplicate.  Chunks are expected to be pre‑chunked
        and optionally summarised.
    threshold:
        Jaccard similarity above which two chunks are considered duplicates.
        Values close to 1.0 require near‑identical text, while lower values
        will merge more aggressively.  The default of 0.85 strikes a
        compromise.

    Returns
    -------
    list of :class:`Chunk`
        The filtered list of unique chunks.

    Notes
    -----
    The algorithm is O(n^2) in the number of chunks.  For large document
    collections consider using MinHash or locality sensitive hashing to
    accelerate duplicate detection.  For the typical ingestion sizes
    encountered in enterprise document repositories, this implementation
    suffices.
    """
    unique_chunks: List[Chunk] = []
    unique_token_sets: List[Set[str]] = []
    for chunk in chunks:
        tokens = _normalise_text(chunk.text)
        is_duplicate = False
        for prev_tokens in unique_token_sets:
            sim = _jaccard(tokens, prev_tokens)
            if sim >= threshold:
                is_duplicate = True
                logger.debug(
                    "Dropping duplicate chunk %s (similarity %.2f)",
                    chunk.id,
                    sim,
                )
                break
        if not is_duplicate:
            unique_chunks.append(chunk)
            unique_token_sets.append(tokens)
    return unique_chunks
