# retrieval/search.py
from __future__ import annotations

"""
Hybrid search utilities for the ADAM API.

This module performs BM25 and dense embedding search over the hybrid index and
fuses the results with configurable weighting.  It also performs a simple
query correction using the corpus vocabulary.  This version normalises and
combines BM25 and dense scores safely, even when the two arrays have
different lengths (e.g., if the dense embedder returns fewer vectors than
there are chunks).  Missing scores are treated as 0.0.
"""

import logging
import re
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from retrieval.index_store import HybridIndex, RetrievalResult

# Initialize logger
log = logging.getLogger(__name__)


# ----------------------------
# Local, domain-agnostic tokenizer
# ----------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> List[str]:
    """Simple tokeniser that yields lowercase alphanumeric tokens."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


# ----------------------------
# Chunk accessor that works with multiple index shapes
# ----------------------------
def _get_chunks(index: HybridIndex):
    """Return a list of chunk objects from the index.

    The HybridIndex implementation may expose chunks via a method or an
    attribute; this helper tries both.  If neither exists, it returns an
    empty list.
    """
    # Prefer a method if it exists
    if hasattr(index, "all_chunks") and callable(getattr(index, "all_chunks")):
        try:
            return index.all_chunks()  # type: ignore[attr-defined]
        except Exception:
            pass
    # Fallback to attribute
    if hasattr(index, "chunks"):
        try:
            return getattr(index, "chunks")
        except Exception:
            pass
    return []


# ----------------------------
# Optional fuzzy correction (RapidFuzz)
# ----------------------------
def _maybe_correct_query_with_vocab(query: str, vocab: Iterable[str]) -> str:
    """Attempt to correct the query using vocabulary terms via rapidfuzz.

    If rapidfuzz is unavailable or no corrections apply, the original query
    string is returned.  Tokens shorter than 3 characters or digits are not
    corrected.  A match requires a token_sort_ratio of at least 87.
    """
    try:
        from rapidfuzz import fuzz, process
    except Exception:
        return query  # rapidfuzz not installed → no-op

    toks = _tokens(query)
    if not toks:
        return query

    vocab_set = set(vocab)
    out: List[str] = []
    for t in toks:
        if len(t) < 3 or t.isdigit() or t in vocab_set:
            out.append(t)
            continue
        best = process.extractOne(t, vocab_set, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= 87:
            out.append(best[0])
        else:
            out.append(t)
    fixed = " ".join(out)
    if fixed != query:
        log.info("retrieval.search: corrected query: %r -> %r", query, fixed)
    return fixed


def _correct_query_with_index(index: HybridIndex, query: str) -> str:
    """Correct the query using the index’s correction method or local vocabulary.

    If the index exposes a `correct_query` method, it will be used.  Otherwise
    a vocabulary is built from chunk text, section and heading metadata, and
    `_maybe_correct_query_with_vocab` is invoked.
    """
    # Use index.correct_query if provided
    if hasattr(index, "correct_query") and callable(getattr(index, "correct_query")):
        try:
            return index.correct_query(query)  # type: ignore[attr-defined]
        except Exception:
            pass
    # Build a quick vocab from current chunks + section/heading
    vocab: set[str] = set()
    for c in _get_chunks(index):
        try:
            vocab.update(_tokens(getattr(c, "text", "")))
            md = getattr(c, "metadata", {}) or {}
            vocab.update(_tokens(str(md.get("section", ""))))
            vocab.update(_tokens(str(md.get("heading", ""))))
        except Exception:
            continue
    if not vocab:
        return query
    return _maybe_correct_query_with_vocab(query, vocab)


# ----------------------------
# Embedding loader (singleton)
# ----------------------------
def _cuda_ok() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model.

    The model name and device can be customised via environment variables.
    """
    import os
    model_name = os.getenv("EMBED_MODEL", "ibm-granite/granite-embedding-english-r2")
    device = "cuda" if _cuda_ok() else "cpu"
    log.info("retrieval.search: loading embedder %s (%s)", model_name, device)
    return SentenceTransformer(model_name, device=device)


# ----------------------------
# Helpers
# ----------------------------
def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    da, db = float(norm(a)), float(norm(b))
    if da == 0 or db == 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0,1].  Returns zeros if `arr` is empty or flat."""
    if arr.size == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)


# ----------------------------
# Build BM25 corpus on demand
# ----------------------------
def _build_bm25(index: HybridIndex) -> Tuple[BM25Okapi, List[str]]:
    """Build (or update) the BM25 index and return the corpus texts."""
    chunks = _get_chunks(index)
    texts = [getattr(c, "text", "") for c in chunks]
    tokenized = [_tokens(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, texts


# ----------------------------
# On-the-fly embeddings (cache small corpora)
# ----------------------------
@lru_cache(maxsize=2)
def _embed_chunk_texts(sig: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    """Internal stub used for caching; replaced at runtime."""
    raise RuntimeError("internal cache stub")


def _get_embeddings_for_chunks(texts: List[str]) -> np.ndarray:
    """Get embeddings for a list of chunk texts, caching them by length signature."""
    sig = tuple((i, len(t)) for i, t in enumerate(texts))
    embedder = _get_embedder()

    def _compute(_: Tuple[Tuple[int, int], ...]) -> np.ndarray:
        embs = embedder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(embs, dtype=np.float32)

    try:
        return _embed_chunk_texts(sig)
    except Exception:
        pass

    vecs = _compute(sig)
    _embed_chunk_texts.cache_clear()

    def _seed(sig_param: Tuple[Tuple[int, int], ...]) -> np.ndarray:
        return vecs

    _seed.__name__ = "_embed_chunk_texts"
    globals()["_embed_chunk_texts"] = lru_cache(maxsize=2)(_seed)  # type: ignore
    return _embed_chunk_texts(sig)


# ----------------------------
# Search (hybrid + correction)
# ----------------------------
def search_documents(index: HybridIndex, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
    """Hybrid retrieval over the hybrid index.

    This function performs the following steps:

      1) Optionally correct the query using the corpus vocabulary or the index’s
         own correction method.
      2) Score all chunks with BM25.
      3) Score all chunks with dense embeddings (cosine similarity).
      4) Min–max normalise both score lists.
      5) Fuse the two scores with weights 0.45 (BM25) and 0.55 (dense),
         padding the shorter score list with zeros so that the fused array
         always matches the length of the chunk list.
      6) Return the top_k `RetrievalResult` objects.
    """
    k = int(top_k or 8)

    # 1) Query correction
    q = _correct_query_with_index(index, query)

    chunks = _get_chunks(index)
    if not chunks:
        return []

    # 2) BM25 search
    bm25, texts = _build_bm25(index)
    q_tokens = _tokens(q)
    bm25_scores = np.asarray(bm25.get_scores(q_tokens), dtype=np.float32)

    # 3) Dense search
    q_vec = _get_embedder().encode([q], convert_to_numpy=True, normalize_embeddings=False)[0].astype(np.float32)
    c_vecs = _get_embeddings_for_chunks(texts)
    dense_scores = np.asarray([_cos_sim(q_vec, v) for v in c_vecs], dtype=np.float32)

    # 4) Normalise
    b_norm = _minmax_norm(bm25_scores)
    d_norm = _minmax_norm(dense_scores)

    # 5) Fuse – pad the shorter array with zeros to match `len(chunks)`
    fused = np.zeros(len(chunks), dtype=np.float32)
    # BM25 and dense score arrays may have different lengths; treat missing values as zero
    for i in range(len(chunks)):
        b_val = b_norm[i] if i < b_norm.size else 0.0
        d_val = d_norm[i] if i < d_norm.size else 0.0
        fused[i] = 0.45 * b_val + 0.55 * d_val

    # 6) Select top_k
    idxs = np.argsort(-fused)[:k]
    results: List[RetrievalResult] = [
        RetrievalResult(chunk=chunks[int(i)], score=float(fused[int(i)])) for i in idxs
    ]

    # Debug preview of top results
    try:
        preview = []
        for rr in results[:3]:
            text = (rr.chunk.text or "").replace("\n", " ")
            preview.append(f"[{rr.score:.3f}] {text[:180]}")
        log.info("retrieval.search: top hits for %r:\n%s", q, "\n".join(preview))
    except Exception:
        pass

    return results