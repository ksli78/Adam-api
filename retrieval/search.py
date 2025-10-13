# retrieval/search.py
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Iterable
from functools import lru_cache

import numpy as np
from numpy.linalg import norm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from retrieval.index_store import HybridIndex, RetrievalResult

log = logging.getLogger(__name__)

# ----------------------------
# Local, domain-agnostic tokenizer
# ----------------------------
import re
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]

# ----------------------------
# Chunk accessor that works with multiple index shapes
# ----------------------------
def _get_chunks(index: HybridIndex):
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
    # Nothing available
    return []

# ----------------------------
# Optional fuzzy correction (RapidFuzz)
# ----------------------------
def _maybe_correct_query_with_vocab(query: str, vocab: Iterable[str]) -> str:
    try:
        from rapidfuzz import process, fuzz
    except Exception:
        return query  # rapidfuzz not installed → no-op

    toks = _tokens(query)
    if not toks:
        return query

    vocab = set(vocab)
    out: List[str] = []
    for t in toks:
        if len(t) < 3 or t.isdigit() or t in vocab:
            out.append(t); continue
        best = process.extractOne(t, vocab, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= 87:
            out.append(best[0])
        else:
            out.append(t)
    fixed = " ".join(out)
    if fixed != query:
        log.info("retrieval.search: corrected query: %r -> %r", query, fixed)
    return fixed

def _correct_query_with_index(index: HybridIndex, query: str) -> str:
    # Use index.correct_query if provided
    if hasattr(index, "correct_query") and callable(getattr(index, "correct_query")):
        try:
            return index.correct_query(query)  # type: ignore[attr-defined]
        except Exception:
            pass
    # Build a quick vocab from current chunks + section/heading
    vocab = set()
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
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    import os
    model_name = os.getenv("EMBED_MODEL", "ibm-granite/granite-embedding-english-r2")
    device = "cuda" if _cuda_ok() else "cpu"
    log.info("retrieval.search: loading embedder %s (%s)", model_name, device)
    return SentenceTransformer(model_name, device=device)

# ----------------------------
# Helpers
# ----------------------------
def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    da, db = float(norm(a)), float(norm(b))
    if da == 0 or db == 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
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
    raise RuntimeError("internal cache stub")

def _get_embeddings_for_chunks(texts: List[str]) -> np.ndarray:
    sig = tuple((i, len(t)) for i, t in enumerate(texts))
    embedder = _get_embedder()

    def _compute(_: Tuple[Tuple[int, int], ...]) -> np.ndarray:
        embs = embedder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=False
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
    globals()["_embed_chunk_texts"] = lru_cache(maxsize=2)(_seed)
    return _embed_chunk_texts(sig)

# ----------------------------
# Search (hybrid + correction)
# ----------------------------
def search_documents(index: HybridIndex, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
    """
    Hybrid retrieval:
      1) Correct query with corpus vocabulary (uses index.correct_query if available; else builds vocab).
      2) Rank with BM25 (rank_bm25).
      3) Rank with dense embeddings (SentenceTransformer).
      4) Fuse (0.55 dense + 0.45 BM25) after min-max normalization.
      5) Return top_k results.
    """
    k = int(top_k or 8)

    # 1) corpus-driven spell-correct (no-op if rapidfuzz missing)
    q = _correct_query_with_index(index, query)

    chunks = _get_chunks(index)
    if not chunks:
        return []

    # --- BM25 ---
    bm25, texts = _build_bm25(index)
    q_tokens = _tokens(q)
    bm25_scores = np.asarray(bm25.get_scores(q_tokens), dtype=np.float32)

    # --- Dense ---
    q_vec = _get_embedder().encode([q], convert_to_numpy=True, normalize_embeddings=False)[0].astype(np.float32)
    c_vecs = _get_embeddings_for_chunks(texts)
    dense_scores = np.asarray([_cos_sim(q_vec, v) for v in c_vecs], dtype=np.float32)

    # --- Fuse ---
    b_norm = _minmax_norm(bm25_scores)
    d_norm = _minmax_norm(dense_scores)
    fused = 0.45 * b_norm + 0.55 * d_norm

    # --- Top-k ---
    idxs = np.argsort(-fused)[:k]
    results: List[RetrievalResult] = [
        RetrievalResult(chunk=chunks[int(i)], score=float(fused[int(i)])) for i in idxs
    ]

    # Debug preview
    try:
        preview = []
        for rr in results[:3]:
            text = (rr.chunk.text or "").replace("\n", " ")
            preview.append(f"[{rr.score:.3f}] {text[:180]}")
        log.info("retrieval.search: top hits for %r:\n%s", q, "\n".join(preview))
    except Exception:
        pass

    return results
