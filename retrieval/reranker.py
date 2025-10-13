# retrieval/reranker.py
from __future__ import annotations
import logging
import math
from typing import Any, List, Union

import numpy as np

try:
    import torch
    from sentence_transformers import CrossEncoder
except Exception as exc:
    CrossEncoder = None
    _IMPORT_ERR = str(exc)
else:
    _IMPORT_ERR = None

import config

logger = logging.getLogger(__name__)

ResultT = Union[dict, Any]  # dict or object with .text/.score etc.


def _as_text(r: ResultT) -> str:
    if isinstance(r, dict):
        return str(r.get("text") or r.get("chunk_text") or "")
    return str(getattr(r, "text", "") or getattr(r, "chunk_text", "") or "")


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return v


def _attach_score(r: ResultT, score: float) -> None:
    score = _safe_float(score)
    if isinstance(r, dict):
        r["rerank_score"] = score
    else:
        setattr(r, "rerank_score", score)


class Reranker:
    def __init__(self) -> None:
        if CrossEncoder is None:
            raise RuntimeError(
                f"CrossEncoder not available: {_IMPORT_ERR}. "
                "Install 'sentence-transformers' and 'torch'."
            )
        model_name = getattr(
            config, "RERANKER_MODEL_NAME",
            "ibm-granite/granite-embedding-reranker-english-r2"
        )
        device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
        logger.info("Loading sentence reranker: %s (%s)", model_name, device)

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=getattr(config, "RERANKER_MAX_LENGTH", 512),
        )
        self.batch_size = getattr(config, "RERANKER_BATCH_SIZE", 16)

    def rank(self, query: str, results: List[ResultT], top_k: int = 5) -> List[ResultT]:
        """Return results with .rerank_score attached, sorted desc; fall back to original order on failure."""
        if not results:
            return []

        pairs = []
        keep_idx = []
        for i, r in enumerate(results):
            t = _as_text(r).strip()
            if t:
                pairs.append((query, t))
                keep_idx.append(i)

        if not pairs:
            logger.warning("Reranker: all texts empty; keeping original order.")
            return results[:max(1, int(top_k))]

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.exception("Reranker predict failed: %s. Keeping original order.", e)
            return results[:max(1, int(top_k))]

        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

        for j, idx in enumerate(keep_idx):
            _attach_score(results[idx], float(scores[j]))

        if float(np.max(scores)) <= 0.0:
            logger.warning("Reranker: all scores <= 0 after sanitize; keeping original order.")
            return results[:max(1, int(top_k))]

        def _score_of(x: ResultT) -> float:
            if isinstance(x, dict):
                s = x.get("rerank_score", None)
                if s is None:
                    s = x.get("score", 0.0)
            else:
                s = getattr(x, "rerank_score", None)
                if s is None:
                    s = getattr(x, "score", 0.0)
            return _safe_float(s)

        results_sorted = sorted(results, key=_score_of, reverse=True)
        return results_sorted[:max(1, int(top_k))]
