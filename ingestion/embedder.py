"""
Embedding utilities.

This module provides a thin wrapper around IBM's Granite embedding model to
produce dense vector representations for text chunks.  We use the
``sentence_transformers`` interface for convenience; it loads the model via
Hugging Face Transformers and handles batching, tokenisation and device
placement internally.  The default model generates 768‑dimensional
embeddings and supports up to 8k token inputs【449332267749196†L57-L83】.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np

from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


_EMBED_MODEL: SentenceTransformer | None = None


def _load_embed_model() -> SentenceTransformer:
    """Load the embedding model from Hugging Face if not already loaded."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        logger.info("Loading embedding model %s", config.EMBED_MODEL_NAME)
        # SentenceTransformer automatically selects the GPU if available.
        _EMBED_MODEL = SentenceTransformer(config.EMBED_MODEL_NAME)
    return _EMBED_MODEL


def embed_texts(texts: Iterable[str], *, batch_size: int = 32) -> List[np.ndarray]:
    """Compute dense embeddings for a list of texts.

    Parameters
    ----------
    texts:
        Iterable of strings to embed.
    batch_size:
        Batch size for embedding.  Tune this according to available GPU
        memory.  Larger batches improve throughput but require more memory.

    Returns
    -------
    list of numpy.ndarray
        The embeddings as a list of NumPy arrays.  Each array has shape
        (embedding_dim,), where ``embedding_dim`` is determined by the
        underlying model (typically 768 for Granite embeddings【449332267749196†L57-L83】).
    """
    model = _load_embed_model()
    # SentenceTransformer returns a list of numpy arrays when convert_to_tensor=False
    logger.debug("Embedding %d texts", len(list(texts)))
    embeddings = model.encode(list(texts), batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return [np.asarray(e, dtype=np.float32) for e in embeddings]
