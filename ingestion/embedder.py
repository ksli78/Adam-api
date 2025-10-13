# ingestion/embedder.py
"""
Embedding utilities.

This module wraps IBM's Granite embedding model to produce dense vectors for text chunks.
We use Hugging Face Transformers via SentenceTransformer for batching and GPU support.
The default model generates 768-dimensional embeddings and supports up to 8k token inputs.
"""
from __future__ import annotations
import logging
from typing import Iterable, List
import numpy as np
import config
import os

try:
    # Use SentenceTransformer for IBM Granite Embedding model
    from sentence_transformers import SentenceTransformer
    import torch
except Exception as exc:
    SentenceTransformer = None
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = str(exc)
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

# Determine embedding model name (allow override via env variable for flexibility)
_EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL", config.EMBED_MODEL_NAME)

EMBEDDINGS_AVAILABLE: bool = SentenceTransformer is not None
EMBEDDINGS_IMPORT_ERROR: str | None = _SENTENCE_TRANSFORMERS_IMPORT_ERROR

if not EMBEDDINGS_AVAILABLE and EMBEDDINGS_IMPORT_ERROR:
    logger.warning(
        "SentenceTransformer unavailable: %s. Dense embeddings disabled; install the embedding dependencies.",
        EMBEDDINGS_IMPORT_ERROR,
    )

_EMBED_MODEL: SentenceTransformer | None = None

def _cuda_ok() -> bool:
    """Check if CUDA is available for PyTorch (for GPU acceleration)."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def _load_embed_model() -> SentenceTransformer:
    """Load (or reuse) the embedding model for generating vector embeddings."""
    global _EMBED_MODEL
    if not EMBEDDINGS_AVAILABLE:
        raise RuntimeError("Embedding model not available. Ensure sentence_transformers and torch are installed.")
    if _EMBED_MODEL is None:
        device = "cuda" if _cuda_ok() else "cpu"
        logger.info("Loading embedding model %s on %s", _EMBED_MODEL_NAME, device)
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME, device=device)
    return _EMBED_MODEL

def embed_texts(texts: Iterable[str], *, batch_size: int = 32) -> List[np.ndarray]:
    """
    Compute dense embeddings for a list of text chunks using the IBM Granite model.
    
    Parameters
    ----------
    texts : Iterable[str]
        The chunk texts to embed.
    batch_size : int
        Batch size for embedding computation (tune based on memory).
    
    Returns
    -------
    List[np.ndarray]
        List of embedding vectors (each a 768-dim float32 NumPy array).
    """
    items = list(texts)
    if not items:
        return []
    model = _load_embed_model()
    logger.debug("Generating embeddings for %d chunks", len(items))
    # Encode all texts into vectors (automatically uses GPU if available)
    embeddings = model.encode(items, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    # Ensure embeddings are float32 NumPy arrays
    return [np.asarray(vec, dtype=np.float32) for vec in embeddings]
