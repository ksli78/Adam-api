"""
Persistent hybrid retrieval index.

This module defines the :class:`HybridIndex` class which maintains a
combined dense and lexical index of document chunks.  Dense vectors are
stored using FAISS, a high‑performance nearest neighbour library.  Lexical
features are stored using a TF‑IDF matrix built with scikit‑learn.  A
configurable weight balances the two scores during search.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

_NUMPY_VERSION = np.__version__
_MATCH = re.match(r"(\d+)\.(\d+)", _NUMPY_VERSION)
_FAISS_AVAILABLE = True
_FAISS_IMPORT_ERROR: Optional[str] = None

if _MATCH and int(_MATCH.group(1)) >= 2:
    _FAISS_AVAILABLE = False
    _FAISS_IMPORT_ERROR = (
        "Faiss Python bindings bundled with this project require NumPy < 2. "
        f"Detected numpy=={_NUMPY_VERSION}. "
        "Please reinstall with \"pip install 'numpy<2 faiss-cpu==1.8.0.post1'\" "
        "or use the provided requirements.txt."
    )

if _FAISS_AVAILABLE:
    try:
        import faiss  # type: ignore
    except (ImportError, AttributeError) as exc:  # pragma: no cover - import guard
        _FAISS_AVAILABLE = False
        _FAISS_IMPORT_ERROR = (
            "Failed to import faiss. Ensure numpy<2 and faiss-cpu==1.8.0.post1 "
            "(see requirements.txt/Dockerfile)."
        )
        logger.warning("FAISS unavailable: %s", exc)
else:  # pragma: no cover - defensive branch
    faiss = None  # type: ignore

if not _FAISS_AVAILABLE and _FAISS_IMPORT_ERROR:
    logger.warning("FAISS disabled: %s", _FAISS_IMPORT_ERROR)

if TYPE_CHECKING:  # pragma: no cover - type checking aid
    import faiss  # type: ignore  # noqa: F401

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from ingestion.chunker import Chunk
from ingestion.embedder import EMBEDDINGS_AVAILABLE, EMBEDDINGS_IMPORT_ERROR, embed_texts


@dataclass
class RetrievalResult:
    """Container for a retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float


class HybridIndex:
    """Hybrid dense/lexical index for retrieval.

    The index consists of a FAISS vector index for dense retrieval and a
    TF‑IDF matrix for lexical retrieval.  Metadata for each chunk is stored
    alongside to facilitate result reconstruction.
    """

    def __init__(self, *, dimension: int = 768) -> None:
        self.dimension = dimension
        # Dense index
        self.faiss_index: Optional["faiss.IndexFlatIP"] = None
        # Lexical index
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        # Metadata mapping index positions to chunks
        self.chunks: List[Chunk] = []

        # Attempt to load existing index from disk
        if config.FAISS_INDEX_PATH.exists() and config.TFIDF_INDEX_PATH.exists():
            try:
                self._load()
            except Exception as exc:
                logger.error("Failed to load existing index: %s", exc)
                # Fall back to new empty index
        if _FAISS_AVAILABLE and EMBEDDINGS_AVAILABLE and self.faiss_index is None:
            # Create a new index for inner product similarity (cosine after normalisation)
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
        if not _FAISS_AVAILABLE:
            logger.warning(
                "FAISS features disabled; dense retrieval will fall back to lexical-only searches."
            )
        if _FAISS_AVAILABLE and not EMBEDDINGS_AVAILABLE:
            logger.warning(
                "Dense retrieval disabled because SentenceTransformer could not be imported%s.",
                f": {EMBEDDINGS_IMPORT_ERROR}" if EMBEDDINGS_IMPORT_ERROR else "",
            )

    def _save(self) -> None:
        """Persist the index and metadata to disk."""
        if _FAISS_AVAILABLE and self.faiss_index is not None:
            # Save FAISS index
            faiss.write_index(self.faiss_index, str(config.FAISS_INDEX_PATH))
        # Save TF‑IDF matrix and vectorizer
        with open(config.TFIDF_INDEX_PATH, "wb") as f:
            np.savez(f, data=self.tfidf_matrix)
        with open(config.TFIDF_INDEX_PATH.with_suffix(".json"), "w", encoding="utf-8") as f:
            # Save vectoriser vocabulary and idf
            vectorizer_state = {
                "vocabulary_": self.tfidf_vectorizer.vocabulary_,
                "idf_": self.tfidf_vectorizer.idf_.tolist(),
            }
            json.dump(vectorizer_state, f)
        # Save metadata
        metadata_path = config.FAISS_INDEX_PATH.with_suffix(".meta.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f)
        logger.info("Saved index with %d chunks", len(self.chunks))

    def _load(self) -> None:
        """Load the index and metadata from disk."""
        # Load FAISS index if available
        if _FAISS_AVAILABLE:
            self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        else:
            logger.debug("Skipping FAISS index load because FAISS is unavailable.")
        # Load TF‑IDF matrix
        with np.load(config.TFIDF_INDEX_PATH, allow_pickle=True) as data:
            self.tfidf_matrix = data["data"]
        # Load vectoriser state
        state_file = config.TFIDF_INDEX_PATH.with_suffix(".json")
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer.vocabulary_ = state["vocabulary_"]
        self.tfidf_vectorizer.idf_ = np.array(state["idf_"])  # type: ignore
        self.tfidf_vectorizer._tfidf._idf_diag = None  # lazy update
        # Load metadata
        metadata_path = config.FAISS_INDEX_PATH.with_suffix(".meta.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        self.chunks = []
        for meta in metadata_list:
            chunk = Chunk(
                id=meta["id"],
                document_id=meta.get("document_id", ""),
                text=meta["text"],
                token_count=meta.get("token_count", 0),
                start_paragraph=meta.get("start_paragraph", 0),
                end_paragraph=meta.get("end_paragraph", 0),
                summary=meta.get("summary"),
            )
            self.chunks.append(chunk)
        logger.info("Loaded index with %d chunks", len(self.chunks))

    def _rebuild_tfidf(self) -> None:
        """Rebuild the TF‑IDF matrix and vectoriser from current chunks."""
        self.tfidf_vectorizer = TfidfVectorizer()
        texts = [c.text for c in self.chunks]
        if texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            self.tfidf_matrix = None

    def add_chunks(self, chunks: Iterable[Chunk], *, persist: bool = True) -> None:
        """Add new chunks to the index and optionally persist it to disk.

        Parameters
        ----------
        chunks:
            Iterable of chunks to index.  Each chunk should have its
            embedding computed before being added.
        persist:
            If ``True``, save the index to disk after adding the chunks.
        """
        new_chunks = list(chunks)
        if not new_chunks:
            return
        # Compute embeddings for new chunks when dense retrieval is available
        if _FAISS_AVAILABLE and EMBEDDINGS_AVAILABLE:
            texts = [c.text for c in new_chunks]
            vectors = embed_texts(texts)
            # Normalise vectors to unit length for cosine similarity
            vectors = [v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v for v in vectors]
            matrix = np.vstack(vectors).astype("float32")
            # Add to FAISS index
            if self.faiss_index is None:
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.faiss_index.add(matrix)
        else:
            if not _FAISS_AVAILABLE:
                logger.debug("Skipping dense embedding generation because FAISS is unavailable.")
            elif not EMBEDDINGS_AVAILABLE:
                logger.debug(
                    "Skipping dense embedding generation because SentenceTransformer dependencies are unavailable."
                )
        # Append new chunks to metadata list
        self.chunks.extend(new_chunks)
        # Rebuild TF‑IDF index from scratch (small overhead compared to ingesting all docs)
        self._rebuild_tfidf()
        if persist:
            self._save()

    def search(self, query: str, *, top_k: int = config.TOP_K, alpha: float = config.ALPHA) -> List[Tuple[int, float]]:
        """Search the index for a given query and return ranked indices.

        The result is a list of tuples ``(chunk_idx, score)`` where
        ``chunk_idx`` indexes into :attr:`chunks`.

        Parameters
        ----------
        query:
            The natural language query to search for.
        top_k:
            Number of top results to return.
        alpha:
            Weighting between dense similarity and lexical similarity【449332267749196†L65-L70】.

        Returns
        -------
        list of (int, float)
            List of index positions and combined scores sorted in descending
            order.
        """
        if not self.chunks:
            return []
        dense_indices: np.ndarray = np.array([], dtype=int)
        dense_scores: np.ndarray = np.array([], dtype=float)
        dense_ready = (
            _FAISS_AVAILABLE
            and EMBEDDINGS_AVAILABLE
            and self.faiss_index is not None
            and self.faiss_index.ntotal > 0
        )
        dense_weight = alpha if dense_ready else 0.0
        if dense_weight > 0:
            # Compute dense embedding for query
            query_vec = embed_texts([query])[0]
            query_norm = np.linalg.norm(query_vec)
            if query_norm > 0:
                query_vec = query_vec / query_norm
            # Dense search
            D, I = self.faiss_index.search(np.array([query_vec], dtype="float32"), top_k)
            dense_scores = D[0]
            dense_indices = I[0]
        # Lexical search
        lexical_scores = None
        lexical_indices = None
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            q_tfidf = self.tfidf_vectorizer.transform([query])
            # Compute cosine similarity between query and documents
            sim = cosine_similarity(q_tfidf, self.tfidf_matrix).flatten()
            # Get top_k lexical indices
            lex_order = np.argsort(sim)[::-1][: top_k]
            lexical_scores = sim[lex_order]
            lexical_indices = lex_order
        # Combine scores
        combined: Dict[int, float] = {}
        # Add dense scores
        for idx, score in zip(dense_indices, dense_scores):
            combined[idx] = combined.get(idx, 0.0) + float(dense_weight * score)
        # Add lexical scores
        lexical_weight = 1.0 - dense_weight
        if lexical_indices is not None and lexical_scores is not None:
            for idx, score in zip(lexical_indices, lexical_scores):
                combined[idx] = combined.get(idx, 0.0) + float(lexical_weight * score)
        if not combined and lexical_indices is not None and lexical_scores is not None:
            # Dense retrieval unavailable; return lexical results directly
            return list(zip(lexical_indices.tolist(), lexical_scores.tolist()))[:top_k]
        # Sort combined scores
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_results(self, ranked_indices: List[Tuple[int, float]]) -> List[RetrievalResult]:
        """Convert ranked index positions into :class:`RetrievalResult` objects."""
        results: List[RetrievalResult] = []
        for idx, score in ranked_indices:
            if 0 <= idx < len(self.chunks):
                results.append(RetrievalResult(chunk=self.chunks[idx], score=score))
        return results
