"""
Persistent hybrid retrieval index.

This module defines the :class:`HybridIndex` class which maintains a
combined dense and lexical index of document chunks. Dense vectors are
stored using FAISS. Lexical features are stored using a TF-IDF matrix.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy import sparse  # robust sparse persistence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from ingestion.chunker import Chunk
from ingestion.embedder import EMBEDDINGS_AVAILABLE, EMBEDDINGS_IMPORT_ERROR, embed_texts

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
        "Please reinstall with \"pip install 'numpy<2 faiss-cpu==1.8.0.post1'\"."
    )

if _FAISS_AVAILABLE:
    try:
        import faiss  # type: ignore
    except (ImportError, AttributeError) as exc:  # pragma: no cover
        _FAISS_AVAILABLE = False
        _FAISS_IMPORT_ERROR = (
            "Failed to import faiss. Ensure numpy<2 and faiss-cpu==1.8.0.post1."
        )
        logger.warning("FAISS unavailable: %s", exc)
else:  # pragma: no cover
    faiss = None  # type: ignore

if not _FAISS_AVAILABLE and _FAISS_IMPORT_ERROR:
    logger.warning("FAISS disabled: %s", _FAISS_IMPORT_ERROR)

if TYPE_CHECKING:  # pragma: no cover
    import faiss  # type: ignore  # noqa: F401


@dataclass
class RetrievalResult:
    """Container for a retrieved chunk and its similarity score."""
    chunk: Chunk
    score: float


class HybridIndex:
    """Hybrid dense/lexical index for retrieval."""
    def __init__(self, *, dimension: int = 768) -> None:
        self.dimension = dimension
        self.faiss_index: Optional["faiss.IndexFlatIP"] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[sparse.spmatrix] = None
        self.chunks: List[Chunk] = []

        # Try to auto-load if everything exists (safe on restarts)
        try:
            self.load_if_exists()
        except Exception as exc:
            logger.error("Auto-load failed; starting empty index: %s", exc)

        if _FAISS_AVAILABLE and EMBEDDINGS_AVAILABLE and self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
        if not _FAISS_AVAILABLE:
            logger.warning("FAISS disabled; dense retrieval falls back to lexical-only.")
        if _FAISS_AVAILABLE and not EMBEDDINGS_AVAILABLE:
            logger.warning(
                "Dense retrieval disabled; SentenceTransformer unavailable%s.",
                f": {EMBEDDINGS_IMPORT_ERROR}" if EMBEDDINGS_IMPORT_ERROR else "",
            )

    # -------------------- persistence helpers --------------------

    @staticmethod
    def _tfidf_matrix_path():
        """
        Persist the sparse TF-IDF matrix as NPZ.
        If config.TFIDF_INDEX_PATH already ends with .npz we use it,
        otherwise we store next to it with a .npz suffix.
        """
        p = config.TFIDF_INDEX_PATH
        return p if p.suffix.lower() == ".npz" else p.with_suffix(".npz")

    @staticmethod
    def _tfidf_vectorizer_path():
        # Keep compatibility with previous naming: <tfidf>.json
        return config.TFIDF_INDEX_PATH.with_suffix(".json")

    @staticmethod
    def _metadata_path():
        return config.FAISS_INDEX_PATH.with_suffix(".meta.json")

    def _save(self) -> None:
        """Persist FAISS, TF-IDF and metadata to disk."""
        try:
            config.ensure_directories()
        except Exception as e:
            logger.warning("ensure_directories failed (continuing): %s", e)

        # Save FAISS (when available)
        if _FAISS_AVAILABLE and self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(config.FAISS_INDEX_PATH))

        # Save TF-IDF sparse matrix
        if self.tfidf_matrix is not None:
            sparse.save_npz(self._tfidf_matrix_path(), self.tfidf_matrix, compressed=True)

        # Save vectorizer state
        if self.tfidf_vectorizer is not None and hasattr(self.tfidf_vectorizer, "vocabulary_"):
            vec_state = {
                "vocabulary_": self.tfidf_vectorizer.vocabulary_,
                "idf_": (
                    self.tfidf_vectorizer.idf_.tolist()
                    if hasattr(self.tfidf_vectorizer, "idf_")
                    else None
                ),
            }
            with open(self._tfidf_vectorizer_path(), "w", encoding="utf-8") as f:
                json.dump(vec_state, f)

        # Save metadata (chunks)
        with open(self._metadata_path(), "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f)

        logger.info(
            "Saved index: %d chunks (dense=%s, lexical=%s)",
            len(self.chunks),
            "yes" if (self.faiss_index is not None and (_FAISS_AVAILABLE)) else "no",
            "yes" if (self.tfidf_matrix is not None) else "no",
        )

    def load_if_exists(self) -> bool:
        """
        Load any existing on-disk artifacts (FAISS, TF-IDF, metadata) if present.
        Returns True if at least one component was loaded; False on first-run.
        Safe to call multiple times.
        """
        loaded_any = False

        # Load FAISS index if the file exists and FAISS available
        if _FAISS_AVAILABLE and config.FAISS_INDEX_PATH.exists():
            try:
                self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
                loaded_any = True
            except Exception as e:
                logger.warning("Failed to read FAISS index (%s). Recreating empty dense index.", e)
                self.faiss_index = faiss.IndexFlatIP(self.dimension)

        # Load TF-IDF sparse matrix and vectorizer state
        mat_path = self._tfidf_matrix_path()
        vec_path = self._tfidf_vectorizer_path()
        if mat_path.exists() and vec_path.exists():
            try:
                self.tfidf_matrix = sparse.load_npz(mat_path)
                with open(vec_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.tfidf_vectorizer = TfidfVectorizer()
                self.tfidf_vectorizer.vocabulary_ = state.get("vocabulary_", {})
                idf = state.get("idf_")
                if idf is not None:
                    self.tfidf_vectorizer.idf_ = np.array(idf, dtype=float)  # type: ignore[attr-defined]
                    # force lazy rebuild of the internal diagonal
                    self.tfidf_vectorizer._tfidf._idf_diag = None
                loaded_any = True
            except Exception as e:
                logger.warning("Failed to load TF-IDF artifacts (%s). Rebuilding on next save.", e)
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None

        # Load metadata (chunks)
        meta_path = self._metadata_path()
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata_list = json.load(f)

                self.chunks = []
                for m in metadata_list:
                    # Backward compatibility:
                    # - use empty dict if 'metadata' missing
                    # - if an old dump had 'source' at top level, fold it into metadata
                    md = m.get("metadata") or ({"source": m.get("source")} if m.get("source") else {})
                    self.chunks.append(
                        Chunk(
                            id=m.get("id"),
                            document_id=m.get("document_id", ""),
                            text=m.get("text", ""),
                            token_count=m.get("token_count", 0),
                            start_paragraph=m.get("start_paragraph", 0),
                            end_paragraph=m.get("end_paragraph", 0),
                            summary=m.get("summary"),
                            metadata=md,  # <-- required by Chunk __init__
                        )
                    )
                loaded_any = True
            except Exception as e:
                logger.warning("Failed to load metadata (%s). Starting with empty chunk list.", e)
                self.chunks = []

        if loaded_any:
            logger.info(
                "Loaded index from disk (chunks=%d, dense=%s, lexical=%s).",
                len(self.chunks),
                "yes" if (self.faiss_index is not None and (_FAISS_AVAILABLE)) else "no",
                "yes" if (self.tfidf_matrix is not None) else "no",
            )
        else:
            logger.info("No on-disk index found yet (first run).")

        return loaded_any

    # -------------------- building / querying --------------------

    def _rebuild_tfidf(self) -> None:
        """Rebuild the TF-IDF matrix and vectoriser from current chunks."""
        texts = [c.text for c in self.chunks]
        if texts:
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = None

    def add_chunks(self, chunks: Iterable[Chunk], *, persist: bool = True) -> None:
        """
        Add new chunks to the index and optionally persist it to disk.
        """
        new_chunks = list(chunks)
        if not new_chunks:
            return

        # Dense embeddings
        if _FAISS_AVAILABLE and EMBEDDINGS_AVAILABLE:
            texts = [c.text for c in new_chunks]
            vectors = embed_texts(texts)
            # L2-normalize for cosine via inner product
            mat = []
            for v in vectors:
                n = np.linalg.norm(v)
                mat.append((v / n) if n > 0 else v)
            matrix = np.vstack(mat).astype("float32")
            if self.faiss_index is None:
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.faiss_index.add(matrix)
        else:
            if not _FAISS_AVAILABLE:
                logger.debug("Skipping dense embeddings: FAISS unavailable.")
            elif not EMBEDDINGS_AVAILABLE:
                logger.debug("Skipping dense embeddings: embedding model unavailable.")

        self.chunks.extend(new_chunks)
        self._rebuild_tfidf()

        if persist:
            self._save()

    def search(self, query: str, *, top_k: int = config.TOP_K, alpha: float = config.ALPHA) -> List[Tuple[int, float]]:
        """
        Search the index for a given query and return ranked indices.
        Returns list of (chunk_idx, score) sorted by score desc.
        """
        if not self.chunks:
            return []

        dense_indices: np.ndarray = np.array([], dtype=int)
        dense_scores: np.ndarray = np.array([], dtype=float)

        dense_ready = (
            _FAISS_AVAILABLE
            and EMBEDDINGS_AVAILABLE
            and self.faiss_index is not None
            and getattr(self.faiss_index, "ntotal", 0) > 0
        )
        dense_weight = alpha if dense_ready else 0.0

        if dense_weight > 0:
            q_vec = embed_texts([query])[0]
            n = np.linalg.norm(q_vec)
            if n > 0:
                q_vec = q_vec / n
            D, I = self.faiss_index.search(np.array([q_vec], dtype="float32"), top_k)
            dense_scores = D[0]
            dense_indices = I[0]

        lexical_scores = None
        lexical_indices = None
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            q_tfidf = self.tfidf_vectorizer.transform([query])
            sim = cosine_similarity(q_tfidf, self.tfidf_matrix).flatten()
            lex_order = np.argsort(sim)[::-1][:top_k]
            lexical_scores = sim[lex_order]
            lexical_indices = lex_order

        combined: Dict[int, float] = {}

        for idx, score in zip(dense_indices, dense_scores):
            combined[idx] = combined.get(idx, 0.0) + float(dense_weight * score)

        lexical_weight = 1.0 - dense_weight
        if lexical_indices is not None and lexical_scores is not None:
            for idx, score in zip(lexical_indices, lexical_scores):
                combined[idx] = combined.get(idx, 0.0) + float(lexical_weight * score)

        if not combined and lexical_indices is not None and lexical_scores is not None:
            return list(zip(lexical_indices.tolist(), lexical_scores.tolist()))[:top_k]

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_results(self, ranked_indices: List[Tuple[int, float]]) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        for idx, score in ranked_indices:
            if 0 <= idx < len(self.chunks):
                results.append(RetrievalResult(chunk=self.chunks[idx], score=score))
        return results


# Global singleton used by the app
index = HybridIndex()
