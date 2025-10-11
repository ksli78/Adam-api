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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from ingestion.chunker import Chunk
from ingestion.embedder import embed_texts

logger = logging.getLogger(__name__)


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
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
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
        if self.faiss_index is None:
            # Create a new index for inner product similarity (cosine after normalisation)
            self.faiss_index = faiss.IndexFlatIP(self.dimension)

    def _save(self) -> None:
        """Persist the index and metadata to disk."""
        assert self.faiss_index is not None
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
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
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
        # Compute embeddings for new chunks
        texts = [c.text for c in new_chunks]
        vectors = embed_texts(texts)
        # Normalise vectors to unit length for cosine similarity
        vectors = [v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v for v in vectors]
        matrix = np.vstack(vectors).astype("float32")
        # Add to FAISS index
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(matrix)
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
        # Compute dense embedding for query
        query_vec = embed_texts([query])[0]
        query_vec = query_vec / np.linalg.norm(query_vec) if np.linalg.norm(query_vec) > 0 else query_vec
        # Dense search
        assert self.faiss_index is not None
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
            combined[idx] = combined.get(idx, 0.0) + float(alpha * score)
        # Add lexical scores
        if lexical_indices is not None and lexical_scores is not None:
            for idx, score in zip(lexical_indices, lexical_scores):
                combined[idx] = combined.get(idx, 0.0) + float((1 - alpha) * score)
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
