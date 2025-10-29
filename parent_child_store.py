"""
Parent-Child Document Store

Manages dual ChromaDB collections for optimal RAG:
- Child chunks: Small, precise chunks for retrieval
- Parent chunks: Large sections for LLM context

Retrieval strategy:
1. Embed query and search child collection (precise)
2. Get matching child chunks
3. Expand to parent chunks (rich context)
4. Deduplicate and rerank parents
5. Pass parent content to LLM
"""

import logging
import os
import uuid
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi

from semantic_chunker import Chunk

logger = logging.getLogger(__name__)

# Default data directory (platform-specific)
DEFAULT_CHROMA_DIR = "D:/data/airgapped_rag/chromadb_advanced" if os.name == 'nt' else "/data/airgapped_rag/chromadb_advanced"

# Import feedback store (lazy import to avoid circular dependencies)
_feedback_store = None

def get_feedback_store_lazy():
    """Lazy load feedback store to avoid circular imports."""
    global _feedback_store
    if _feedback_store is None:
        try:
            from feedback_store import get_feedback_store
            _feedback_store = get_feedback_store()
        except Exception as e:
            logger.warning(f"Could not load feedback store: {e}")
            _feedback_store = None
    return _feedback_store

# English stop words for BM25 filtering
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'which',
    'how', 'this', 'these', 'those', 'can', 'could', 'should', 'would',
    'do', 'does', 'did', 'have', 'had', 'been', 'being', 'my', 'your',
    'their', 'our', 'his', 'her'
}


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenize text for BM25 search with stop word filtering and punctuation stripping.

    Removes common English stop words and strips punctuation to improve keyword matching accuracy.
    """
    import string
    tokens = text.lower().split()
    # Strip punctuation from each token
    tokens = [token.strip(string.punctuation) for token in tokens]
    # Filter out stop words, empty tokens, and keep only meaningful terms
    return [token for token in tokens if token and token not in STOP_WORDS and len(token) > 1]


class ParentChildDocumentStore:
    """
    Manages parent and child chunk collections in ChromaDB.

    Uses sentence-transformers for local embedding generation.
    """

    def __init__(
        self,
        persist_directory: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        child_collection_name: str = "child_chunks",
        parent_collection_name: str = "parent_chunks"
    ):
        """
        Initialize the parent-child document store.

        Args:
            persist_directory: Directory for ChromaDB persistence (defaults to platform-specific path)
            embedding_model: Sentence-transformers model for embeddings
            child_collection_name: Name for child chunks collection
            parent_collection_name: Name for parent chunks collection
        """
        # Use default directory if not specified
        if persist_directory is None:
            persist_directory = DEFAULT_CHROMA_DIR

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"ParentChildDocumentStore using ChromaDB at: {self.persist_directory}")

        self.child_collection_name = child_collection_name
        self.parent_collection_name = parent_collection_name

        # Initialize sentence-transformers for embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collections
        self.child_collection = self.client.get_or_create_collection(
            name=child_collection_name,
            metadata={"description": "Small chunks for precise retrieval"}
        )

        self.parent_collection = self.client.get_or_create_collection(
            name=parent_collection_name,
            metadata={"description": "Large chunks for LLM context"}
        )

        logger.info(
            f"ParentChildDocumentStore initialized: "
            f"persist={persist_directory}, "
            f"children={self.child_collection.count()}, "
            f"parents={self.parent_collection.count()}"
        )

    def add_document_chunks(
        self,
        parent_chunks: List[Chunk],
        child_chunks: List[Chunk],
        document_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Add parent and child chunks for a document.

        Args:
            parent_chunks: List of parent chunk objects
            child_chunks: List of child chunk objects
            document_metadata: Optional document-level metadata

        Returns:
            Dict with statistics about added chunks
        """
        document_id = parent_chunks[0].metadata.get("document_id") if parent_chunks else str(uuid.uuid4())

        logger.info(
            f"Adding document {document_id}: "
            f"{len(parent_chunks)} parents, {len(child_chunks)} children"
        )

        # Add parent chunks (no embeddings needed, used only for retrieval expansion)
        parent_ids = []
        parent_documents = []
        parent_metadatas = []

        for parent_chunk in parent_chunks:
            parent_ids.append(parent_chunk.id)
            parent_documents.append(parent_chunk.text)

            # Merge document metadata with chunk metadata
            metadata = {**(document_metadata or {}), **parent_chunk.metadata}
            # Convert non-string values to strings for ChromaDB
            metadata = self._sanitize_metadata(metadata)
            parent_metadatas.append(metadata)

        self.parent_collection.add(
            ids=parent_ids,
            documents=parent_documents,
            metadatas=parent_metadatas
        )

        # Add child chunks (with embeddings for retrieval)
        child_ids = []
        child_documents = []
        child_embeddings = []
        child_metadatas = []

        # Generate embeddings in batch for efficiency
        child_texts = [chunk.text for chunk in child_chunks]
        logger.debug(f"Generating embeddings for {len(child_texts)} child chunks...")
        embeddings = self.embedding_model.encode(
            child_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        for chunk, embedding in zip(child_chunks, embeddings):
            child_ids.append(chunk.id)
            child_documents.append(chunk.text)
            child_embeddings.append(embedding.tolist())

            # Merge metadata
            metadata = {**(document_metadata or {}), **chunk.metadata}
            metadata = self._sanitize_metadata(metadata)
            child_metadatas.append(metadata)

        self.child_collection.add(
            ids=child_ids,
            documents=child_documents,
            embeddings=child_embeddings,
            metadatas=child_metadatas
        )

        stats = {
            "document_id": document_id,
            "parent_chunks_added": len(parent_chunks),
            "child_chunks_added": len(child_chunks),
            "total_child_chunks": self.child_collection.count(),
            "total_parent_chunks": self.parent_collection.count()
        }

        logger.info(f"Document added successfully: {stats}")
        return stats

    def retrieve_with_parent_expansion(
        self,
        query: str,
        top_k: int = 10,
        expand_to_parents: bool = True,
        parent_limit: int = 3,
        metadata_filter: Dict[str, Any] = None,
        use_hybrid: bool = True,
        bm25_weight: float = 0.5,
        use_feedback_weighting: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieve child chunks and optionally expand to parent chunks.

        Strategy:
        1. Run hybrid search (BM25 + Semantic) on child chunks
        2. Fuse scores with weighted combination
        3. Apply feedback weighting to boost/demote chunks based on user feedback
        4. If expand_to_parents, get corresponding parent chunks
        5. Deduplicate parents
        6. Return both child and parent chunks

        Args:
            query: Query text
            top_k: Number of child chunks to retrieve
            expand_to_parents: Whether to expand to parent chunks
            parent_limit: Maximum number of unique parent chunks to return
            metadata_filter: Optional metadata filter for search
            use_hybrid: Whether to use hybrid (BM25 + semantic) search
            bm25_weight: Weight for BM25 scores (0.0-1.0), semantic gets (1-bm25_weight)
            use_feedback_weighting: Whether to apply feedback-based score adjustments

        Returns:
            Tuple of (child_results, parent_results)
        """
        logger.debug(f"Retrieving for query: {query[:100]}... (hybrid={use_hybrid})")

        if use_hybrid:
            child_chunks = self._hybrid_search(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
                bm25_weight=bm25_weight,
                use_feedback_weighting=use_feedback_weighting
            )
        else:
            # Pure semantic search (original method)
            child_chunks = self._semantic_search(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter
            )

        logger.info(f"Retrieved {len(child_chunks)} child chunks")

        # Expand to parents if requested
        parent_chunks = []
        if expand_to_parents and child_chunks:
            parent_chunks = self._expand_to_parents(child_chunks, parent_limit)

        return child_chunks, parent_chunks

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Pure semantic search using embeddings.

        Args:
            query: Query text
            top_k: Number of results
            metadata_filter: Optional metadata filter

        Returns:
            List of child chunk results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()

        # Search child collection
        where_filter = metadata_filter if metadata_filter else None

        child_results = self.child_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format child results
        child_chunks = []
        for i in range(len(child_results['ids'][0])):
            child_chunks.append({
                "id": child_results['ids'][0][i],
                "text": child_results['documents'][0][i],
                "metadata": child_results['metadatas'][0][i],
                "distance": child_results['distances'][0][i],
                "score": 1.0 / (1.0 + child_results['distances'][0][i]),
                "retrieval_method": "semantic"
            })

        return child_chunks

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: Dict[str, Any] = None,
        bm25_weight: float = 0.5,
        use_feedback_weighting: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining BM25 (keyword) and semantic (embedding) search.

        Uses weighted score fusion:
        final_score = (bm25_weight * bm25_score) + ((1 - bm25_weight) * semantic_score)

        Optionally applies feedback weighting:
        feedback_adjusted_score = final_score * (1 + feedback_weight * quality_score)

        Relevance filtering:
        - Requires BM25 score >= 0.95 (strong keyword match)
        - Requires semantic score >= 0.2 (minimum semantic relevance)
        - Both thresholds must be met to prevent keyword-only matches

        Args:
            query: Query text
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            bm25_weight: Weight for BM25 scores (0.0-1.0)
            use_feedback_weighting: Apply feedback-based score adjustments

        Returns:
            List of child chunk results sorted by fused score (empty if no docs pass thresholds)
        """
        logger.debug(f"Running hybrid search: BM25_weight={bm25_weight}")

        # Step 1: Get all child chunks for BM25
        # (We need all chunks to build BM25 index - could cache this for performance)
        all_chunks = self.child_collection.get(
            where=metadata_filter if metadata_filter else None,
            include=["documents", "metadatas"]
        )

        if not all_chunks['ids']:
            logger.warning("No chunks found in collection")
            return []

        # Step 2: Build BM25 index (with stop word filtering)
        tokenized_corpus = [tokenize_for_bm25(doc) for doc in all_chunks['documents']]
        bm25 = BM25Okapi(tokenized_corpus)

        # Step 3: Get BM25 scores (with stop word filtering)
        tokenized_query = tokenize_for_bm25(query)
        logger.info(f"BM25 query tokens (stop words filtered): {tokenized_query}")
        bm25_scores = bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to 0-1 range
        if max(bm25_scores) > 0:
            bm25_scores_normalized = bm25_scores / max(bm25_scores)
        else:
            bm25_scores_normalized = bm25_scores

        # Step 4: Get semantic search scores (fetch more for fusion)
        semantic_results = self._semantic_search(
            query=query,
            top_k=min(top_k * 3, len(all_chunks['ids'])),  # Get 3x for better fusion
            metadata_filter=metadata_filter
        )

        # Create semantic scores lookup
        semantic_scores = {
            result['id']: result['score']
            for result in semantic_results
        }

        # Step 5: Fuse scores and filter out weak matches
        # In hybrid mode, require both strong keyword AND semantic relevance
        # This prevents documents with keyword matches but no semantic relevance
        MIN_BM25_THRESHOLD = 0.80  # Lowered to 80% to capture more relevant chunks (e.g. 0.848, 0.820) from multiple PTO policy sections
        MIN_SEMANTIC_THRESHOLD = 0.2  # Require minimum semantic relevance

        fused_results = []

        # Log top BM25 scores for debugging
        if len(bm25_scores_normalized) > 0:
            top_bm25_indices = sorted(range(len(bm25_scores_normalized)),
                                     key=lambda i: bm25_scores_normalized[i],
                                     reverse=True)[:5]
            logger.info(f"Top 5 BM25 scores: {[f'{bm25_scores_normalized[i]:.3f}' for i in top_bm25_indices]}")

        # Track chunks that pass BM25 threshold
        bm25_passed = []

        for i, chunk_id in enumerate(all_chunks['ids']):
            # BM25 score (already normalized 0-1)
            bm25_score = float(bm25_scores_normalized[i])

            # Skip chunks with weak keyword relevance in hybrid mode
            # With 0.95 threshold, only chunks with nearly perfect keyword matches pass
            # This filters out documents scoring 0.937, 0.469, etc. (only common word matches)
            if bm25_weight > 0 and bm25_score < MIN_BM25_THRESHOLD:
                continue

            # Semantic score (0 if not in semantic results)
            semantic_score = semantic_scores.get(chunk_id, 0.0)

            # Log all chunks that passed BM25 threshold
            bm25_passed.append({
                'chunk_id': chunk_id[:12],
                'bm25': bm25_score,
                'semantic': semantic_score,
                'doc_title': all_chunks['metadatas'][i].get('document_title', 'unknown')[:30]
            })

        if bm25_passed:
            logger.info(f"Chunks passing BM25 threshold (>={MIN_BM25_THRESHOLD}): {len(bm25_passed)}")
            for chunk_info in bm25_passed[:10]:  # Show first 10
                logger.info(f"  {chunk_info['chunk_id']}... BM25={chunk_info['bm25']:.3f} Semantic={chunk_info['semantic']:.3f} Doc={chunk_info['doc_title']}")

        # Now apply semantic threshold
        for i, chunk_id in enumerate(all_chunks['ids']):
            bm25_score = float(bm25_scores_normalized[i])
            if bm25_weight > 0 and bm25_score < MIN_BM25_THRESHOLD:
                continue

            semantic_score = semantic_scores.get(chunk_id, 0.0)

            # Skip chunks with very low semantic relevance, but with an important exception:
            # Only apply semantic threshold to chunks with moderate BM25 scores.
            # If BM25 passes threshold (>=0.85), trust it even with low semantic score.
            # This handles cases where relevant chunks aren't in top-30 semantic results.
            # Example: PTO policy chunks may have BM25=0.855 but semantic=0.0 if not in top-30.
            if semantic_score < MIN_SEMANTIC_THRESHOLD and bm25_score < MIN_BM25_THRESHOLD:
                logger.info(
                    f"Filtered out chunk {chunk_id[:8]}... - "
                    f"low semantic relevance (BM25={bm25_score:.3f}, semantic={semantic_score:.3f})"
                )
                continue

            # Weighted fusion
            fused_score = (bm25_weight * bm25_score) + ((1 - bm25_weight) * semantic_score)

            fused_results.append({
                "id": chunk_id,
                "text": all_chunks['documents'][i],
                "metadata": all_chunks['metadatas'][i],
                "score": fused_score,
                "bm25_score": bm25_score,
                "semantic_score": semantic_score,
                "retrieval_method": "hybrid"
            })

        # Step 6: Apply feedback weighting if enabled
        if use_feedback_weighting and fused_results:
            feedback_store = get_feedback_store_lazy()
            if feedback_store:
                # Get feedback scores for all chunks
                chunk_ids = [r['id'] for r in fused_results]
                feedback_scores = feedback_store.get_chunk_quality_scores(chunk_ids)

                # Apply feedback weighting
                # quality_score ranges from -1.0 (all bad) to +1.0 (all good)
                # We use a moderate weight so feedback doesn't override relevance
                FEEDBACK_WEIGHT = 0.15  # 15% adjustment based on feedback

                for result in fused_results:
                    chunk_id = result['id']
                    quality_score = feedback_scores.get(chunk_id, 0.0)

                    if quality_score != 0.0:
                        # Adjust score: positive feedback boosts, negative demotes
                        original_score = result['score']
                        adjustment = FEEDBACK_WEIGHT * quality_score
                        adjusted_score = original_score * (1 + adjustment)

                        result['score'] = max(0.0, adjusted_score)  # Keep >= 0
                        result['feedback_quality'] = quality_score
                        result['feedback_adjustment'] = adjustment

                        logger.debug(
                            f"Chunk {chunk_id[:8]}... feedback adjusted: "
                            f"{original_score:.3f} -> {adjusted_score:.3f} "
                            f"(quality: {quality_score:.2f})"
                        )

        # Step 7: Sort by (possibly feedback-adjusted) score and return top_k
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = fused_results[:top_k]

        if not top_results:
            logger.info("Hybrid search: no documents passed relevance thresholds")
            return []

        logger.info(
            f"Hybrid search: top result BM25={top_results[0]['bm25_score']:.3f}, "
            f"semantic={top_results[0]['semantic_score']:.3f}, "
            f"fused={top_results[0]['score']:.3f}"
        )

        # Debug: Log BM25 scores of top 5 results to diagnose filtering
        top5_bm25 = [r['bm25_score'] for r in top_results[:5]]
        logger.info(f"Top 5 BM25 scores: {top5_bm25}")

        return top_results

    def _expand_to_parents(
        self,
        child_chunks: List[Dict[str, Any]],
        parent_limit: int
    ) -> List[Dict[str, Any]]:
        """
        Expand child chunks to their parent chunks.

        Args:
            child_chunks: List of child chunk results
            parent_limit: Maximum number of unique parents to return

        Returns:
            List of parent chunk results (deduplicated)
        """
        # Get unique parent IDs in order of child chunk relevance
        # Child chunks are already sorted by score, so preserve that ordering
        parent_ids_ordered = []
        seen_parents = set()
        for child in child_chunks:
            parent_id = child['metadata'].get('parent_chunk_id')
            if parent_id and parent_id not in seen_parents:
                parent_ids_ordered.append(parent_id)
                seen_parents.add(parent_id)

        if not parent_ids_ordered:
            logger.warning("No parent IDs found in child chunks")
            return []

        logger.debug(f"Found {len(parent_ids_ordered)} unique parent chunks from {len(child_chunks)} child chunks")

        # Retrieve parents from collection (limit to top N by relevance)
        parent_ids_list = parent_ids_ordered[:parent_limit]  # Take most relevant parents

        try:
            parent_results = self.parent_collection.get(
                ids=parent_ids_list,
                include=["documents", "metadatas"]
            )

            # Format parent results
            parents = []
            for i in range(len(parent_results['ids'])):
                parent_metadata = parent_results['metadatas'][i]
                parents.append({
                    "id": parent_results['ids'][i],
                    "text": parent_results['documents'][i],
                    "metadata": parent_metadata,
                    "score": 1.0  # Parents don't have relevance scores initially
                })

                # Log which sections are being sent to LLM
                section_title = parent_metadata.get('section_title', 'Unknown')
                doc_title = parent_metadata.get('document_title', 'Unknown')
                logger.info(f"  Parent {i+1}: {doc_title} - Section: {section_title[:50]}")

            logger.info(f"Expanded to {len(parents)} parent chunks for LLM context")
            return parents

        except Exception as e:
            logger.error(f"Error retrieving parent chunks: {e}")
            return []

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete all chunks (parent and child) for a document.

        Args:
            document_id: Document ID to delete

        Returns:
            Dict with deletion statistics
        """
        logger.info(f"Deleting document: {document_id}")

        # Delete children
        child_results = self.child_collection.get(
            where={"document_id": document_id}
        )
        if child_results['ids']:
            self.child_collection.delete(ids=child_results['ids'])
            logger.info(f"Deleted {len(child_results['ids'])} child chunks")

        # Delete parents
        parent_results = self.parent_collection.get(
            where={"document_id": document_id}
        )
        if parent_results['ids']:
            self.parent_collection.delete(ids=parent_results['ids'])
            logger.info(f"Deleted {len(parent_results['ids'])} parent chunks")

        return {
            "document_id": document_id,
            "child_chunks_deleted": len(child_results['ids']),
            "parent_chunks_deleted": len(parent_results['ids'])
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the store.

        Returns:
            List of document metadata
        """
        # Get all parent chunks (one per section)
        parent_results = self.parent_collection.get(
            include=["metadatas"]
        )

        # Group by document ID
        documents = {}
        for metadata in parent_results['metadatas']:
            doc_id = metadata.get('document_id')
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "document_title": metadata.get('document_title', 'Unknown'),
                    "source_url": metadata.get('source_url', ''),
                    "document_type": metadata.get('document_type', 'unknown'),
                    "section_count": 0
                }
            if doc_id:
                documents[doc_id]["section_count"] += 1

        return list(documents.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_parent_chunks": self.parent_collection.count(),
            "total_child_chunks": self.child_collection.count(),
            "documents": len(self.list_documents()),
            "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
            "persist_directory": str(self.persist_directory)
        }

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Sanitize metadata for ChromaDB storage.

        ChromaDB requires all metadata values to be strings, ints, floats, or bools.
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists to comma-separated strings
                sanitized[key] = ", ".join(str(v) for v in value)
            else:
                # Convert other types to string
                sanitized[key] = str(value)

        return sanitized


# Singleton instance
_store_instance = None


def get_parent_child_store(**kwargs) -> ParentChildDocumentStore:
    """
    Get or create singleton ParentChildDocumentStore instance.

    Args:
        **kwargs: Arguments to pass to ParentChildDocumentStore constructor

    Returns:
        ParentChildDocumentStore instance
    """
    global _store_instance

    if _store_instance is None:
        _store_instance = ParentChildDocumentStore(**kwargs)

    return _store_instance


if __name__ == "__main__":
    # Test the store
    logging.basicConfig(level=logging.DEBUG)

    from semantic_chunker import Chunk, DocumentSection

    # Create test chunks
    parent1 = Chunk(
        id="parent_001",
        text="This is a test parent chunk with lots of content about PTO policies.",
        token_count=100,
        metadata={
            "document_id": "test_doc",
            "section_title": "PTO Policy"
        }
    )

    child1 = Chunk(
        id="child_001",
        text="Employees accrue 15 days of PTO per year.",
        token_count=20,
        parent_id="parent_001",
        metadata={
            "document_id": "test_doc",
            "parent_chunk_id": "parent_001",
            "section_title": "PTO Policy"
        }
    )

    child2 = Chunk(
        id="child_002",
        text="PTO requests must be submitted 2 weeks in advance.",
        token_count=20,
        parent_id="parent_001",
        metadata={
            "document_id": "test_doc",
            "parent_chunk_id": "parent_001",
            "section_title": "PTO Policy"
        }
    )

    # Initialize store
    store = get_parent_child_store(
        persist_directory="/tmp/test_chromadb"
    )

    # Add chunks
    stats = store.add_document_chunks(
        parent_chunks=[parent1],
        child_chunks=[child1, child2],
        document_metadata={"document_title": "Test Document"}
    )
    print(f"\n=== Added chunks: {stats} ===")

    # Retrieve
    query = "How much PTO do employees get?"
    child_results, parent_results = store.retrieve_with_parent_expansion(
        query=query,
        top_k=5,
        expand_to_parents=True
    )

    print(f"\n=== Retrieved {len(child_results)} children, {len(parent_results)} parents ===")
    for child in child_results:
        print(f"  Child: {child['text'][:100]}... (score: {child['score']:.3f})")

    for parent in parent_results:
        print(f"  Parent: {parent['text'][:100]}...")

    # Get statistics
    stats = store.get_statistics()
    print(f"\n=== Store Statistics ===")
    print(stats)
