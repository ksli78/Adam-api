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
import re
import uuid
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi

from semantic_chunker import Chunk

logger = logging.getLogger(__name__)

# Path to acronym expansions config file
ACRONYMS_CONFIG_PATH = Path(__file__).parent / "config" / "acronyms.json"

# English stop words for BM25 filtering
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'which',
    'how', 'this', 'these', 'those', 'can', 'could', 'should', 'would',
    'do', 'does', 'did', 'have', 'had', 'been', 'being', 'my', 'your',
    'their', 'our', 'his', 'her'
}

# Default acronym expansions (used if config file doesn't exist)
DEFAULT_ACRONYM_EXPANSIONS = {
    'pto': 'Paid Time Off (PTO)',
    'hr': 'Human Resources (HR)',
    'it': 'Information Technology (IT)',
    'cui': 'Controlled Unclassified Information (CUI)',
    'pii': 'Personally Identifiable Information (PII)',
}

# Global variable to hold loaded acronyms
_acronym_expansions = None


def load_acronym_expansions() -> Dict[str, str]:
    """
    Load acronym expansions from JSON config file.

    Returns:
        Dictionary of acronym -> expansion mappings
    """
    global _acronym_expansions

    try:
        if ACRONYMS_CONFIG_PATH.exists():
            with open(ACRONYMS_CONFIG_PATH, 'r') as f:
                _acronym_expansions = json.load(f)
            logger.info(f"Loaded {len(_acronym_expansions)} acronym expansions from {ACRONYMS_CONFIG_PATH}")
        else:
            logger.warning(f"Acronyms config file not found at {ACRONYMS_CONFIG_PATH}, using defaults")
            _acronym_expansions = DEFAULT_ACRONYM_EXPANSIONS.copy()
            # Create the config file with defaults
            save_acronym_expansions(_acronym_expansions)
    except Exception as e:
        logger.error(f"Error loading acronym expansions: {e}, using defaults")
        _acronym_expansions = DEFAULT_ACRONYM_EXPANSIONS.copy()

    return _acronym_expansions


def save_acronym_expansions(expansions: Dict[str, str]) -> None:
    """
    Save acronym expansions to JSON config file.

    Args:
        expansions: Dictionary of acronym -> expansion mappings
    """
    global _acronym_expansions

    try:
        # Ensure config directory exists
        ACRONYMS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Save to file with pretty formatting
        with open(ACRONYMS_CONFIG_PATH, 'w') as f:
            json.dump(expansions, f, indent=2, sort_keys=True)

        # Update global cache
        _acronym_expansions = expansions
        logger.info(f"Saved {len(expansions)} acronym expansions to {ACRONYMS_CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error saving acronym expansions: {e}")
        raise


def get_acronym_expansions() -> Dict[str, str]:
    """
    Get current acronym expansions (loads from file if not cached).

    Returns:
        Dictionary of acronym -> expansion mappings
    """
    global _acronym_expansions

    if _acronym_expansions is None:
        load_acronym_expansions()

    return _acronym_expansions.copy()  # Return copy to prevent external modification


def expand_query_acronyms(query: str) -> str:
    """
    Expand common acronyms in queries to help semantic search.

    Example: "What is the PTO policy?" -> "What is the Paid Time Off (PTO) policy?"

    This helps the embedding model understand domain-specific acronyms that it
    wasn't trained on.

    Args:
        query: Original query text

    Returns:
        Query with expanded acronyms
    """
    expanded_query = query
    query_lower = query.lower()

    # Get current acronym expansions from config
    acronym_expansions = get_acronym_expansions()

    # Check for each acronym (case-insensitive)
    for acronym, expansion in acronym_expansions.items():
        # Match whole words only (not substrings)
        # Use word boundaries to avoid matching "pto" in "laptop"
        pattern = r'\b' + re.escape(acronym) + r'\b'

        if re.search(pattern, query_lower, re.IGNORECASE):
            # Replace with expansion, preserving original case for the acronym part
            expanded_query = re.sub(pattern, expansion, expanded_query, flags=re.IGNORECASE)
            logger.info(f"Expanded acronym '{acronym.upper()}' -> '{expansion}' in query")
            break  # Only expand first matched acronym to avoid over-expansion

    return expanded_query


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenize text for BM25 search with stop word filtering.

    Removes common English stop words to improve keyword matching accuracy.
    """
    tokens = text.lower().split()
    # Filter out stop words and keep only meaningful terms
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


class ParentChildDocumentStore:
    """
    Manages parent and child chunk collections in ChromaDB.

    Uses sentence-transformers for local embedding generation.
    """

    def __init__(
        self,
        persist_directory: str = "/data/airgapped_rag/chromadb_advanced",
        embedding_model: str = "intfloat/e5-large-v2",  # State-of-the-art retrieval model (1024 dims)
        child_collection_name: str = "child_chunks",
        parent_collection_name: str = "parent_chunks"
    ):
        """
        Initialize the parent-child document store.

        Args:
            persist_directory: Directory for ChromaDB persistence
            embedding_model: Sentence-transformers model for embeddings
            child_collection_name: Name for child chunks collection
            parent_collection_name: Name for parent chunks collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

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
        bm25_weight: float = 0.5
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieve child chunks and optionally expand to parent chunks.

        Strategy:
        1. Run hybrid search (BM25 + Semantic) on child chunks
        2. Fuse scores with weighted combination
        3. If expand_to_parents, get corresponding parent chunks
        4. Deduplicate parents
        5. Return both child and parent chunks

        Args:
            query: Query text
            top_k: Number of child chunks to retrieve
            expand_to_parents: Whether to expand to parent chunks
            parent_limit: Maximum number of unique parent chunks to return
            metadata_filter: Optional metadata filter for search
            use_hybrid: Whether to use hybrid (BM25 + semantic) search
            bm25_weight: Weight for BM25 scores (0.0-1.0), semantic gets (1-bm25_weight)

        Returns:
            Tuple of (child_results, parent_results)
        """
        logger.debug(f"Retrieving for query: {query[:100]}... (hybrid={use_hybrid})")

        if use_hybrid:
            child_chunks = self._hybrid_search(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
                bm25_weight=bm25_weight
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
        # Expand acronyms to help semantic understanding
        expanded_query = expand_query_acronyms(query)

        # Generate query embedding (using expanded query for better semantic match)
        query_embedding = self.embedding_model.encode(
            expanded_query,
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
        bm25_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic (embedding) and BM25 (keyword) search.

        Strategy: SEMANTIC-FIRST APPROACH
        1. Use semantic embeddings as PRIMARY retrieval (finds semantically relevant chunks)
        2. Use BM25 as BOOSTER (boosts chunks with exact keyword matches)
        3. Only apply lenient filtering to remove truly irrelevant results

        This approach leverages the power of embeddings to understand query intent
        while still rewarding exact keyword matches.

        Args:
            query: Query text
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            bm25_weight: Weight for BM25 scores (0.0-1.0)

        Returns:
            List of child chunk results sorted by fused score
        """
        logger.debug(f"Running semantic-first hybrid search: BM25_weight={bm25_weight}")

        # Step 1: Get semantic search results (SEMANTIC FIRST!)
        # Fetch more results to allow BM25 boosting to rerank them
        semantic_results = self._semantic_search(
            query=query,
            top_k=min(top_k * 5, 100),  # Get 5x results for better fusion, cap at 100
            metadata_filter=metadata_filter
        )

        if not semantic_results:
            logger.info("No semantic results found")
            return []

        # Step 2: Get chunks for BM25 scoring
        # Only get chunks from semantic results (much more efficient than indexing all chunks)
        chunk_ids = [r['id'] for r in semantic_results]

        chunks_for_bm25 = self.child_collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"]
        )

        # Extract just the content text for BM25 (remove metadata prefix)
        # Chunk format: "Document: XXX.pdf | Section: YYY\n\n[content]"
        # We only want [content] for BM25 to avoid matching on section titles
        bm25_texts = []
        for doc in chunks_for_bm25['documents']:
            # Split on double newline to separate metadata from content
            parts = doc.split('\n\n', 1)
            if len(parts) > 1:
                # Use only the content part
                bm25_texts.append(parts[1])
            else:
                # Fallback if format is different
                bm25_texts.append(doc)

        # Step 3: Build BM25 index and get scores
        tokenized_corpus = [tokenize_for_bm25(text) for text in bm25_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = tokenize_for_bm25(query)
        logger.info(f"BM25 query tokens (stop words filtered): {tokenized_query}")
        bm25_scores = bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to 0-1 range
        if max(bm25_scores) > 0:
            bm25_scores_normalized = bm25_scores / max(bm25_scores)
        else:
            bm25_scores_normalized = bm25_scores

        # Step 4: Fuse semantic and BM25 scores
        # LENIENT filtering - only filter out completely irrelevant results
        MIN_SEMANTIC_THRESHOLD = 0.15  # Very low - trust the embeddings!

        fused_results = []
        for i, result in enumerate(semantic_results):
            semantic_score = result['score']

            # Filter only truly irrelevant results (very low semantic similarity)
            if semantic_score < MIN_SEMANTIC_THRESHOLD:
                logger.debug(
                    f"Filtered out chunk {result['id'][:8]}... - "
                    f"very low semantic relevance (semantic={semantic_score:.3f})"
                )
                continue

            # Get BM25 score for this chunk (0 if not found)
            bm25_score = float(bm25_scores_normalized[i]) if i < len(bm25_scores_normalized) else 0.0

            # Weighted fusion (semantic is primary, BM25 is booster)
            fused_score = ((1 - bm25_weight) * semantic_score) + (bm25_weight * bm25_score)

            fused_results.append({
                "id": result['id'],
                "text": result['text'],
                "metadata": result['metadata'],
                "score": fused_score,
                "semantic_score": semantic_score,
                "bm25_score": bm25_score,
                "retrieval_method": "hybrid_semantic_first"
            })

        # Step 5: Sort by fused score and return top_k
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = fused_results[:top_k]

        if not top_results:
            logger.info("Hybrid search: no documents passed relevance threshold")
            return []

        logger.info(
            f"Hybrid search: top result semantic={top_results[0]['semantic_score']:.3f}, "
            f"BM25={top_results[0]['bm25_score']:.3f}, "
            f"fused={top_results[0]['score']:.3f}"
        )

        # Debug: Log top 5 semantic and BM25 scores
        top5_semantic = [f"{r['semantic_score']:.3f}" for r in top_results[:5]]
        top5_bm25 = [f"{r['bm25_score']:.3f}" for r in top_results[:5]]
        logger.info(f"Top 5 semantic scores: {top5_semantic}")
        logger.info(f"Top 5 BM25 scores: {top5_bm25}")

        # Debug: Log which documents are in top results
        for i, result in enumerate(top_results[:10], 1):
            doc_title = result['metadata'].get('document_title', 'Unknown')
            section = result['metadata'].get('section_title', 'Unknown')
            logger.info(
                f"Rank {i}: {doc_title} - {section} "
                f"(semantic={result['semantic_score']:.3f}, BM25={result['bm25_score']:.3f}, fused={result['score']:.3f})"
            )

        return top_results

    def _expand_to_parents(
        self,
        child_chunks: List[Dict[str, Any]],
        parent_limit: int
    ) -> List[Dict[str, Any]]:
        """
        Expand child chunks to their parent chunks.

        Preserves the ranking order from child chunks - the parent of the
        highest-ranked child is returned first.

        Args:
            child_chunks: List of child chunk results (sorted by relevance)
            parent_limit: Maximum number of unique parents to return

        Returns:
            List of parent chunk results (deduplicated, rank-ordered)
        """
        # Get unique parent IDs IN ORDER of child chunk ranking
        # Use a list + seen set to preserve order while deduplicating
        parent_ids_ordered = []
        seen_parent_ids = set()

        for child in child_chunks:
            parent_id = child['metadata'].get('parent_chunk_id')
            if parent_id and parent_id not in seen_parent_ids:
                parent_ids_ordered.append(parent_id)
                seen_parent_ids.add(parent_id)

                # Stop when we have enough parents
                if len(parent_ids_ordered) >= parent_limit:
                    break

        if not parent_ids_ordered:
            logger.warning("No parent IDs found in child chunks")
            return []

        logger.debug(f"Expanding to {len(parent_ids_ordered)} unique parent chunks (rank-ordered)")

        # Retrieve parents from collection
        try:
            parent_results = self.parent_collection.get(
                ids=parent_ids_ordered,
                include=["documents", "metadatas"]
            )

            # Format parent results, preserving order
            parents = []
            # parent_results['ids'] may be in different order than requested
            # Create a lookup dict and reorder based on our original order
            parent_lookup = {
                parent_results['ids'][i]: {
                    "id": parent_results['ids'][i],
                    "text": parent_results['documents'][i],
                    "metadata": parent_results['metadatas'][i],
                    "score": 1.0
                }
                for i in range(len(parent_results['ids']))
            }

            # Reorder according to our parent_ids_ordered
            for parent_id in parent_ids_ordered:
                if parent_id in parent_lookup:
                    parents.append(parent_lookup[parent_id])

            logger.info(f"Expanded to {len(parents)} parent chunks")

            # Debug: Log which parent chunks were retrieved
            for i, parent in enumerate(parents, 1):
                logger.info(
                    f"Parent {i}: {parent['metadata'].get('document_title', 'Unknown')} - "
                    f"{parent['metadata'].get('section_title', 'Unknown')}"
                )

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
