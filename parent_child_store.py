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
        embedding_model: str = "all-MiniLM-L6-v2",
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
        bm25_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining BM25 (keyword) and semantic (embedding) search.

        Uses weighted score fusion:
        final_score = (bm25_weight * bm25_score) + ((1 - bm25_weight) * semantic_score)

        Args:
            query: Query text
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            bm25_weight: Weight for BM25 scores (0.0-1.0)

        Returns:
            List of child chunk results sorted by fused score
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

        # Step 5: Fuse scores and filter out weak BM25 matches
        # In hybrid mode, require minimum keyword relevance to avoid semantic-only matches
        MIN_BM25_THRESHOLD = 0.3  # Require at least 30% of max BM25 score

        fused_results = []
        for i, chunk_id in enumerate(all_chunks['ids']):
            # BM25 score (already normalized 0-1)
            bm25_score = float(bm25_scores_normalized[i])

            # Skip chunks with weak keyword relevance in hybrid mode
            # This filters out documents that only match common words like "policy"
            # but miss important keywords like "PTO"
            if bm25_weight > 0 and bm25_score < MIN_BM25_THRESHOLD:
                continue

            # Semantic score (0 if not in semantic results)
            semantic_score = semantic_scores.get(chunk_id, 0.0)

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

        # Step 6: Sort by fused score and return top_k
        fused_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = fused_results[:top_k]

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
        # Get unique parent IDs
        parent_ids = set()
        for child in child_chunks:
            parent_id = child['metadata'].get('parent_chunk_id')
            if parent_id:
                parent_ids.add(parent_id)

        if not parent_ids:
            logger.warning("No parent IDs found in child chunks")
            return []

        logger.debug(f"Expanding to {len(parent_ids)} unique parent chunks")

        # Retrieve parents from collection
        parent_ids_list = list(parent_ids)[:parent_limit]  # Limit number of parents

        try:
            parent_results = self.parent_collection.get(
                ids=parent_ids_list,
                include=["documents", "metadatas"]
            )

            # Format parent results
            parents = []
            for i in range(len(parent_results['ids'])):
                parents.append({
                    "id": parent_results['ids'][i],
                    "text": parent_results['documents'][i],
                    "metadata": parent_results['metadatas'][i],
                    "score": 1.0  # Parents don't have relevance scores initially
                })

            logger.info(f"Expanded to {len(parents)} parent chunks")
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
