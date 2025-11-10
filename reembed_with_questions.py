"""
Re-embed existing documents with answerable_questions included.

This script updates all existing child chunks to include answerable_questions
in their embeddings and stored text, enabling better semantic and BM25 search.
"""

import logging
import os
from pathlib import Path
from parent_child_store import get_parent_child_store
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "/data/airgapped_rag"))
CHROMA_DIR = DATA_DIR / "chromadb_advanced"


def reembed_all_documents():
    """Re-embed all documents with answerable_questions included."""

    logger.info("Starting re-embedding process...")

    # Initialize document store
    logger.info(f"Loading document store from {CHROMA_DIR}")
    store = get_parent_child_store(persist_directory=str(CHROMA_DIR))

    # Get all unique documents
    all_docs = store.get_all_documents_with_metadata()
    logger.info(f"Found {len(all_docs)} documents to process")

    if not all_docs:
        logger.warning("No documents found in store!")
        return

    total_chunks_updated = 0

    # Process each document
    for doc in tqdm(all_docs, desc="Re-embedding documents"):
        doc_id = doc['document_id']
        doc_title = doc['document_title']

        # Extract answerable questions
        answerable_questions = doc.get('answerable_questions', [])
        if not answerable_questions:
            logger.warning(f"No answerable questions found for {doc_title}, skipping")
            continue

        logger.info(f"Processing {doc_title} ({len(answerable_questions)} questions)")

        # Get all child chunks for this document
        child_chunks = store.child_collection.get(
            where={"document_id": doc_id},
            include=["documents", "metadatas"]
        )

        if not child_chunks['ids']:
            logger.warning(f"No child chunks found for {doc_title}")
            continue

        num_chunks = len(child_chunks['ids'])
        logger.info(f"  Found {num_chunks} child chunks")

        # Extract original chunk text (strip questions if they exist)
        original_texts = []
        for doc_text in child_chunks['documents']:
            # Check if questions are already in text
            if '[Questions this document answers:]' in doc_text:
                # Strip existing questions section
                original_text = doc_text.split('[Questions this document answers:]')[0].strip()
                original_texts.append(original_text)
            else:
                original_texts.append(doc_text)

        # Create augmented texts (original + questions)
        questions_text = "\n\n[Questions this document answers:]\n" + "\n".join(f"- {q}" for q in answerable_questions)
        augmented_texts = [text + questions_text for text in original_texts]

        # Generate new embeddings for augmented texts
        logger.info(f"  Generating new embeddings...")
        new_embeddings = store.embedding_model.encode(
            augmented_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Update ChromaDB with new embeddings and documents
        logger.info(f"  Updating ChromaDB...")
        store.child_collection.update(
            ids=child_chunks['ids'],
            documents=augmented_texts,  # Update stored text
            embeddings=[emb.tolist() for emb in new_embeddings]  # Update embeddings
        )

        total_chunks_updated += num_chunks
        logger.info(f"  ✓ Updated {num_chunks} chunks for {doc_title}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Re-embedding complete!")
    logger.info(f"Documents processed: {len(all_docs)}")
    logger.info(f"Total chunks updated: {total_chunks_updated}")
    logger.info(f"{'='*60}")

    # Show new statistics
    stats = store.get_statistics()
    logger.info(f"Final statistics: {stats}")


if __name__ == "__main__":
    try:
        reembed_all_documents()
        logger.info("\n✅ Re-embedding completed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Re-embedding failed: {e}", exc_info=True)
        raise
