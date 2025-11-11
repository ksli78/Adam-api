"""
Re-embed chunks with CLEAN text (remove answerable_questions).

This script removes answerable_questions from chunk text and re-embeds
all child chunks with clean text only, eliminating semantic noise.
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


def clean_chunk_text(text: str) -> str:
    """
    Remove answerable_questions section from chunk text.

    Args:
        text: Chunk text potentially containing questions section

    Returns:
        Clean chunk text without questions
    """
    # Check if questions section exists
    if '[Questions this document answers:]' in text:
        # Split and take only the part before questions
        clean_text = text.split('[Questions this document answers:]')[0].strip()
        return clean_text

    # No questions section, return as is
    return text


def reembed_clean_chunks():
    """Re-embed all chunks with clean text (no questions)."""

    logger.info("Starting clean re-embedding process...")
    logger.info("This will remove answerable_questions from chunk embeddings")

    # Initialize document store
    logger.info(f"Loading document store from {CHROMA_DIR}")
    store = get_parent_child_store(persist_directory=str(CHROMA_DIR))

    # Get all child chunks
    logger.info("Fetching all child chunks...")
    all_chunks = store.child_collection.get(
        include=["documents", "metadatas"]
    )

    total_chunks = len(all_chunks['ids'])
    logger.info(f"Found {total_chunks} child chunks to process")

    if not total_chunks:
        logger.warning("No chunks found in store!")
        return

    # Clean all chunk texts
    logger.info("Cleaning chunk texts (removing questions)...")
    cleaned_texts = []
    chunks_with_questions = 0

    for text in all_chunks['documents']:
        clean_text = clean_chunk_text(text)
        cleaned_texts.append(clean_text)

        # Count how many had questions
        if '[Questions this document answers:]' in text:
            chunks_with_questions += 1

    logger.info(f"  Chunks with questions removed: {chunks_with_questions}/{total_chunks}")
    logger.info(f"  Chunks already clean: {total_chunks - chunks_with_questions}/{total_chunks}")

    # Generate new embeddings with clean text
    logger.info("Generating new embeddings (this will take ~10-15 minutes)...")

    # Process in batches to show progress
    batch_size = 100
    all_embeddings = []

    for i in tqdm(range(0, len(cleaned_texts), batch_size), desc="Embedding batches"):
        batch_texts = cleaned_texts[i:i+batch_size]
        batch_embeddings = store.embedding_model.encode(
            batch_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        all_embeddings.extend(batch_embeddings)

    # Update ChromaDB with clean texts and new embeddings
    logger.info("Updating ChromaDB with clean chunks...")
    store.child_collection.update(
        ids=all_chunks['ids'],
        documents=cleaned_texts,  # Clean text without questions
        embeddings=[emb.tolist() for emb in all_embeddings],  # New embeddings
        metadatas=all_chunks['metadatas']  # CRITICAL: Preserve metadata including parent_chunk_id!
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Clean re-embedding complete!")
    logger.info(f"Total chunks updated: {total_chunks}")
    logger.info(f"Questions removed from: {chunks_with_questions} chunks")
    logger.info(f"{'='*60}")

    # Show new statistics
    stats = store.get_statistics()
    logger.info(f"Final statistics: {stats}")


if __name__ == "__main__":
    try:
        reembed_clean_chunks()
        logger.info("\n✅ Clean re-embedding completed successfully!")
        logger.info("Chunks now contain only their original text, no questions appended.")
        logger.info("This should improve search accuracy by eliminating semantic noise.")
    except Exception as e:
        logger.error(f"\n❌ Re-embedding failed: {e}", exc_info=True)
        raise
