"""
Restore parent_chunk_id metadata that was lost during re-embedding.

When reembed_clean_chunks.py was run without preserving metadata,
all parent_chunk_id fields were lost. This script restores them by
extracting the parent ID from the child chunk ID format.

Child chunk IDs follow the format: {parent_id}_child_{index}
Example: "abc123_child_0" has parent_id "abc123"
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


def extract_parent_id(child_id: str) -> str:
    """
    Extract parent_id from child chunk ID.

    Child chunk IDs follow format: {parent_id}_child_{index}

    Args:
        child_id: Child chunk ID

    Returns:
        Parent chunk ID
    """
    # Split on "_child_" and take the first part
    if "_child_" in child_id:
        return child_id.split("_child_")[0]
    else:
        logger.warning(f"Unexpected child ID format: {child_id}")
        return None


def restore_parent_metadata():
    """Restore parent_chunk_id metadata for all child chunks."""

    logger.info("Starting metadata restoration process...")
    logger.info("This will restore parent_chunk_id fields lost during re-embedding")

    # Initialize document store
    logger.info(f"Loading document store from {CHROMA_DIR}")
    store = get_parent_child_store(persist_directory=str(CHROMA_DIR))

    # Get all child chunks
    logger.info("Fetching all child chunks...")
    all_chunks = store.child_collection.get(
        include=["metadatas"]
    )

    total_chunks = len(all_chunks['ids'])
    logger.info(f"Found {total_chunks} child chunks to process")

    if not total_chunks:
        logger.warning("No chunks found in store!")
        return

    # Restore parent_chunk_id in metadata
    logger.info("Restoring parent_chunk_id metadata...")
    updated_metadatas = []
    chunks_restored = 0
    chunks_already_had_parent = 0
    chunks_failed = 0

    for i, (chunk_id, metadata) in enumerate(zip(all_chunks['ids'], all_chunks['metadatas'])):
        # Check if parent_chunk_id already exists
        if metadata.get('parent_chunk_id'):
            chunks_already_had_parent += 1
            updated_metadatas.append(metadata)
            continue

        # Extract parent_id from chunk ID
        parent_id = extract_parent_id(chunk_id)

        if parent_id:
            # Add parent_chunk_id to metadata
            updated_metadata = metadata.copy()
            updated_metadata['parent_chunk_id'] = parent_id
            updated_metadatas.append(updated_metadata)
            chunks_restored += 1
        else:
            # Keep original metadata if we can't extract parent_id
            updated_metadatas.append(metadata)
            chunks_failed += 1

    # Update ChromaDB with restored metadata
    logger.info("Updating ChromaDB with restored metadata...")
    store.child_collection.update(
        ids=all_chunks['ids'],
        metadatas=updated_metadatas
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Metadata restoration complete!")
    logger.info(f"Total chunks processed: {total_chunks}")
    logger.info(f"Metadata restored: {chunks_restored}")
    logger.info(f"Already had parent_chunk_id: {chunks_already_had_parent}")
    logger.info(f"Failed to extract parent_id: {chunks_failed}")
    logger.info(f"{'='*60}")

    # Verify restoration worked
    logger.info("\nVerifying restoration...")
    sample = store.child_collection.get(
        limit=5,
        include=["metadatas"]
    )

    for i, (chunk_id, metadata) in enumerate(zip(sample['ids'], sample['metadatas']), 1):
        parent_id = metadata.get('parent_chunk_id', 'MISSING')
        logger.info(f"  Sample {i}: child_id={chunk_id[:30]}..., parent_chunk_id={parent_id[:30] if parent_id != 'MISSING' else 'MISSING'}...")


if __name__ == "__main__":
    try:
        restore_parent_metadata()
        logger.info("\n✅ Metadata restoration completed successfully!")
        logger.info("Parent expansion should now work correctly in semantic reranking.")
    except Exception as e:
        logger.error(f"\n❌ Metadata restoration failed: {e}", exc_info=True)
        raise
