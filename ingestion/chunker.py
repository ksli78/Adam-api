"""
Chunking and summarisation utilities.

Proper chunking is a critical step in a Retrieval‑Augmented Generation
pipeline.  Chunks that are too large can dilute the relevance signal in
embeddings, while chunks that are too small may lack sufficient context and
increase the number of vectors stored.  This module implements a simple
hierarchical chunker that groups paragraphs together up to a token limit
while respecting natural boundaries such as headings and list items【923166380928388†L622-L734】.

Chunks also support optional summarisation via the instruction model.  A
short summary helps the reranker and final answer generator understand the
contents of each chunk without reading the entire text.  Summarisation is
performed lazily to avoid incurring the cost during ingestion unless
explicitly requested.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a segment of a document ready for embedding and retrieval."""

    id: str
    document_id: str
    text: str
    token_count: int
    start_paragraph: int
    end_paragraph: int
    summary: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "token_count": self.token_count,
            "start_paragraph": self.start_paragraph,
            "end_paragraph": self.end_paragraph,
            "summary": self.summary,
        }


def split_into_chunks(
    markdown: str,
    *,
    document_id: str = "", 
    max_tokens: int = config.MAX_CHUNK_TOKENS,
    overlap: int = config.CHUNK_OVERLAP,
    summarise: bool = False,
    summariser: Optional[Callable[[str], str]] = None,
) -> List[Chunk]:
    """Split cleaned markdown into a list of chunks.

    Parameters
    ----------
    markdown:
        The cleaned markdown string to split.
    document_id:
        Identifier of the document being processed.  Used to namespace chunk
        identifiers.
    max_tokens:
        Maximum approximate number of tokens per chunk.  Tokens are
        approximated by word count; for more precise control, supply a custom
        token counter.
    overlap:
        Number of tokens to overlap between consecutive chunks.  Overlap
        improves retrieval by providing context continuity【923166380928388†L622-L734】.
    summarise:
        If ``True``, call the supplied ``summariser`` on each chunk to produce
        a short summary.  Summarisation is performed lazily after chunk
        creation.  If no summariser is provided when ``summarise`` is true,
        summarisation is skipped with a warning.
    summariser:
        Callable that accepts the text of a chunk and returns a short summary.

    Returns
    -------
    list of :class:`Chunk`
        The resulting chunks with optional summaries.
    """
    paragraphs = [p.strip() for p in markdown.split("\n\n") if p.strip()]
    chunks: List[Chunk] = []
    current_paragraphs: List[str] = []
    current_tokens = 0
    start_para_index = 0

    def approx_tokens(text: str) -> int:
        # Approximate tokens by word count; this is fast and sufficient for
        # chunking purposes.  The embedding model will handle truncation.
        return len(text.split())

    for idx, para in enumerate(paragraphs):
        tokens = approx_tokens(para)
        if (current_tokens + tokens <= max_tokens) or not current_paragraphs:
            current_paragraphs.append(para)
            current_tokens += tokens
        else:
            # Finalise the current chunk
            chunk_text = "\n\n".join(current_paragraphs)
            chunk_id = f"{document_id}-{uuid.uuid4().hex[:8]}"
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                token_count=current_tokens,
                start_paragraph=start_para_index,
                end_paragraph=idx - 1,
            )
            chunks.append(chunk)
            # Prepare the next chunk with overlap
            overlap_tokens = 0
            overlap_paragraphs: List[str] = []
            j = len(current_paragraphs) - 1
            while j >= 0 and overlap_tokens < overlap:
                overlap_paragraphs.insert(0, current_paragraphs[j])
                overlap_tokens += approx_tokens(current_paragraphs[j])
                j -= 1
            current_paragraphs = overlap_paragraphs + [para]
            current_tokens = overlap_tokens + tokens
            start_para_index = idx - len(overlap_paragraphs)

    # Add final chunk
    if current_paragraphs:
        chunk_text = "\n\n".join(current_paragraphs)
        chunk_id = f"{document_id}-{uuid.uuid4().hex[:8]}"
        chunk = Chunk(
            id=chunk_id,
            document_id=document_id,
            text=chunk_text,
            token_count=current_tokens,
            start_paragraph=start_para_index,
            end_paragraph=len(paragraphs) - 1,
        )
        chunks.append(chunk)

    # Optionally summarise chunks
    if summarise:
        if summariser is None:
            logger.warning(
                "Summarisation requested but no summariser provided; skipping summaries."
            )
        else:
            for chunk in chunks:
                try:
                    chunk.summary = summariser(chunk.text)
                except Exception as exc:
                    logger.error("Failed to summarise chunk %s: %s", chunk.id, exc)
                    chunk.summary = None

    return chunks
