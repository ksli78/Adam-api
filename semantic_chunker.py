"""
Semantic Chunker Service

Creates parent-child chunk relationships for optimal RAG retrieval:
- Parent chunks: Large sections (1000-2000 tokens) for LLM context
- Child chunks: Small semantic units (200-400 tokens) for precise retrieval

Uses sentence-aware splitting to avoid breaking mid-sentence.
"""

import logging
import re
import uuid
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import nltk

# Download NLTK data if not present
# Newer NLTK versions use punkt_tab instead of punkt
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        # Fallback for older NLTK versions
        nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: str = None
    chunk_index: int = 0


@dataclass
class DocumentSection:
    """Represents a document section with hierarchy."""
    title: str
    text: str
    level: int  # 0 = document, 1 = main section, 2 = subsection, etc.
    section_number: str = ""  # e.g., "4.3"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticChunker:
    """
    Creates semantic chunks with parent-child relationships.

    Strategy:
    1. Split document into sections (using structure from Docling)
    2. Create parent chunks from full sections
    3. Split parent chunks into smaller child chunks at sentence boundaries
    4. Add contextual information to child chunks
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,  # tokens
        child_chunk_size: int = 300,  # tokens
        chunk_overlap: int = 50,  # tokens
        chars_per_token: float = 4.0  # approximate
    ):
        """
        Initialize the semantic chunker.

        Args:
            parent_chunk_size: Target size for parent chunks (tokens)
            child_chunk_size: Target size for child chunks (tokens)
            chunk_overlap: Overlap between child chunks (tokens)
            chars_per_token: Approximate characters per token for estimation
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.chars_per_token = chars_per_token

        logger.info(
            f"SemanticChunker initialized: "
            f"parent={parent_chunk_size}tok, child={child_chunk_size}tok, overlap={chunk_overlap}tok"
        )

    def chunk_document(
        self,
        sections: List[DocumentSection],
        document_title: str,
        document_id: str = None
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Chunk a document into parent and child chunks.

        Args:
            sections: List of document sections with hierarchy
            document_title: Title of the document
            document_id: Optional document ID

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        if document_id is None:
            document_id = str(uuid.uuid4())

        parent_chunks = []
        all_child_chunks = []

        for section_idx, section in enumerate(sections):
            # Create parent chunk from full section
            parent_chunk = self._create_parent_chunk(
                section=section,
                section_idx=section_idx,
                document_title=document_title,
                document_id=document_id
            )
            parent_chunks.append(parent_chunk)

            # Split parent into child chunks
            child_chunks = self._create_child_chunks(
                parent_chunk=parent_chunk,
                section=section,
                document_title=document_title
            )
            all_child_chunks.extend(child_chunks)

        logger.info(
            f"Created {len(parent_chunks)} parent chunks and "
            f"{len(all_child_chunks)} child chunks for document '{document_title}'"
        )

        return parent_chunks, all_child_chunks

    def _create_parent_chunk(
        self,
        section: DocumentSection,
        section_idx: int,
        document_title: str,
        document_id: str
    ) -> Chunk:
        """Create a parent chunk from a document section."""
        parent_id = f"{document_id}_parent_{section_idx}"

        # Full section text (may be large)
        full_text = section.text

        # Estimate token count
        token_count = self._estimate_tokens(full_text)

        # If section is too large, truncate (but warn)
        max_parent_chars = int(self.parent_chunk_size * self.chars_per_token * 1.5)
        if len(full_text) > max_parent_chars:
            logger.warning(
                f"Section '{section.title}' is very large ({len(full_text)} chars), "
                f"truncating to {max_parent_chars} chars"
            )
            full_text = full_text[:max_parent_chars]
            token_count = self._estimate_tokens(full_text)

        return Chunk(
            id=parent_id,
            text=full_text,
            token_count=token_count,
            metadata={
                "type": "parent",
                "document_id": document_id,
                "document_title": document_title,
                "section_title": section.title,
                "section_number": section.section_number,
                "section_level": section.level,
                "section_index": section_idx,
                **section.metadata
            },
            parent_id=None,
            chunk_index=section_idx
        )

    def _create_child_chunks(
        self,
        parent_chunk: Chunk,
        section: DocumentSection,
        document_title: str
    ) -> List[Chunk]:
        """
        Split parent chunk into smaller child chunks at sentence boundaries.

        Args:
            parent_chunk: The parent chunk to split
            section: Original document section
            document_title: Title of the document

        Returns:
            List of child chunks
        """
        # Add context to chunk text (for better embeddings)
        context_prefix = self._create_context_prefix(document_title, section.title)

        # Split into sentences
        sentences = sent_tokenize(parent_chunk.text)

        # Group sentences into chunks
        child_chunks = []
        current_chunk_sentences = []
        current_token_count = len(context_prefix.split())  # Start with context

        target_child_tokens = self.child_chunk_size
        overlap_tokens = self.chunk_overlap

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # Check if adding this sentence would exceed target
            if current_token_count + sentence_tokens > target_child_tokens and current_chunk_sentences:
                # Create chunk from accumulated sentences
                chunk_text = context_prefix + " ".join(current_chunk_sentences)
                child_chunk = Chunk(
                    id=f"{parent_chunk.id}_child_{len(child_chunks)}",
                    text=chunk_text,
                    token_count=self._estimate_tokens(chunk_text),
                    parent_id=parent_chunk.id,
                    chunk_index=len(child_chunks),
                    metadata={
                        "type": "child",
                        "document_id": parent_chunk.metadata.get("document_id"),
                        "document_title": document_title,
                        "section_title": section.title,
                        "section_number": section.section_number,
                        "parent_chunk_id": parent_chunk.id,
                        **parent_chunk.metadata
                    }
                )
                child_chunks.append(child_chunk)

                # Start new chunk with overlap
                # Keep last few sentences for context continuity
                overlap_sentence_count = max(1, len(current_chunk_sentences) // 4)
                current_chunk_sentences = current_chunk_sentences[-overlap_sentence_count:]
                current_token_count = sum(
                    self._estimate_tokens(s) for s in current_chunk_sentences
                )

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = context_prefix + " ".join(current_chunk_sentences)
            child_chunk = Chunk(
                id=f"{parent_chunk.id}_child_{len(child_chunks)}",
                text=chunk_text,
                token_count=self._estimate_tokens(chunk_text),
                parent_id=parent_chunk.id,
                chunk_index=len(child_chunks),
                metadata={
                    "type": "child",
                    "document_id": parent_chunk.metadata.get("document_id"),
                    "document_title": document_title,
                    "section_title": section.title,
                    "section_number": section.section_number,
                    "parent_chunk_id": parent_chunk.id,
                    **parent_chunk.metadata
                }
            )
            child_chunks.append(child_chunk)

        logger.debug(f"Split parent chunk into {len(child_chunks)} child chunks")
        return child_chunks

    def _create_context_prefix(self, document_title: str, section_title: str) -> str:
        """
        Create contextual prefix for chunk text.

        This helps with embedding quality by providing context.
        """
        prefix_parts = []

        if document_title:
            prefix_parts.append(f"Document: {document_title}")

        if section_title:
            prefix_parts.append(f"Section: {section_title}")

        if prefix_parts:
            return " | ".join(prefix_parts) + "\n\n"
        else:
            return ""

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.

        Uses simple heuristic: ~4 characters per token.
        For production, could use actual tokenizer.
        """
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))

    def extract_sections_from_markdown(self, markdown_text: str) -> List[DocumentSection]:
        """
        Extract sections from markdown text based on headers.

        Handles Docling's multi-line format where section numbers and titles
        appear on separate lines:
            1.0
            Purpose
            Content...

        Args:
            markdown_text: Markdown formatted text

        Returns:
            List of document sections with hierarchy
        """
        sections = []
        lines = markdown_text.split('\n')

        current_section = {
            "title": "Document",
            "text": [],
            "level": 0,
            "section_number": ""
        }

        # Patterns
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        # Docling puts section numbers on their own line (e.g., "1.0", "2.0", "5.1", "5.3.4")
        section_number_pattern = re.compile(r'^(\d+(?:\.\d+)*)$')

        # Skip these patterns when looking for section titles (boilerplate)
        boilerplate_patterns = [
            re.compile(r'This\s+document\s+contains\s+proprietary', re.IGNORECASE),
            re.compile(r'Unauthorized\s+use,\s+reproduction', re.IGNORECASE),
            re.compile(r'Copyright.*All\s+rights\s+reserved', re.IGNORECASE),
            re.compile(r'Page:\s+\d+\s+of\s+\d+', re.IGNORECASE),
            re.compile(r'Revision:\s+[A-Z]', re.IGNORECASE),
            re.compile(r'^[A-Z]{2}-[A-Z]{2}-\d{4}$'),  # Document IDs like EN-PO-0301
        ]

        # Pattern to detect if a "title" is actually content (too long or starts with lowercase)
        def is_valid_section_title(title: str) -> bool:
            """Check if a title looks like an actual section title, not content."""
            if not title:
                return False
            # Titles should be reasonably short (not full paragraphs)
            if len(title) > 200:
                return False
            # Titles that start with numbers followed by period and space are likely content
            # e.g., "5.1.1 The established standard..." is content, not a title
            if re.match(r'^\d+\.\d+\.\d+\s+', title):
                return False
            return True

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for markdown header (# Title)
            header_match = header_pattern.match(line)
            if header_match:
                # Save previous section if it has content
                if current_section["text"]:
                    sections.append(DocumentSection(
                        title=current_section["title"],
                        text="\n".join(current_section["text"]),
                        level=current_section["level"],
                        section_number=current_section["section_number"]
                    ))

                # Start new section
                level = len(header_match.group(1))  # Number of #
                title = header_match.group(2).strip()
                current_section = {
                    "title": title,
                    "text": [],
                    "level": level,
                    "section_number": ""
                }
                i += 1
                continue

            # Check for Docling-style section number on its own line
            section_num_match = section_number_pattern.match(line)
            if section_num_match:
                section_num = section_num_match.group(1)

                # Look ahead for the section title (next non-empty, non-boilerplate line)
                section_title = ""
                title_line_idx = i
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line:
                        # Check if it's another section number (edge case: consecutive sections)
                        if section_number_pattern.match(next_line):
                            break

                        # Check if it's boilerplate (skip it)
                        is_boilerplate = any(pattern.search(next_line) for pattern in boilerplate_patterns)
                        if not is_boilerplate:
                            # Found a potential title - validate it
                            if is_valid_section_title(next_line):
                                section_title = next_line
                                title_line_idx = j
                            break
                    j += 1

                # If we found a valid title, create a new section
                if section_title:
                    # Save previous section
                    if current_section["text"]:
                        sections.append(DocumentSection(
                            title=current_section["title"],
                            text="\n".join(current_section["text"]),
                            level=current_section["level"],
                            section_number=current_section["section_number"]
                        ))

                    # Start new section
                    level = section_num.count('.') + 1
                    current_section = {
                        "title": section_title,
                        "text": [],
                        "level": level,
                        "section_number": section_num
                    }
                    i = title_line_idx + 1  # Skip past the title line
                    continue
                else:
                    # No valid title found - skip this bare section number
                    # (it's likely a formatting artifact)
                    i += 1
                    continue

            # Add line to current section (if not empty or if we already have content)
            # But skip boilerplate lines and bare section numbers
            is_boilerplate = any(pattern.search(line) for pattern in boilerplate_patterns)
            is_bare_section_number = section_number_pattern.match(line)

            if not is_boilerplate and not is_bare_section_number and (line or current_section["text"]):
                current_section["text"].append(lines[i])  # Use original line with whitespace

            i += 1

        # Don't forget the last section
        if current_section["text"]:
            # Clean up trailing empty lines
            while current_section["text"] and not current_section["text"][-1].strip():
                current_section["text"].pop()

            if current_section["text"]:
                sections.append(DocumentSection(
                    title=current_section["title"],
                    text="\n".join(current_section["text"]),
                    level=current_section["level"],
                    section_number=current_section["section_number"]
                ))

        logger.info(f"Extracted {len(sections)} sections from markdown")
        return sections


# Singleton instance
_chunker_instance = None


def get_semantic_chunker(**kwargs) -> SemanticChunker:
    """
    Get or create singleton SemanticChunker instance.

    Args:
        **kwargs: Arguments to pass to SemanticChunker constructor

    Returns:
        SemanticChunker instance
    """
    global _chunker_instance

    if _chunker_instance is None:
        _chunker_instance = SemanticChunker(**kwargs)

    return _chunker_instance


if __name__ == "__main__":
    # Test the chunker
    logging.basicConfig(level=logging.DEBUG)

    test_markdown = """
# Time Off Policy

This document outlines the time off policies for all employees.

## 1. Purpose

The purpose of this policy is to provide clear guidelines on time off benefits.
All employees are eligible for paid time off as described in this document.

## 2. Scope

This policy applies to all full-time and part-time employees.
Contractors and temporary workers are not covered by this policy.

## 4.3 PTO Accrual

Full-time employees accrue 15 days of PTO per year.
Part-time employees accrue PTO on a pro-rated basis.
PTO accrual begins on the employee's start date.

Employees can carry over up to 5 days of unused PTO to the next year.
Any PTO exceeding this limit will be forfeited.

## 5. Requesting Time Off

Employees must request time off at least 2 weeks in advance.
Requests are submitted through the HR portal.
"""

    chunker = get_semantic_chunker(
        parent_chunk_size=500,  # Smaller for testing
        child_chunk_size=150
    )

    sections = chunker.extract_sections_from_markdown(test_markdown)
    print(f"\n=== Extracted {len(sections)} sections ===")
    for s in sections:
        print(f"  {s.section_number} {s.title} (level {s.level}): {len(s.text)} chars")

    parent_chunks, child_chunks = chunker.chunk_document(
        sections=sections,
        document_title="EN-PO-0301 Time Off Policy",
        document_id="test-doc-001"
    )

    print(f"\n=== Created {len(parent_chunks)} parent chunks ===")
    for p in parent_chunks:
        print(f"  {p.id}: {p.token_count} tokens, section '{p.metadata['section_title']}'")

    print(f"\n=== Created {len(child_chunks)} child chunks ===")
    for c in child_chunks:
        print(f"  {c.id}: {c.token_count} tokens, parent={c.parent_id}")
        print(f"    Preview: {c.text[:100]}...")
