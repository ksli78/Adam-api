# ingestion/chunker.py
from __future__ import annotations
import re
import uuid
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, List, Optional, Dict
import logging

try:
    from transformers import AutoTokenizer
except Exception as exc:
    AutoTokenizer = None
    _TOKENIZER_IMPORT_ERROR = str(exc)
else:
    _TOKENIZER_IMPORT_ERROR = None

import config

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: Dict
    token_count: int = 0                 # number of tokens in this chunk’s text
    start_paragraph: int = 0            # index of first source paragraph covered
    end_paragraph: int = 0              # index of last source paragraph covered
    summary: Optional[str] = None       # optional summary of chunk (for metadata)

    def to_dict(self) -> dict:
        """Convert Chunk to dict for serialization (metadata is kept as-is)."""
        d = asdict(self)
        return d

# Pattern to detect numbered section headings (e.g. "3.1 Title of Section")
_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+)$")
# Pattern to detect section headings with parenthesis (e.g. "1) Title")
_SECTION_RE_PAREN = re.compile(r"^\s*(\d+)\)\s+(.+)$")

# Load tokenizer for the embedding model (to count tokens for chunking)
_TOKENIZER = None
def _get_tokenizer():
    """Load the HF tokenizer for the embedder model, if available."""
    global _TOKENIZER
    if AutoTokenizer is None:
        return None
    if _TOKENIZER is None:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(config.EMBED_MODEL_NAME)
        except Exception as e:
            logger.warning("Could not load tokenizer for %s: %s", config.EMBED_MODEL_NAME, e)
            _TOKENIZER = None
    return _TOKENIZER

def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs by blank lines (preserving meaningful newlines)."""
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return parts

def _is_likely_heading(paragraph: str) -> bool:
    """
    Heuristic to identify if a paragraph is a section heading (without explicit numbering).
    - It should be short (<= ~100 chars) and not end with sentence punctuation.
    - It should consist mostly of title-case or uppercase words (indicating a title).
    - It should not be an obvious list item (e.g. starting with a bullet or number).
    """
    text = paragraph.strip()
    if not text or len(text) > 100:
        return False
    # If line ends in period, question mark, etc., treat it as a sentence, not a heading.
    if text[-1] in ".?!;:":
        return False
    # Skip lines that start with list markers (bullets or enumerated list items).
    if re.match(r"^(\*|-|•|\d+[.)])\s", text):
        return False
    # Count capitalization: consider it a heading if most words are Capitalized or UPPER.
    words = text.split()
    if not words:
        return False
    num_capitalized = sum(1 for w in words if w[:1].isupper() and w[1:].islower())
    num_upper = sum(1 for w in words if w.isupper())
    num_lower = sum(1 for w in words if w.islower())
    # If no lowercase words, or more capitalized/upper words than lower, likely a heading.
    if num_lower == 0 or (num_capitalized + num_upper) > num_lower:
        return True
    return False

def _group_by_section(paragraphs: List[str]) -> List[Dict]:
    """
    Group paragraphs under the nearest preceding section heading.
    Recognizes numeric headings (1.2.3), markdown # headings, and likely title-case headings.
    Returns a list of section groups, each with keys: section, heading, content, start_paragraph, end_paragraph.
    """
    groups: List[Dict] = []
    current_section = {"section": "", "heading": "", "content": []}
    current_start_idx = 0  # index of first paragraph in the current section group
    for i, para in enumerate(paragraphs):
        # Check for numbered headings (e.g. "2. Title" or "2) Title"):
        m = _SECTION_RE.match(para)
        m2 = _SECTION_RE_PAREN.match(para) if not m else None
        if m or m2:
            # Flush the previous section group (if it has any content).
            if current_section["content"]:
                groups.append({
                    "section": current_section["section"],
                    "heading": current_section["heading"],
                    "content": "\n".join(current_section["content"]).strip(),
                    "start_paragraph": current_start_idx,
                    "end_paragraph": i - 1
                })
            # Start a new section group with this heading
            if m:
                section_num, heading_text = m.group(1), m.group(2)
            else:
                section_num, heading_text = m2.group(1), m2.group(2)
            current_section = {"section": section_num, "heading": heading_text, "content": []}
            current_start_idx = i  # new group starts at this heading paragraph index
            continue
        # Check for markdown-style headings (e.g. lines starting with "#"):
        if para.strip().startswith("#"):
            heading_line = para.strip().lstrip("#").strip()
            if heading_line:  # a valid heading after '#'
                if current_section["content"]:
                    groups.append({
                        "section": current_section["section"],
                        "heading": current_section["heading"],
                        "content": "\n".join(current_section["content"]).strip(),
                        "start_paragraph": current_start_idx,
                        "end_paragraph": i - 1
                    })
                # If the heading line itself contains a number, separate it (e.g. "# 1 Introduction"):
                m3 = _SECTION_RE.match(heading_line)
                if m3:
                    section_num, heading_text = m3.group(1), m3.group(2)
                else:
                    section_num, heading_text = "", heading_line
                current_section = {"section": section_num, "heading": heading_text, "content": []}
                current_start_idx = i
                continue
        # Check for an unnumbered heading using heuristics:
        if _is_likely_heading(para):
            if current_section["content"]:
                groups.append({
                    "section": current_section["section"],
                    "heading": current_section["heading"],
                    "content": "\n".join(current_section["content"]).strip(),
                    "start_paragraph": current_start_idx,
                    "end_paragraph": i - 1
                })
            current_section = {"section": "", "heading": para.strip(), "content": []}
            current_start_idx = i
            continue
        # Otherwise, this paragraph is content; add to current section's content.
        current_section["content"].append(para)
    # Flush the final section if any content remains
    if current_section["content"]:
        groups.append({
            "section": current_section["section"],
            "heading": current_section["heading"],
            "content": "\n".join(current_section["content"]).strip(),
            "start_paragraph": current_start_idx,
            "end_paragraph": len(paragraphs) - 1
        })
    return groups

def _yield_chunks_from_groups(groups: List[Dict], document_id: str, base_metadata: Dict,
                              target_tokens: int, overlap_tokens: int) -> Iterable[Chunk]:
    """
    Yield Chunk objects for each section group, splitting long sections into smaller overlapping chunks.
    Uses token-based windowing (target_tokens, overlap_tokens) for precise control of chunk size.
    """
    tokenizer = _get_tokenizer()
    for g in groups:
        text = g["content"]
        if not text:
            continue  # skip empty sections (no content)
        # Prepare base metadata for chunks in this section (include section title for citations):
        meta = {**(base_metadata or {}), "section": g["section"], "heading": g["heading"]}
        # If no tokenizer available (embedding model not loaded), fall back to simple char splitting
        if tokenizer is None:
            chunk_size_chars = target_tokens * 4  # approximate token->char ratio
            overlap_chars = overlap_tokens * 4
            start = 0
            while start < len(text):
                end = min(len(text), start + chunk_size_chars)
                chunk_text = text[start:end]
                chunk = Chunk(
                    id=f"{document_id}-{uuid.uuid4().hex[:8]}",
                    document_id=document_id,
                    text=chunk_text,
                    metadata=meta,
                    token_count=0,
                    start_paragraph=g.get("start_paragraph", 0),
                    end_paragraph=g.get("end_paragraph", 0)
                )
                yield chunk
                if end == len(text):
                    break
                # Slide window with overlap in characters
                start = max(end - overlap_chars, start + 1)
        else:
            # Tokenize the section text (without adding special tokens)
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            tokens = enc["input_ids"]
            offsets = enc["offset_mapping"]
            total_tokens = len(tokens)
            if total_tokens <= target_tokens:
                # The section fits in one chunk
                chunk = Chunk(
                    id=f"{document_id}-{uuid.uuid4().hex[:8]}",
                    document_id=document_id,
                    text=text,
                    metadata=meta,
                    token_count=total_tokens,
                    start_paragraph=g.get("start_paragraph", 0),
                    end_paragraph=g.get("end_paragraph", 0)
                )
                yield chunk
            else:
                # Slide a token window over the section text
                start_idx = 0
                while start_idx < total_tokens:
                    end_idx = min(total_tokens, start_idx + target_tokens)
                    if end_idx < total_tokens:
                        # Avoid cutting off in the middle of a word: if next token continues the word (no space), include it.
                        while end_idx < total_tokens and offsets[end_idx][0] == offsets[end_idx-1][1]:
                            end_idx += 1
                        end_idx = min(end_idx, total_tokens)
                    # Determine char span for tokens [start_idx, end_idx)
                    start_char = offsets[start_idx][0]
                    end_char = offsets[end_idx-1][1] if end_idx > start_idx else offsets[start_idx][1]
                    chunk_text = text[start_char:end_char]
                    chunk_token_count = end_idx - start_idx
                    chunk = Chunk(
                        id=f"{document_id}-{uuid.uuid4().hex[:8]}",
                        document_id=document_id,
                        text=chunk_text,
                        metadata=meta,
                        token_count=chunk_token_count,
                        start_paragraph=g.get("start_paragraph", 0),
                        end_paragraph=g.get("end_paragraph", 0)
                    )
                    yield chunk
                    if end_idx >= total_tokens:
                        break
                    # Set next start index with overlap
                    next_start = max(0, end_idx - overlap_tokens)
                    # Also ensure we start at a token that begins a word (not mid-word)
                    while next_start > 0 and offsets[next_start][0] == offsets[next_start-1][1]:
                        next_start -= 1
                    start_idx = next_start

def split_into_chunks(text: str, *, document_id: str, summarise: bool = False,
                      summariser: Optional[Callable[[str], str]] = None,
                      base_metadata: Optional[Dict] = None) -> List[Chunk]:
    """
    Split a cleaned document text into semantically meaningful chunks for RAG.
    - Uses document structure (sections/headings) to group content.
    - Uses token-based chunking (~config.MAX_CHUNK_TOKENS tokens per chunk with overlap) to preserve context.
    - Attaches section titles and source info in chunk.metadata for better citations.
    - Optionally generates a summary for each chunk using an LLM (stored in metadata/summary).
    """
    if not text:
        return []
    paragraphs = _split_into_paragraphs(text)
    groups = _group_by_section(paragraphs)
    chunks: List[Chunk] = list(_yield_chunks_from_groups(
        groups, document_id, base_metadata or {},
        target_tokens=config.MAX_CHUNK_TOKENS, overlap_tokens=config.CHUNK_OVERLAP
    ))
    # Filter out very small chunks (less than MIN_SECTION_LENGTH characters) as they are likely not informative
    if config.MIN_SECTION_LENGTH:
        chunks = [c for c in chunks if len(c.text) >= config.MIN_SECTION_LENGTH]
    # Optionally summarize each chunk’s text using the provided summariser (e.g. IBM Granite-Instruct via Ollama)
    if summarise and summariser:
        for c in chunks:
            try:
                summary_text = summariser(c.text)
                if summary_text:
                    c.summary = summary_text.strip()
                    c.metadata["summary"] = c.summary  # also store in metadata for reference
            except Exception as e:
                logger.warning("Summariser failed on chunk %s: %s", c.id, e)
    return chunks
