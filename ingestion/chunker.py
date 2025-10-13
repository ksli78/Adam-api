# ingestion/chunker.py
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, List, Optional, Dict

@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: Dict

    def to_dict(self) -> dict:
        d = asdict(self)
        # keep metadata as-is
        return d

_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+)$")

def _split_into_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return parts

def _group_by_section(paragraphs: List[str]) -> List[Dict]:
    """
    Groups paragraphs under nearest preceding numbered heading like:
      '4.1 Continuation of Group Health Benefits - Subject to ...'
    Returns list of dicts with keys: section, heading, content
    """
    groups: List[Dict] = []
    current = {"section": "", "heading": "", "content": []}

    for p in paragraphs:
        m = _SECTION_RE.match(p)
        if m:
            # flush current block
            if current["content"]:
                groups.append({
                    "section": current["section"],
                    "heading": current["heading"],
                    "content": "\n".join(current["content"]).strip()
                })
            # start new section
            current = {
                "section": m.group(1),
                "heading": m.group(2),
                "content": []
            }
        else:
            current["content"].append(p)

    if current["content"]:
        groups.append({
            "section": current["section"],
            "heading": current["heading"],
            "content": "\n".join(current["content"]).strip()
        })
    return groups

def _yield_chunks_from_groups(groups: List[Dict], document_id: str, base_metadata: Dict,
                              target_chars: int = 1200, overlap_chars: int = 120) -> Iterable[Chunk]:
    """
    Creates chunks per section, further splitting long bodies to ~target_chars.
    Adds section/heading/document_title/source_url to metadata for citations.
    """
    for g in groups:
        body = g["content"]
        meta = {
            **(base_metadata or {}),
            "section": g["section"],
            "heading": g["heading"],
        }
        if len(body) <= target_chars:
            yield Chunk(
                id=f"{document_id}-{uuid.uuid4().hex[:8]}",
                document_id=document_id,
                text=body,
                metadata=meta,
            )
        else:
            # sliding window by characters (simple & robust)
            start = 0
            while start < len(body):
                end = min(len(body), start + target_chars)
                piece = body[start:end]
                yield Chunk(
                    id=f"{document_id}-{uuid.uuid4().hex[:8]}",
                    document_id=document_id,
                    text=piece,
                    metadata=meta,
                )
                if end == len(body):
                    break
                start = max(end - overlap_chars, start + 1)

def split_into_chunks(
    text: str,
    *,
    document_id: str,
    summarise: bool = False,
    summariser: Optional[Callable[[str], str]] = None,
    base_metadata: Optional[Dict] = None,
) -> List[Chunk]:
    """
    Splits cleaned text into section-aware chunks.
    - No domain hardcoding
    - Preserves section number & heading in metadata
    - Passes through document_title and source_url from base_metadata
    """
    paragraphs = _split_into_paragraphs(text)
    groups = _group_by_section(paragraphs)

    chunks: List[Chunk] = list(_yield_chunks_from_groups(groups, document_id, base_metadata))

    # Optionally attach summaries (stored in metadata only; text remains original)
    if summarise and summariser:
        for c in chunks:
            try:
                c.metadata["summary"] = summariser(c.text)
            except Exception:
                pass

    return chunks
