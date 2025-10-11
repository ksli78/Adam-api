"""
Text cleaning utilities.

The quality of a retrieval system is highly sensitive to the quality of the
text it indexes.  Unwanted boilerplate text (e.g. headers, footers and
confidentiality warnings) can dominate similarity scores and drown out
relevant content.  This module implements a handful of simple cleaning
functions to strip such noise from markdown produced by Docling.

The functions defined here are intentionally conservative: they remove
lines containing configured unwanted phrases and common header/footer
patterns, collapse excessive whitespace, and filter out extremely short
sections.  You can customise the behaviour by editing
:data:`adam_rag.config.UNWANTED_PHRASES` or by extending the cleaner with
additional heuristics or regular expressions.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, List

import config

logger = logging.getLogger(__name__)


def _remove_unwanted_phrases(lines: Iterable[str]) -> List[str]:
    """Filter out lines containing any unwanted phrases defined in config."""
    unwanted_lower = [p.lower() for p in config.UNWANTED_PHRASES]
    filtered: List[str] = []
    for line in lines:
        line_lower = line.lower()
        if any(p in line_lower for p in unwanted_lower):
            continue
        filtered.append(line)
    return filtered


def _remove_common_headers_and_footers(lines: List[str]) -> List[str]:
    """Detect and remove lines that repeat frequently across the document.

    Many corporate documents include the same header or footer on every
    page (e.g. a file name, legal disclaimer, or date).  These lines can
    overpower the TF‑IDF statistics, causing irrelevant matches.  This
    function counts the frequency of each non‑blank line and removes those
    appearing in more than three pages.  Only lines shorter than 120
    characters are considered candidates for removal to avoid stripping
    meaningful paragraphs.
    """
    # Count frequencies ignoring leading/trailing whitespace
    freqs = Counter(l.strip() for l in lines if l.strip())
    # Identify repeated lines
    repeated = {
        line
        for line, count in freqs.items()
        if count > 3 and len(line) < 120
    }
    if repeated:
        logger.debug("Removing repeated header/footer lines: %s", list(repeated))
    return [l for l in lines if l.strip() not in repeated]


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of more than two blank lines into at most one blank line."""
    lines = text.splitlines()
    new_lines: List[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            # skip multiple consecutive blank lines
            if blank_count > 1:
                continue
        else:
            blank_count = 0
        new_lines.append(line)
    return "\n".join(new_lines)


def clean_markdown(markdown: str) -> str:
    """Perform basic cleaning of Docling‑produced markdown.

    Steps performed:

    1. Remove lines containing any phrases from :data:`config.UNWANTED_PHRASES`.
    2. Remove lines that repeat more than three times across the document
       (heuristic for headers/footers).
    3. Collapse excessive blank lines.
    4. Filter out tiny sections shorter than :data:`config.MIN_SECTION_LENGTH`.

    Parameters
    ----------
    markdown:
        The raw markdown string extracted from Docling.

    Returns
    -------
    str
        A cleaned markdown string ready for chunking.
    """
    lines = markdown.splitlines()
    lines = _remove_unwanted_phrases(lines)
    lines = _remove_common_headers_and_footers(lines)
    cleaned = "\n".join(lines)
    cleaned = _collapse_whitespace(cleaned)
    # Remove sections that are too short
    sections = [s.strip() for s in cleaned.split("\n\n") if s.strip()]
    long_sections = [s for s in sections if len(s) >= config.MIN_SECTION_LENGTH]
    return "\n\n".join(long_sections)
