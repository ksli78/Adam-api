# ingestion/text_cleaner.py
from __future__ import annotations

import re
import unicodedata

HEADER_FOOTER_PATTERNS = [
    r"^This document contains proprietary information.*$",   # header/footer boilerplate
    r"^Uncontrolled if printed\..*$",
    r"^Before using this document.*$",
    r"^Copyright © \d{4} .* All rights reserved\.$",
    r"^Management System\s+Policy$",
    r"^Document No\.\s*:\s*.*$",
    r"^Page:\s*\d+\s+of\s+\d+\s*$",
    r"^Effective Date:\s*.*$",
    r"^Revision:\s*.*$",
    r"^Accountable Organization:.*$",
    r"^Change/revision Date.*$",
    r"^Revision authorized by .*",
]

HEADER_FOOTER_RE = re.compile("|".join(HEADER_FOOTER_PATTERNS), re.IGNORECASE)

def _normalize_unicode(s: str) -> str:
    # Normalize unicode; convert fancy dashes/quotes to ASCII equivalents
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # en/em dashes -> hyphen
    s = s.replace("\u2212", "-")  # minus
    s = s.replace("\u00b7", "-")  # middle dot
    s = s.replace("\u2022", "-")  # bullet
    s = s.replace("•", "-")
    return s

def _drop_headers_footers(lines):
    out = []
    for ln in lines:
        if HEADER_FOOTER_RE.search(ln.strip()):
            continue
        out.append(ln)
    return out

def clean_markdown(md: str) -> str:
    """
    Conservative cleaner:
    - normalize unicode
    - drop known headers/footers
    - keep numbered sections like '4.1 Continuation of Group Health Benefits—Subject ...'
    - collapse extra whitespace, but preserve line breaks between paragraphs
    """
    if not md:
        return ""

    md = _normalize_unicode(md)

    # Normalize whitespace
    md = md.replace("\r", "")
    lines = [ln.strip() for ln in md.split("\n")]

    # Drop boilerplate headers/footers
    lines = _drop_headers_footers(lines)

    # Remove empty line bursts while preserving paragraph boundaries
    cleaned = []
    last_blank = False
    for ln in lines:
        ln = re.sub(r"\s+", " ", ln).strip()
        if not ln:
            if not last_blank:
                cleaned.append("")  # keep single blank as paragraph break
            last_blank = True
            continue
        last_blank = False
        cleaned.append(ln)

    text = "\n".join(cleaned)

    # Ensure section headings remain on their own lines
    # e.g., "4.1 Continuation of Group Health Benefits - Subject to ..."
    # Keep the hyphen and words so tokenization finds 'health'/'benefits'
    return text.strip()
