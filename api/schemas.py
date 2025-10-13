# api/schemas.py
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, HttpUrl


class IngestRequest(BaseModel):
    url: str
    summarise: bool = False


class IngestResponse(BaseModel):
    document_id: str
    num_chunks: int
    duplicates_removed: int
    message: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 8


class Citation(BaseModel):
    id: str                               # chunk id
    score: float
    excerpt: str
    section: Optional[str] = None
    heading: Optional[str] = None
    document_title: Optional[str] = None
    source_url: Optional[str] = None      # URL or file:// path, if available


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
