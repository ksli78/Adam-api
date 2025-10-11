"""
Pydantic schemas for the FastAPI endpoints.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request body for document ingestion from a remote URL."""

    url: str = Field(..., description="URL of the PDF to ingest")
    summarise: bool = Field(
        default=False,
        description="Whether to generate summaries for each chunk during ingestion",
    )


class IngestResponse(BaseModel):
    """Response returned after document ingestion."""

    document_id: str
    num_chunks: int
    duplicates_removed: int
    message: str


class QueryRequest(BaseModel):
    """Request body for querying the knowledge base."""

    question: str
    top_k: Optional[int] = Field(
        default=None,
        description="Number of top results to consider during search and reranking",
    )


class QueryResponse(BaseModel):
    """Response returned after answering a question."""

    answer: str
    citations: List[str]
