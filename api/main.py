"""
FastAPI application exposing ingestion, search and QA endpoints.

Use ``uvicorn adam_rag.api.main:app --host 0.0.0.0 --port 8000`` to run the
service.  The server maintains an in‑memory hybrid index which is
persisted to disk after each ingestion.  Concurrency is limited by the
Python GIL for CPU‑bound tasks; use multiple worker processes (e.g. via
gunicorn) to scale ingestion throughput.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import config
from ingestion.docling_parser import convert_pdf_to_markdown
from ingestion.text_cleaner import clean_markdown
from ingestion.chunker import split_into_chunks, Chunk
from ingestion.deduplication import deduplicate_chunks
from retrieval.index_store import HybridIndex, RetrievalResult
from retrieval.search import search_documents
from qa.answer import answer_question, generate_summary
from schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ADAM RAG API", version="0.1")

# Global in‑memory index
index = HybridIndex(dimension=768)


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    summarise: bool = Form(False),
) -> IngestResponse:
    """Ingest a PDF uploaded directly to the API."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    # Save uploaded file to a temporary location
    temp_path = config.PDF_CACHE_DIR or Path("/tmp")
    temp_path.mkdir(parents=True, exist_ok=True)
    local_file = temp_path / f"upload-{uuid.uuid4().hex}.pdf"
    with open(local_file, "wb") as f:
        content = await file.read()
        f.write(content)
    logger.info("Received uploaded PDF: %s", file.filename)
    return _process_document(local_file, summarise=summarise)


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(request: IngestRequest) -> IngestResponse:
    """Ingest a PDF specified by URL.  The PDF is downloaded using Docling."""
    url = request.url
    if not url.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="URL must point to a PDF file")
    return _process_document(url, summarise=request.summarise)


def _process_document(source: str | Path, *, summarise: bool) -> IngestResponse:
    """Internal helper to ingest a document and update the index."""
    # Convert PDF to markdown
    markdown = convert_pdf_to_markdown(source)
    cleaned = clean_markdown(markdown)
    document_id = uuid.uuid4().hex
    # Generate summaries on demand
    summariser_fn = generate_summary if summarise else None
    chunks = split_into_chunks(
        cleaned,
        document_id=document_id,
        summarise=summarise,
        summariser=summariser_fn,
    )
    # Deduplicate
    before = len(chunks)
    unique_chunks = deduplicate_chunks(chunks)
    duplicates_removed = before - len(unique_chunks)
    # Add to index
    index.add_chunks(unique_chunks, persist=True)
    message = f"Ingested {len(unique_chunks)} unique chunks from document."
    return IngestResponse(
        document_id=document_id,
        num_chunks=len(unique_chunks),
        duplicates_removed=duplicates_removed,
        message=message,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Answer a user question using the retrieval index and generative model."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    # Perform search and rerank
    results: List[RetrievalResult] = search_documents(index, req.question, top_k=req.top_k)
    # Generate answer
    answer_text = answer_question(req.question, results)
    citations = [res.chunk.id for res in results]
    return QueryResponse(answer=answer_text, citations=citations)


@app.get("/search")
async def search_endpoint(query: str, top_k: Optional[int] = None):
    """Return the top chunks matching a query without running the generative model."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results: List[RetrievalResult] = search_documents(index, query, top_k=top_k)
    return JSONResponse(
        [
            {
                "id": res.chunk.id,
                "score": res.score,
                "text": res.chunk.text[:500] + ("..." if len(res.chunk.text) > 500 else ""),
            }
            for res in results
        ]
    )
