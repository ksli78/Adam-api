# api/main.py
"""
FastAPI application exposing ingestion, search and QA endpoints.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import os
import glob
import math
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import config
from ingestion.docling_parser import convert_pdf_to_markdown
from ingestion.text_cleaner import clean_markdown
from ingestion.chunker import split_into_chunks
from ingestion.deduplication import deduplicate_chunks
from retrieval.index_store import HybridIndex, RetrievalResult
from retrieval.search import search_documents
from qa.answer import answer_question, generate_summary, warm_answer_models
from .schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, Citation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ADAM RAG API", version="0.4")

# Global in-memory index
index = HybridIndex(dimension=768)


@app.on_event("startup")
async def _startup() -> None:
    try:
        warm_answer_models()
        logger.info("Startup warm completed.")
    except Exception as e:
        logger.warning("Warm-up failed: %s", e)


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    summarise: bool = Form(False),
    # canonical source (e.g., SharePoint URL). Optional; if omitted we store file:// path.
    source_url: Optional[str] = Form(None),
) -> IngestResponse:
    """Ingest a PDF uploaded directly to the API."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    cache_dir = config.PDF_CACHE_DIR or Path("./.pdf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_file = cache_dir / f"upload-{uuid.uuid4().hex}.pdf"
    with open(local_file, "wb") as f:
        f.write(await file.read())

    logger.info("Received uploaded PDF: %s", file.filename)

    base_metadata = {
        "document_title": file.filename,
        "source_url": source_url or local_file.resolve().as_uri(),
    }
    return _process_document(local_file, summarise=summarise, base_metadata=base_metadata)


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(request: IngestRequest) -> IngestResponse:
    """Ingest a PDF specified by URL. The PDF is downloaded using Docling."""
    url = request.url
    if not url.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="URL must point to a PDF file")
    base_metadata = {
        "document_title": Path(url).name or "document",
        "source_url": url,
    }
    return _process_document(url, summarise=request.summarise, base_metadata=base_metadata)


def _process_document(source: str | Path, *, summarise: bool, base_metadata: dict) -> IngestResponse:
    """Internal helper to ingest a document and update the index."""
    markdown = convert_pdf_to_markdown(source)
    cleaned = clean_markdown(markdown)
    document_id = uuid.uuid4().hex

    summariser_fn = generate_summary if summarise else None
    chunks = split_into_chunks(
        cleaned,
        document_id=document_id,
        summarise=summarise,
        summariser=summariser_fn,
        base_metadata=base_metadata,  # ensure title + source_url propagate to all chunks
    )

    before = len(chunks)
    unique_chunks = deduplicate_chunks(chunks)
    duplicates_removed = before - len(unique_chunks)

    index.add_chunks(unique_chunks, persist=True)
    logger.info("Saved index with %d chunks (removed %d duplicates).", len(unique_chunks), duplicates_removed)

    return IngestResponse(
        document_id=document_id,
        num_chunks=len(unique_chunks),
        duplicates_removed=duplicates_removed,
        message=f"Ingested {len(unique_chunks)} unique chunks from document.",
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Answer a user question using retrieval + grounded generation."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    results: List[RetrievalResult] = search_documents(index, req.question, top_k=req.top_k)
    answer_text, cites = answer_question(req.question, results)

    citations_out: List[Citation] = []
    for c in cites:
        raw_score = c.get("score", 0.0)
        try:
            score_val = float(raw_score)
        except Exception: 
            score_val = 0.0 

        if not math.isfinite(score_val):
            logging.warning(f"Invalid score {raw_score} for id ={c.get('id')}, replaceing with 0.0")
            score_val = 0.0 

        citations_out.append(
            Citation(
                id=c.get("id", ""),
                score=score_val,
                excerpt=c.get("excerpt", "")[:500],
                section=c.get("section"),
                heading=c.get("heading"),
                document_title=c.get("document_title"),
                source_url=c.get("source_url"),
            )
        )

    return QueryResponse(answer=answer_text, citations=citations_out)


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
                "section": res.chunk.metadata.get("section", ""),
                "heading": res.chunk.metadata.get("heading", ""),
                "document_title": res.chunk.metadata.get("document_title", ""),
                "source_url": res.chunk.metadata.get("source_url", ""),
                "text": res.chunk.text[:500] + ("..." if len(res.chunk.text) > 500 else ""),
            }
            for res in results
        ]
    )

@app.post("/admin/clear-index", tags=["admin"])
async def clear_index():
    """
    Completely clear all local retrieval indexes (FAISS + TF-IDF) and metadata.
    Use this only if you want to rebuild from scratch.
    """
    deleted_files = []

    try:
        # List of possible index paths from config
        index_paths = []
        if hasattr(config, "FAISS_INDEX_PATH"):
            index_paths.append(config.FAISS_INDEX_PATH)
        if hasattr(config, "TFIDF_INDEX_PATH"):
            index_paths.append(config.TFIDF_INDEX_PATH)

        # Expand to include .meta.json, .json variants
        all_files = []
        for path in index_paths:
            base = os.path.splitext(path)[0]
            all_files += glob.glob(f"{base}*")

        for f in all_files:
            if os.path.exists(f):
                os.remove(f)
                deleted_files.append(os.path.basename(f))

        # Log what was deleted
        if deleted_files:
            logger.info("Cleared index files: %s", deleted_files)
        else:
            logger.info("No index files found to delete.")

        return {
            "message": "Index cleared successfully.",
            "deleted_files": deleted_files,
        }

    except Exception as e:
        logger.exception("Failed to clear index: %s", e)
        raise HTTPException(status_code=500, detail=f"Error clearing index: {e}")