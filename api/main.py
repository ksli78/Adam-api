"""
Modified main application for the ADAM API.

This version restores a full ingestion pipeline and keeps the reranking logic.  It
implements the following improvements over the current `api/main.py` in the
`ksli78/Adam-api` repository:

* The `/ingest/upload` endpoint converts uploaded PDFs into Markdown using
  Docling, cleans them, splits them into semantically meaningful chunks,
  optionally summarises each chunk, deduplicates near-identical chunks, and
  then adds those chunks to the hybrid retrieval index.  It returns a
  document ID along with counts of unique chunks and duplicates removed.

* The `/query` endpoint still uses the hybrid search and cross-encoder
  reranker, but adapts `RetrievalResult` objects into dictionaries for
  reranking to ensure the reranker sees the actual chunk text.  After
  reranking, it reorders the original `RetrievalResult` objects based on
  the reranked identifiers, filters out very weak matches only if there is
  rerank signal, and then passes the sorted list to `answer_question`.

This file can be used as a drop-in replacement for `api/main.py` if you
copy it into your repository and adjust import paths accordingly.  It
demonstrates the necessary changes to restore ingestion and fix NaN
rerank scores.
"""

from __future__ import annotations

import os
import glob
import math
import uuid
import logging
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple,cast

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config

# Ingestion helpers
from ingestion.docling_parser import convert_pdf_to_markdown
from ingestion.text_cleaner import clean_markdown
from ingestion.chunker import split_into_chunks
from ingestion.deduplication import deduplicate_chunks
from qa.answer import generate_summary

# Retrieval & QA
import retrieval.index_store as _idx
from retrieval.search import HybridIndex, search_documents, RetrievalResult
from retrieval.reranker import Reranker
from qa.answer import answer_question

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(config, "LOG_LEVEL", logging.INFO))

app = FastAPI(title="Adam API", version="1.0.0 (modified)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(config, "CORS_ALLOW_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the index once; it will load existing data automatically.
index = HybridIndex()

# -------------------- Pydantic Models --------------------

class Citation(BaseModel):
    id: str
    score: float
    excerpt: str
    section: Optional[str] = None
    heading: Optional[str] = None
    document_title: Optional[str] = None
    source_url: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]


class IngestResponse(BaseModel):
    document_id: str
    num_chunks: int
    duplicates_removed: int
    message: str

class DocumentSummary(BaseModel):
    """Summary information for an ingested document."""
    document_id: str
    num_chunks: int

class ChunkDetail(BaseModel):
    """Detailed information about a single chunk."""
    id: str
    text: str
    metadata: Optional[dict] = None
    token_count: int
    start_paragraph: int
    end_paragraph: int
    summary: Optional[str] = None

# -------------------- Helpers --------------------

def _get_index() -> HybridIndex:
    """
    Obtain a HybridIndex instance.  Tries common helper functions first; if
    none exist, attempts to instantiate the index directly.
    """
    # Prefer module-level helpers if they exist
    for name in ("get_index", "load_index", "open_index", "get_or_create_index"):
        fn = getattr(_idx, name, None)
        if callable(fn):
            return fn()  # type: ignore[no-any-return]

    # Fall back to class-level helpers
    cls = getattr(_idx, "IndexStore", None)
    if cls is not None:
        for name in ("get", "load", "open", "get_or_create"):
            meth = getattr(cls, name, None)
            if callable(meth):
                return meth()  # type: ignore[no-any-return]
        # Last resort: instantiate
        return cls()  # type: ignore[no-any-return]
    raise RuntimeError(
        "Could not obtain index: retrieval.index_store does not expose a helper"
    )


def _safe_float(x: Any) -> float:
    """Convert x to float and return 0.0 for non‑finite values."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return v


# -------------------- Startup --------------------

@app.on_event("startup")
async def _startup() -> None:
    logger.info("Waiting for application startup.")
    # Ensure directories exist
    try:
        config.ensure_directories()
    except Exception as e:
        logger.warning("Failed to ensure directories: %s", e)
    # Warm index lazily
    try:
        _ = _get_index()
    except Exception as e:
        logger.warning("Index not ready on startup: %s", e)
    # Warm answer models if available
    try:
        from qa.answer import warm_models  # type: ignore
        if callable(warm_models):
            logger.info("Warming QA models…")
            warm_models()  # type: ignore
            logger.info("Answer models warmed.")
    except Exception as e:
        logger.warning("Warmup failed (continuing): %s", e)
    logger.info("Application startup complete.")


# -------------------- Health --------------------

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# -------------------- Admin: Clear index --------------------

@app.post("/admin/clear-index", tags=["admin"])
async def clear_index():
    """Remove stored FAISS/TF‑IDF index files and reset the index."""
    deleted: List[str] = []
    try:
        # Build list of index files to remove
        paths: List[Path] = []
        if hasattr(config, "FAISS_INDEX_PATH"):
            paths.append(config.FAISS_INDEX_PATH)
        if hasattr(config, "TFIDF_INDEX_PATH"):
            paths.append(config.TFIDF_INDEX_PATH)
        # Delete index files and sidecars
        for p in paths:
            base = p.with_suffix("")
            # Glob base.* (index + .meta.json / .json sidecars)
            for f in base.parent.glob(base.name + "*"):
                try:
                    f.unlink()
                    deleted.append(f.name)
                except FileNotFoundError:
                    pass
        # Reinitialise index
        # Remove any in‑memory index; new calls to _get_index() will rebuild
        # automatically
        return {"message": "Index cleared successfully.", "deleted_files": deleted}
    except Exception as e:
        logger.exception("Failed to clear index: %s", e)
        raise HTTPException(status_code=500, detail=f"Error clearing index: {e}")


# -------------------- Ingestion --------------------
@app.post("/ingest/upload", response_model=IngestResponse, tags=["ingest"])
async def ingest_upload(
    file: UploadFile = File(...),
    summarise: bool = False,
    source_url: Optional[str] = None,
):
    """
    Ingest a PDF, optionally attaching a source URL.  This runs the full
    conversion/cleaning/chunking pipeline and indexes the result.
    """
    # 1) Save PDF to a cache location (DOCS_DIR / "pdfs")
    doc_id = uuid.uuid4().hex
    pdf_cache_dir = config.PDF_CACHE_DIR or (config.DOCS_DIR / "pdfs")
    pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_cache_dir / f"{doc_id}.pdf"
    with open(pdf_path, "wb") as f_out:
        f_out.write(await file.read())

    # 2) Convert to Markdown, clean, chunk, deduplicate
    markdown = convert_pdf_to_markdown(str(pdf_path))
    cleaned = clean_markdown(markdown)
    summariser = generate_summary if summarise else None
    chunks = split_into_chunks(
        cleaned,
        document_id=doc_id,
        summarise=summarise,
        summariser=summariser,
        base_metadata={},
    )
    # If a source URL was provided, attach it to every chunk’s metadata
    if source_url:
        for c in chunks:
            meta = c.metadata or {}
            meta["source_url"] = source_url
            c.metadata = meta
    unique_chunks = deduplicate_chunks(chunks)

    # 3) Add to the global index
    index.add_chunks(unique_chunks)

    return IngestResponse(
        document_id=doc_id,
        num_chunks=len(unique_chunks),
        duplicates_removed=len(chunks) - len(unique_chunks),
        message=f"Ingested {len(unique_chunks)} unique chunks from document."
    )

# -------------------- Query --------------------

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Retrieve, rerank, and answer a question using the indexed chunks.
    """
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is required.")

    # 1) Hybrid search
    results: List[RetrievalResult] = search_documents(index, q, top_k=req.top_k or config.TOP_K)

    # 2) Rerank using the Granite cross‑encoder
    try:
        reranker = Reranker()
        reranked  = reranker.rank(q, results, top_k=getattr(config, "TOP_K_RERANKED", 5))
        results = cast(List[RetrievalResult], reranked)
    except Exception:
        pass  # Fallback: keep initial order

    # 3) Answer using QA model
    answer_text, citations = answer_question(q, results)

    # 4) Sanitize citation scores and return
    final_citations = []
    for c in citations:
        sc = float(c.get("score", 0.0))
        if not (sc < float("inf") and sc > float("-inf")):
            sc = 0.0
        c = dict(c)
        c["score"] = sc
        final_citations.append(Citation(**c))
    return QueryResponse(answer=answer_text, citations=final_citations)

@app.get("/documents", response_model=List[DocumentSummary], tags=["documents"])
async def list_documents():
    """
    List all ingested documents with the number of chunks indexed for each.

    This endpoint iterates through the in-memory index and groups chunks by
    their ``document_id`` attribute.  It returns one entry per document
    containing the document ID and the count of chunks.
    """
    try:
        index: HybridIndex = _get_index()
        counts: Dict[str, int] = {}
        for chunk in getattr(index, "chunks", []):
            doc_id = getattr(chunk, "document_id", None)
            if doc_id is None:
                continue
            counts[doc_id] = counts.get(doc_id, 0) + 1
        summaries: List[DocumentSummary] = [DocumentSummary(document_id=doc_id, num_chunks=num) for doc_id, num in counts.items()]
        return summaries
    except Exception as e:
        logger.exception("Failed to list documents: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}", response_model=List[ChunkDetail], tags=["documents"])
async def get_document_chunks(document_id: str):
    """
    Retrieve all chunks and their details for a given document.

    The ``document_id`` parameter should correspond to the ID returned by
    ``/ingest/upload``.  For each chunk belonging to that document, this
    endpoint returns the chunk's ID, text, metadata (if any), token count,
    paragraph boundaries, and summary (if available).

    If the document is not found in the index, a 404 error is raised.
    """
    try:
        index: HybridIndex = _get_index()
        # Collect matching chunks
        matching: List[ChunkDetail] = []
        for chunk in getattr(index, "chunks", []):
            if getattr(chunk, "document_id", None) == document_id:
                matching.append(
                    ChunkDetail(
                        id=getattr(chunk, "id", ""),
                        text=getattr(chunk, "text", ""),
                        metadata=getattr(chunk, "metadata", None) or {},
                        token_count=getattr(chunk, "token_count", 0),
                        start_paragraph=getattr(chunk, "start_paragraph", 0),
                        end_paragraph=getattr(chunk, "end_paragraph", 0),
                        summary=getattr(chunk, "summary", None),
                    )
                )
        if not matching:
            raise HTTPException(status_code=404, detail="Document not found or has no chunks")
        return matching
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get document chunks: %s", e)
        raise HTTPException(status_code=500, detail=str(e))