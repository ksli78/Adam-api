# api/main.py
from __future__ import annotations

import os
import glob
import math
import logging
from typing import Any, List, Optional, cast

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config

# Retrieval & QA
import retrieval.index_store as _idx
from retrieval.search import HybridIndex
from retrieval.search import search_documents, RetrievalResult
from retrieval.reranker import Reranker
from qa.answer import answer_question

# Optional warm function if present
try:
    from qa.answer import warm_models  # type: ignore
except Exception:  # pragma: no cover
    warm_models = None  # type: ignore




logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(config, "LOG_LEVEL", logging.INFO))




app = FastAPI(title="Adam API", version="1.0.0")

# --- CORS (adjust if you have stricter settings) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(config, "CORS_ALLOW_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ---------- Models ----------
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


# ---------- Helpers ----------
def _get_index():
    """
    Robustly obtain the index instance from retrieval.index_store, regardless of
    the specific helper your repo exposes.
    Tries common function and class patterns in a safe order.
    """
    # Prefer module-level helpers if they exist
    for name in ("get_index", "load_index", "open_index", "get_or_create_index"):
        fn = getattr(_idx, name, None)
        if callable(fn):
            return fn()

    # Fall back to class-based APIs
    cls = getattr(_idx, "IndexStore", None)
    if cls is not None:
        # Try common classmethods/constructors
        for name in ("get", "load", "open", "get_or_create"):
            meth = getattr(cls, name, None)
            if callable(meth):
                return meth()
        try:
            # last resort: instantiate
            return cls()
        except Exception:
            pass

    raise RuntimeError(
        "Could not obtain index: retrieval.index_store does not expose "
        "get_index/load_index/open_index/get_or_create_index or IndexStore."
    )
def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return v

def _pref_score(r: Any) -> float:
    """
    Prefer r.rerank_score if present, else r.score; always return a finite float.
    Supports both dicts and objects.
    """
    if isinstance(r, dict):
        s = r.get("rerank_score")
        if s is None:
            s = r.get("score", 0.0)
        return _safe_float(s)
    s = getattr(r, "rerank_score", None)
    if s is None:
        s = getattr(r, "score", 0.0)
    return _safe_float(s)

def _to_retrieval_result(r: Any) -> RetrievalResult:
    """
    Normalize dict/object from retrieval/rerank into RetrievalResult
    while adapting to your dataclass constructor signature dynamically.
    """
    # Extract common fields safely
    text = ""
    metadata = {}
    score = _pref_score(r)
    chunk = None

    if isinstance(r, dict):
        text = str(r.get("text") or r.get("chunk_text") or "")
        metadata = r.get("metadata") or {}
        chunk = r.get("chunk")  # some repos keep a chunk object
    else:
        text = str(getattr(r, "text", None) or getattr(r, "chunk_text", "") or "")
        metadata = getattr(r, "metadata", None) or {}
        chunk = getattr(r, "chunk", None)

    # now introspect RetrievalResult’s constructor args
    import inspect
    sig = inspect.signature(RetrievalResult)
    params = sig.parameters

    kwargs = {}

    if "id" in params:
        kwargs["id"] = getattr(r, "id", None) or r.get("id") if isinstance(r, dict) else ""
    if "chunk_id" in params:
        kwargs["chunk_id"] = getattr(r, "chunk_id", None) or (r.get("chunk_id") if isinstance(r, dict) else "")
    if "chunk" in params:
        kwargs["chunk"] = chunk
    if "text" in params:
        kwargs["text"] = text
    if "metadata" in params:
        kwargs["metadata"] = metadata
    if "score" in params:
        kwargs["score"] = score

    # Fill any missing required params with sane defaults
    for name, param in params.items():
        if name not in kwargs:
            if param.default is inspect.Parameter.empty:
                # required param with no default
                if param.annotation == float:
                    kwargs[name] = 0.0
                elif param.annotation == dict:
                    kwargs[name] = {}
                else:
                    kwargs[name] = None

    return RetrievalResult(**kwargs)

# ---------- Startup ----------
@app.on_event("startup")
async def _startup() -> None:
    logger.info("Waiting for application startup.")
    # Warm index lazily; creation happens on first get_index() call
    try:
        _ = _get_index()
    except Exception as e:
        logger.warning("Index not ready on startup: %s", e)
    # Warm answer models if function is available
    try:
        if callable(warm_models):  # type: ignore
            logger.info("Warming QA models…")
            warm_models()  # type: ignore
            logger.info("Answer models warmed.")
    except Exception as e:
        logger.warning("Warmup failed (continuing): %s", e)
    logger.info("Application startup complete.")

# ---------- Health ----------
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# ---------- Admin: Clear indexes ----------
@app.post("/admin/clear-index", tags=["admin"])
async def clear_index():
    """
    Completely clear all local retrieval indexes (FAISS + TF-IDF) and metadata.
    After this, you must re-ingest documents.
    """
    deleted: List[str] = []
    try:
        paths = []
        if hasattr(config, "FAISS_INDEX_PATH"):
            paths.append(config.FAISS_INDEX_PATH)
        if hasattr(config, "TFIDF_INDEX_PATH"):
            paths.append(config.TFIDF_INDEX_PATH)

        all_files: List[str] = []
        for p in paths:
            base, _ext = os.path.splitext(p)
            # Collect base.* (index + sidecars like .meta.json/.json)
            all_files.extend(glob.glob(f"{base}*"))

        for f in set(all_files):
            if os.path.exists(f):
                os.remove(f)
                deleted.append(os.path.basename(f))

        logger.info("Cleared index files: %s", deleted or "none")
        return {"message": "Index cleared successfully.", "deleted_files": deleted}
    except Exception as e:
        logger.exception("Failed to clear index: %s", e)
        raise HTTPException(status_code=500, detail=f"Error clearing index: {e}")

# ---------- Query ----------
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Retrieve, robustly rerank, adapt results, and generate a grounded answer.
    """
    try:
        q = (req.question or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Question is required.")

        index: HybridIndex = cast(HybridIndex, _get_index())

        # 1) Initial retrieval
        raw_top_k = req.top_k or getattr(config, "TOP_K", 12)
        results: List[Any] = search_documents(index, q, top_k=raw_top_k)

        # 2) Cross-encoder rerank (keeps original order if unusable)
        try:
            reranker = Reranker()
            keep_k = getattr(config, "TOP_K_RERANKED", 6)
            results = reranker.rank(q, results, top_k=keep_k)
        except Exception as e:
            logger.exception("Reranker unavailable/failed: %s. Proceeding without rerank.", e)

        # 3) Filter weak matches ONLY if there is real signal (avoid emptying list)
        if results:
            max_score = max((_pref_score(r) for r in results), default=0.0)
            if max_score > 0.0:
                min_keep = getattr(config, "MIN_RERANK_SCORE", 0.15)
                filtered = [r for r in results if _pref_score(r) >= min_keep]
                if filtered:
                    results = filtered

        # 4) Normalize to RetrievalResult for answer_question
        norm_results: List[RetrievalResult] = [_to_retrieval_result(r) for r in results]

        # 5) Generate grounded answer
        # NOTE: in your repo, answer_question returns (answer_text: str, citations: List[dict])
        answer_text, citations = answer_question(q, norm_results)

        # 6) Ensure citation scores are JSON-safe finite floats
        safe_citations: List[dict] = []
        for c in citations or []:
            try:
                sc = _safe_float(c.get("score", 0.0))
            except Exception:
                sc = 0.0
            c = dict(c)
            c["score"] = sc
            safe_citations.append(c)

        return QueryResponse(answer=answer_text, citations=safe_citations)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ---------- (Optional) Ingest endpoint ----------
# Keep this minimal to avoid fighting with your existing ingestion flow.
# If you already have a richer /ingest/upload, delete this block.
@app.post("/ingest/upload", tags=["ingest"])
async def ingest_upload(file: UploadFile = File(...), summarise: bool = False):
    """
    Minimal passthrough upload. Your existing ingestion pipeline likely overrides this.
    Left here only so the app works out-of-the-box if needed.
    """
    try:
        name = file.filename or "uploaded"
        logger.info("Received uploaded file: %s", name)
        # If your repository already has a full ingestion workflow, call into it here.
        # Otherwise, just persist the file so your offline job/another endpoint can consume it.
        out_dir = getattr(config, "UPLOAD_DIR", "./uploads")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, name)
        with open(out_path, "wb") as f:
            f.write(await file.read())
        return {"message": "Uploaded", "filename": name, "path": out_path, "summarise": summarise}
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Local run ----------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
