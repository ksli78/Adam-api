"""
Configuration for the ADAM RAG system.

All configurable values are collected in a single module to avoid the use of
environment variables and magic constants sprinkled throughout the code.  You
should review and customise these values before deploying your system.  For
example, you can point ``DOCS_DIR`` at a persistent volume on your
production server, or change the model names to use smaller quantised
variants that fit on your GPU.  This version of ADAM does not integrate
directly with SharePoint; use your own crawler to download PDFs and
submit them via the API.
"""

from pathlib import Path

###############################################################################
# Paths and storage
###############################################################################
# Directory where ingested documents and their metadata will be stored.  Make
# sure this location is persisted between container restarts in production.
DOCS_DIR: Path = Path("/data/docs").resolve()

# Path to the FAISS index file used for dense retrieval.  The index will be
# created automatically if it does not exist.
FAISS_INDEX_PATH: Path = DOCS_DIR / "faiss_index.bin"

# Path to the TF‑IDF index file used for lexical retrieval.  This will be
# created automatically and stored alongside the FAISS index.
TFIDF_INDEX_PATH: Path = DOCS_DIR / "tfidf_index.pkl"

# Optional directory where raw PDFs processed by the system will be cached.
# If set to ``None``, PDFs are processed from memory only.  By default we keep
# a copy of the original files under DOCS_DIR/pdfs.
PDF_CACHE_DIR: Path | None = DOCS_DIR / "pdfs"

###############################################################################
# Model names
###############################################################################
# Name of the embedding model on Hugging Face Hub.  The default uses the
# Granite embedding R2 English model released in September 2025, which
# generates 768‑dimensional vectors and supports inputs up to 8k tokens【449332267749196†L57-L83】.
EMBED_MODEL_NAME: str = "ibm-granite/granite-embedding-english-r2"

# Name of the cross‑encoder reranker model.  The default uses the Granite
# embedding reranker R2 model which jointly encodes query/document pairs to
# compute high quality relevance scores【449332267749196†L65-L70】.
RERANKER_MODEL_NAME: str = "ibm-granite/granite-embedding-reranker-english-r2"

# Name of the instruction‑tuned generative model used for answering
# questions.  Granite‑3.3‑8B‑Instruct supports a 128K context window and
# instruction following tasks【214111365068263†L50-L86】.  If your GPU cannot
# accommodate this model (6 GB VRAM is generally insufficient for 8 B models),
# consider using a smaller variant such as ``ibm-granite/granite-3.3-2b-instruct``
# or loading the model in 4‑bit quantisation.
INSTRUCT_MODEL_NAME: str = "ibm-granite/granite-3.3-8b-instruct"

###############################################################################
# Ingestion and chunking parameters
###############################################################################
# Maximum number of tokens per chunk passed to the embedder.  This should be
# conservative enough to avoid splitting sentences across boundaries but not so
# small that context is lost.  Granite embeddings allow up to 8k tokens, but
# smaller chunks improve retrieval precision and reduce memory requirements.
MAX_CHUNK_TOKENS: int = 800

# Number of tokens to overlap between consecutive chunks.  Overlap helps
# preserve context across boundaries but increases the number of vectors stored.
CHUNK_OVERLAP: int = 100

# When cleaning text, remove sections shorter than this many characters.  Very
# small fragments are unlikely to be informative and can add noise to the
# retrieval index.
MIN_SECTION_LENGTH: int = 50

# List of phrases to strip from all documents.  Headers, footers and legal
# warnings often repeat on every page and harm search quality.  Add your
# organisation’s boilerplate phrases here.
UNWANTED_PHRASES: list[str] = [
    "Do not copy or distribute",  # example
    "Confidential and proprietary",  # example
]

###############################################################################
# Retrieval parameters
###############################################################################
# Alpha controls the weighting between dense and lexical retrieval scores.  A
# value of 0.5 gives equal weight to vector similarity and TF‑IDF cosine
# similarity.  Set alpha closer to 1.0 to favour dense embeddings; closer to
# 0.0 to favour lexical search.
ALPHA: float = 0.5

# Number of top documents to return from the combined search before
# reranking.  The reranker further refines this set.
TOP_K: int = 10

# Number of top documents to return after reranking.  This should be <= TOP_K.
TOP_K_RERANKED: int = 5

###############################################################################
# Removed SharePoint configuration
###############################################################################
# This application no longer integrates directly with SharePoint.  If you wish
# to ingest documents from SharePoint, use an external crawler to download
# the PDFs and submit them via the upload or URL ingestion endpoints.

###############################################################################
# Miscellaneous
###############################################################################
# Seed for any random operations, such as embedding model initialisation.
RANDOM_SEED: int = 42

###############################################################################
# Helper functions
###############################################################################
def ensure_directories() -> None:
    """Create necessary directories if they don't already exist."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if PDF_CACHE_DIR is not None:
        PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TFIDF_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

# Ensure directories at import time when running as a script.  It is safe to
# call this multiple times.
ensure_directories()
