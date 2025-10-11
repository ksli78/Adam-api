"""
ADAM RAG Package
=================

This package implements a Retrieval‑Augmented Generation (RAG) system built
around IBM's Granite and Docling models.  It contains components for
document ingestion, text cleaning, chunking, embedding, indexing, search,
reranking, question answering, and integration with SharePoint.  All
configuration values live in :mod:`adam_rag.config`, so you never need to
set environment variables.  See the individual modules for detailed
documentation.

To start using the system, import the high level functions from
``api.main`` or call the ingestion and retrieval routines directly.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"
