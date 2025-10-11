"""
API package for the ADAM RAG system.

This package exposes a FastAPI application with endpoints for ingesting
documents, searching the index and answering questions.  To run the
application locally, execute ``uvicorn adam_rag.api.main:app --reload``.
"""

from .main import app

__all__ = ["app"]