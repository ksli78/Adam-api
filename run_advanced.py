"""
Runner for Advanced RAG System

Simple wrapper to run the advanced RAG API with proper configuration.
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # CRITICAL: Force unbuffered output for streaming responses
    # This ensures SSE events are sent immediately, not buffered
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=False)

    # Set environment variables if not already set
    os.environ.setdefault("DATA_DIR", "/data/airgapped_rag")
    os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
    os.environ.setdefault("LLM_MODEL", "llama3:8b")

    print("=" * 60)
    print("Advanced Air-Gapped RAG System v2.0")
    print("=" * 60)
    print(f"Data directory: {os.environ['DATA_DIR']}")
    print(f"Ollama host: {os.environ['OLLAMA_HOST']}")
    print(f"LLM model: {os.environ['LLM_MODEL']}")
    print("=" * 60)
    print("\nStarting server on http://0.0.0.0:8000")
    print("API docs: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(
        "airgapped_rag_advanced:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
