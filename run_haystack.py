"""
Simple runner for Haystack-based RAG API
Avoids Windows multiprocessing issues by using single worker
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "airgapped_rag_haystack:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )
