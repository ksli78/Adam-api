"""
Development server runner for Windows.

This avoids multiprocessing issues on Windows that can cause
"No module named 'api'" errors with cached imports.

Usage:
    python run_dev.py
"""

import uvicorn

if __name__ == "__main__":
    # Run without reload to avoid multiprocessing on Windows
    uvicorn.run(
        "airgapped_rag:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # Disable reload to avoid Windows subprocess issues
        workers=1,
        log_level="info"
    )
