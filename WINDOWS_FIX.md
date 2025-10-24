# Fix for Windows: "No module named 'api'" Error

The error occurs because Python has cached bytecode from the old modules we deleted.

## Quick Fix (Run these commands in PowerShell in your project directory)

```powershell
# 1. Deactivate virtual environment if active
deactivate

# 2. Delete all Python cache files
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Force -Recurse
Get-ChildItem -Path . -Include *.pyc -Recurse -Force | Remove-Item -Force

# 3. Delete .env file if it exists (it has old settings)
if (Test-Path .env) { Remove-Item .env }

# 4. Recreate virtual environment (optional but recommended)
Remove-Item -Recurse -Force .venv
python -m venv .venv

# 5. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 6. Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 7. Create new .env file for Windows
@"
# Ollama in Docker on Windows
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3:8b

# Local data directory (Windows path - change to your preferred location)
DATA_DIR=C:\data\airgapped_rag

# API configuration
HOST=127.0.0.1
PORT=8000
"@ | Out-File -FilePath .env -Encoding UTF8

# 8. Run the API
python airgapped_rag.py
```

## Alternative: Quick Clean Command

If you just want to clean cache without recreating venv:

```powershell
# Clean all Python cache
Get-ChildItem -Path . -Include __pycache__,*.pyc -Recurse -Force | Remove-Item -Force -Recurse

# Activate venv
.\.venv\Scripts\Activate.ps1

# Run API
python airgapped_rag.py
```

## Using Git Bash (if you have it)

```bash
# Clean cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Run API
python airgapped_rag.py
```

## Verify Before Running

Make sure:
1. ✅ Ollama is running in Docker: `docker ps | findstr ollama`
2. ✅ Models are pulled: `docker exec ollama-airgapped-rag ollama list`
3. ✅ Virtual environment is activated: Your prompt shows `(.venv)`

## Expected Output After Fix

```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
============================================================
Air-Gapped RAG API Starting
============================================================
Data directory: C:\data\airgapped_rag
Documents indexed: 0
Ollama base URL: http://localhost:11434
Embedding model: nomic-embed-text
LLM model: llama3:8b
============================================================
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

## If Still Not Working

Check if you have an old Python path cached:

```powershell
# Check Python's sys.path
python -c "import sys; print('\n'.join(sys.path))"

# Should NOT show any 'api', 'ingestion', 'qa', or 'retrieval' paths
```

If you see old paths, restart your terminal/PowerShell completely.
