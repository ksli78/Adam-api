# Step-by-Step Fix for "No module named 'api'" Error

## The Problem

Uvicorn is trying to import the old `api` module. This could be caused by:
1. Cached Python bytecode
2. Old `.env` file with incorrect settings
3. VS Code launch configuration
4. Running the wrong command

## Complete Fix (Step-by-Step)

### Step 1: Stop Everything

```powershell
# Close VS Code completely
# Close all PowerShell/terminal windows
# This ensures no cached processes
```

### Step 2: Clean Your Project Directory

Open a NEW PowerShell window in your project directory:

```powershell
# Navigate to project
cd D:\Projects\Web\AdamIBM

# Delete ALL cache files recursively
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Include *.pyc -Recurse -Force | Remove-Item -Force -ErrorAction SilentlyContinue

# Delete .env file (it might have old settings)
Remove-Item .env -ErrorAction SilentlyContinue

# Delete VS Code settings that might cache paths
Remove-Item .vscode -Recurse -Force -ErrorAction SilentlyContinue
```

### Step 3: Verify Files Don't Exist

```powershell
# These should return nothing or "Path does not exist"
Test-Path .\api
Test-Path .\ingestion
Test-Path .\qa
Test-Path .\retrieval
Test-Path .\config.py

# If any return True, you need to pull the latest code
git status
git pull origin claude/air-gapped-rag-api-011CUSV9tEGJ19GvsJ3pgn4j
```

### Step 4: Check Your Virtual Environment

```powershell
# Deactivate if active
deactivate

# OPTION A: Clean reinstall (recommended)
Remove-Item .venv -Recurse -Force -ErrorAction SilentlyContinue
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# OPTION B: If you want to keep your venv, just reinstall
.\.venv\Scripts\Activate.ps1
pip install --force-reinstall -r requirements.txt
```

### Step 5: Verify Ollama is Running

```powershell
# Check Docker
docker ps

# If no ollama container, start it
docker-compose -f docker-compose.ollama.yml up -d

# Pull models if needed
docker exec ollama-airgapped-rag ollama pull nomic-embed-text
docker exec ollama-airgapped-rag ollama pull llama3:8b

# Verify models
docker exec ollama-airgapped-rag ollama list
```

### Step 6: Run the API Correctly

**IMPORTANT: Run it this way:**

```powershell
# Make sure venv is activated
.\.venv\Scripts\Activate.ps1

# Run directly with Python (NOT uvicorn command)
python airgapped_rag.py
```

**DO NOT run these commands:**
```powershell
# ❌ WRONG - Don't use these
uvicorn api.main:app  # Old module
uvicorn main:app      # Wrong file
```

### Step 7: If It Still Fails

Check what command is actually being run:

```powershell
# Show the exact Python being used
Get-Command python | Format-List *

# Check if there are any lingering imports
python -c "import sys; [print(p) for p in sys.path if 'api' in p.lower()]"

# If output shows paths with 'api', your Python path is corrupted
# Solution: Restart PowerShell completely
```

## Alternative: Run Without Uvicorn Subprocess

If the multiprocessing is causing issues on Windows, modify the startup:

Create a file `run_dev.py`:

```python
import uvicorn

if __name__ == "__main__":
    # Run without reload and without subprocess
    uvicorn.run(
        "airgapped_rag:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1
    )
```

Then run:
```powershell
python run_dev.py
```

## VS Code Debug Configuration

If you're running from VS Code, update `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Air-Gapped RAG API",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/airgapped_rag.py",
      "console": "integratedTerminal",
      "python": "${workspaceFolder}/.venv/Scripts/python.exe",
      "env": {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "DATA_DIR": "C:/data/airgapped_rag"
      },
      "justMyCode": true
    }
  ]
}
```

## Nuclear Option: Complete Reset

If nothing else works:

```powershell
# 1. Navigate OUT of the project
cd ..

# 2. Delete the entire folder
Remove-Item -Recurse -Force AdamIBM

# 3. Clone fresh
git clone <your-repo-url> AdamIBM
cd AdamIBM

# 4. Checkout the correct branch
git checkout claude/air-gapped-rag-api-011CUSV9tEGJ19GvsJ3pgn4j

# 5. Create new venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 6. Start Ollama
docker-compose -f docker-compose.ollama.yml up -d

# 7. Run API
python airgapped_rag.py
```

## Expected Success Output

```
INFO:     Started server process [12345]
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
⚠ Ollama connection failed: [Errno 10061] ...
  Make sure Ollama is running and accessible
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

(The Ollama warning is OK if you haven't started it yet)

## Quick Checklist

Before running `python airgapped_rag.py`:

- [ ] Git pull completed
- [ ] No `api/`, `ingestion/`, `qa/`, `retrieval/` folders exist
- [ ] Virtual environment activated (`(.venv)` in prompt)
- [ ] `pip list` shows chromadb, ollama, fastapi installed
- [ ] Ollama container running: `docker ps | findstr ollama`
- [ ] Current directory is project root (contains `airgapped_rag.py`)
- [ ] Running command: `python airgapped_rag.py` (not uvicorn)

## Still Having Issues?

Please share:
1. Output of `dir` (list files in project directory)
2. Output of `python --version`
3. Output of `pip list`
4. The EXACT command you're running
5. Any .env file contents
