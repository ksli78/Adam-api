# Docker Setup for Advanced RAG System

## Overview

The Advanced RAG API runs in a **single Docker container** that connects to your **existing Ollama** instance.

**Setup**:
1. **Ollama** - Already running (via your existing docker-compose.ollama.yml)
2. **RAG API container** - Connects to Ollama via `host.docker.internal`

This keeps the containers separate so you can:
- Debug the RAG API locally before containerizing
- Keep Ollama configuration independent
- Update/restart either service without affecting the other

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Host Machine (Windows/Linux/Mac)                   │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Existing Ollama Container                   │  │
│  │  (via docker-compose.ollama.yml)             │  │
│  │  Port: 11434                                 │  │
│  │  Model: llama3:8b                            │  │
│  └────────────────────┬─────────────────────────┘  │
│                       │ host.docker.internal        │
│                       │ :11434                      │
│  ┌────────────────────▼─────────────────────────┐  │
│  │  RAG API Container                           │  │
│  │  (docker-compose.advanced.yml)               │  │
│  │  Port: 8000                                  │  │
│  │  ├─ Docling (PDF extraction)                 │  │
│  │  ├─ poppler-utils (PDF tools)                │  │
│  │  ├─ tesseract-ocr (OCR)                      │  │
│  │  ├─ ChromaDB (vector storage)                │  │
│  │  ├─ DocumentCleaner                          │  │
│  │  ├─ SemanticChunker                          │  │
│  │  ├─ MetadataExtractor                        │  │
│  │  └─ ParentChildStore                         │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  OR Debug Locally (no container):                  │
│  ┌──────────────────────────────────────────────┐  │
│  │  python run_advanced.py                      │  │
│  │  ├─ Connects to Ollama at localhost:11434    │  │
│  │  └─ Stores data in /data/airgapped_rag       │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## System Dependencies Explanation

### What goes where?

| Dependency | Type | Location | Installation |
|------------|------|----------|--------------|
| **poppler-utils** | System | RAG API container | `apt-get` in Dockerfile |
| **tesseract-ocr** | System | RAG API container | `apt-get` in Dockerfile |
| **Python packages** | Python | RAG API container | `pip` via requirements.txt |
| **Ollama** | Service | Ollama container | Pre-built Docker image |
| **llama3:8b model** | LLM | Ollama container | `ollama pull` command |

### Why poppler-utils and tesseract-ocr?

These are **system-level dependencies** needed by **Docling** (the PDF extraction library):

- **poppler-utils** - PDF rendering and conversion tools
  - Converts PDF pages to images
  - Extracts text from PDFs
  - Required for: `pdftotext`, `pdftoppm`, `pdfinfo`

- **tesseract-ocr** - Optical Character Recognition (OCR)
  - Extracts text from images/scanned PDFs
  - Required when PDF text layer is poor quality
  - Improves extraction accuracy

**These CANNOT go in requirements.txt** because:
- They are compiled binaries (not Python packages)
- They must be installed at the OS level
- They must be in the system PATH

## Quick Start

### Prerequisites

**Ensure Ollama is already running**:
```bash
# Check if Ollama container is running
docker ps | grep ollama

# If not running, start it with your existing docker-compose
docker-compose -f docker-compose.ollama.yml up -d

# Verify Ollama is accessible
curl http://localhost:11434/api/tags

# Ensure llama3:8b model is pulled
docker exec ollama ollama list | grep llama3
# If not present: docker exec ollama ollama pull llama3:8b
```

### Option 1: Local Development (Debug First)

**Recommended workflow**: Test locally before containerizing

```bash
# 1. Install Python dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"

# 2. Install system dependencies (Windows)
choco install poppler tesseract
# OR skip for quick testing (reduced PDF quality)

# 3. Run locally
python run_advanced.py
# Connects to Ollama at localhost:11434

# 4. Test
curl http://localhost:8000/health

# 5. Upload a document
curl -X POST http://localhost:8000/upload-document \
  -F "file=@test.pdf" \
  -F "source_url=https://test.com/test.pdf"

# 6. Debug, iterate, test...
```

### Option 2: Docker Container (After Debugging)

**Once you're happy with local testing**, containerize it:

```bash
# 1. Ensure Ollama is running (from prerequisites)
docker ps | grep ollama

# 2. Build and start RAG API container
docker-compose -f docker-compose.advanced.yml up -d

# 3. Check logs
docker logs -f rag-api-advanced

# 4. Verify health
curl http://localhost:8000/health

# 5. Test with document upload
curl -X POST http://localhost:8000/upload-document \
  -F "file=@test.pdf" \
  -F "source_url=https://test.com/test.pdf"
```

## Local Development (Windows)

For **local Windows development** without Docker:

### Prerequisites

1. **Install Ollama for Windows**
   ```powershell
   # Download from: https://ollama.com/download/windows
   # Or via winget:
   winget install Ollama.Ollama

   # Start Ollama
   ollama serve

   # Pull model
   ollama pull llama3:8b
   ```

2. **Install Python 3.10+**
   ```powershell
   # Download from: https://www.python.org/downloads/
   # Or via Microsoft Store
   ```

3. **Install Poppler (for PDF processing)**
   ```powershell
   # Option A: Chocolatey
   choco install poppler

   # Option B: Manual download
   # Download from: https://github.com/oschwartz10612/poppler-windows/releases
   # Extract and add bin/ folder to PATH
   ```

4. **Install Tesseract (for OCR)**
   ```powershell
   # Option A: Chocolatey
   choco install tesseract

   # Option B: Manual download
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   # Install and add to PATH
   ```

5. **Install Python dependencies**
   ```powershell
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt_tab')"
   ```

6. **Run the API**
   ```powershell
   python run_advanced.py
   ```

### Skip Poppler/Tesseract (Quick Start)

If you just want to test quickly without installing system dependencies:

```powershell
# Docling will fall back to basic extraction (reduced quality)
python run_advanced.py
```

**Note**: You'll get warnings but the system will still work with reduced PDF extraction quality.

## Production Deployment

### Build Multi-Platform Images

```bash
# Build for both amd64 and arm64
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.advanced \
  -t myregistry/rag-api-advanced:latest \
  --push .
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/data/airgapped_rag` | Storage location for docs/DB |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama API endpoint |
| `LLM_MODEL` | `llama3:8b` | Ollama model name |

### Persistent Storage

Both containers use Docker volumes for persistence:

```bash
# View volumes
docker volume ls

# Inspect RAG data volume
docker volume inspect adam-api_rag_data

# Backup RAG data
docker run --rm -v adam-api_rag_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/rag-backup.tar.gz /data

# Restore RAG data
docker run --rm -v adam-api_rag_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/rag-backup.tar.gz -C /
```

## Troubleshooting

### "Ollama not reachable" from RAG API

**Problem**: RAG API can't connect to Ollama at `http://host.docker.internal:11434`

**Solutions**:

```bash
# 1. Check Ollama is running
docker ps | grep ollama

# 2. Check Ollama is accessible from host
curl http://localhost:11434/api/tags

# 3. Check Ollama is accessible from RAG container
docker exec rag-api-advanced curl http://host.docker.internal:11434/api/tags

# 4. If on Linux, add extra_hosts to docker-compose.advanced.yml:
#    Uncomment this section:
#    extra_hosts:
#      - "host.docker.internal:host-gateway"

# 5. Check RAG API logs for connection errors
docker logs rag-api-advanced

# 6. Restart RAG API container
docker-compose -f docker-compose.advanced.yml restart
```

**Alternative**: If `host.docker.internal` doesn't work, find your host IP:
```bash
# On Linux, get Docker bridge IP
ip addr show docker0

# Then update docker-compose.advanced.yml:
# - OLLAMA_HOST=http://172.17.0.1:11434  # Use actual bridge IP
```

### "Model not found: llama3:8b"

**Problem**: Model wasn't pulled into Ollama container

**Solution**:
```bash
# Pull model
docker exec ollama ollama pull llama3:8b

# Verify it's available
docker exec ollama ollama list
```

### "Docling extraction failed"

**Problem**: Poppler or Tesseract not installed in container

**Solution**:
```bash
# Verify system dependencies in container
docker exec rag-api which pdftotext
docker exec rag-api which tesseract

# If missing, rebuild with Dockerfile.advanced
docker-compose -f docker-compose.advanced.yml build --no-cache
```

### "Permission denied" writing to /data

**Problem**: Docker volume permissions

**Solution**:
```bash
# Fix permissions
docker run --rm -v adam-api_rag_data:/data alpine chown -R 1000:1000 /data

# Or run container as root
docker-compose -f docker-compose.advanced.yml up --build
```

## Resource Requirements

### Minimum (for testing)
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 20GB
  - Ollama model: ~5GB
  - Docker images: ~3GB
  - Document storage: ~10GB

### Recommended (for production)
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Disk**: 100GB+
- **GPU**: Optional (speeds up LLM inference 10-20x)

### With GPU Support

```yaml
# In docker-compose.advanced.yml, uncomment GPU section for Ollama:
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**Prerequisites**:
- NVIDIA GPU with CUDA support
- nvidia-docker2 installed
- NVIDIA Container Toolkit installed

## Monitoring

### Check Container Health

```bash
# Container status
docker-compose -f docker-compose.advanced.yml ps

# Ollama health
curl http://localhost:11434/api/tags

# RAG API health
curl http://localhost:8000/health

# System statistics
curl http://localhost:8000/statistics
```

### View Logs

```bash
# All containers
docker-compose -f docker-compose.advanced.yml logs -f

# Just RAG API
docker-compose -f docker-compose.advanced.yml logs -f rag-api

# Just Ollama
docker-compose -f docker-compose.advanced.yml logs -f ollama
```

### Resource Usage

```bash
# Container stats
docker stats ollama rag-api-advanced

# Disk usage
docker system df
```

## Updating

### Update RAG API Code

```bash
# 1. Pull latest code
git pull

# 2. Rebuild RAG API container
docker-compose -f docker-compose.advanced.yml build rag-api

# 3. Restart with new image
docker-compose -f docker-compose.advanced.yml up -d rag-api
```

### Update Ollama Model

```bash
# Pull newer version
docker exec ollama ollama pull llama3:8b

# Or switch models
docker exec ollama ollama pull llama3:70b
```

## Summary

✅ **System dependencies** (poppler, tesseract) → Dockerfile.advanced
✅ **Python dependencies** → requirements.txt
✅ **Ollama** → Existing container (docker-compose.ollama.yml) - **stays separate**
✅ **RAG API** → New container (docker-compose.advanced.yml) - connects to existing Ollama
✅ **Windows development** → Install poppler/tesseract manually OR skip (reduced quality)

### Workflow

```
1. Debug Locally:
   └─> python run_advanced.py (connects to Ollama at localhost:11434)

2. When Ready:
   └─> docker-compose -f docker-compose.advanced.yml up -d
       └─> RAG API container connects to Ollama via host.docker.internal:11434
```

---

**Quick Commands Cheatsheet**:

```bash
# === Local Development ===
# Start Ollama (if not running)
docker ps | grep ollama  # Check first
docker-compose -f docker-compose.ollama.yml up -d

# Run RAG API locally
python run_advanced.py

# === Docker Deployment ===
# Build and start RAG API container
docker-compose -f docker-compose.advanced.yml up -d

# Check RAG API health
curl http://localhost:8000/health

# View RAG API logs
docker logs -f rag-api-advanced

# Restart RAG API (keep Ollama running)
docker-compose -f docker-compose.advanced.yml restart

# Stop RAG API (keep Ollama running)
docker-compose -f docker-compose.advanced.yml down

# Stop RAG API and delete data (fresh start)
docker-compose -f docker-compose.advanced.yml down -v

# === Both Systems ===
# Check Ollama model
docker exec ollama ollama list

# Pull/update Ollama model
docker exec ollama ollama pull llama3:8b
```
