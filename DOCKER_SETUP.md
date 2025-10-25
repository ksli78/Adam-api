# Docker Setup for Advanced RAG System

## Overview

The Advanced RAG system runs in **two separate Docker containers**:

1. **Ollama container** - Runs the LLM (llama3:8b)
2. **RAG API container** - Runs the document processing and query API

These containers communicate via Docker networking.

## Architecture

```
┌─────────────────────────────────────┐
│  Host Machine (Windows/Linux/Mac)   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  Docker Network             │   │
│  │                             │   │
│  │  ┌──────────────────────┐   │   │
│  │  │  Ollama Container    │   │   │
│  │  │  Port: 11434         │   │   │
│  │  │  Model: llama3:8b    │   │   │
│  │  └──────────┬───────────┘   │   │
│  │             │                │   │
│  │  ┌──────────▼───────────┐   │   │
│  │  │  RAG API Container   │   │   │
│  │  │  Port: 8000          │   │   │
│  │  │  + Docling           │   │   │
│  │  │  + poppler-utils     │   │   │
│  │  │  + tesseract-ocr     │   │   │
│  │  │  + ChromaDB          │   │   │
│  │  └──────────────────────┘   │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
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

### Option 1: Docker Compose (Recommended)

```bash
# 1. Build and start both containers
docker-compose -f docker-compose.advanced.yml up -d

# 2. Wait for Ollama to be ready (takes ~30 seconds)
docker logs ollama -f
# Wait until you see: "Ollama is running"

# 3. Pull the LLM model into Ollama container
docker exec ollama ollama pull llama3:8b
# This downloads ~4.7GB - takes 5-10 minutes depending on connection

# 4. Verify RAG API is ready
curl http://localhost:8000/health
# Should return: {"status": "healthy", "version": "2.0.0"}

# 5. Test with a document upload
curl -X POST http://localhost:8000/upload-document \
  -F "file=@test.pdf" \
  -F "source_url=https://test.com/test.pdf"
```

### Option 2: Separate Containers

If you want to build/run containers separately:

```bash
# 1. Start Ollama
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_models:/root/.ollama \
  ollama/ollama:latest

# 2. Pull LLM model
docker exec ollama ollama pull llama3:8b

# 3. Build RAG API
docker build -f Dockerfile.advanced -t rag-api-advanced .

# 4. Run RAG API (linked to Ollama)
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  -v rag_data:/data/airgapped_rag \
  -e OLLAMA_HOST=http://ollama:11434 \
  --link ollama \
  rag-api-advanced
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
   python -c "import nltk; nltk.download('punkt')"
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

**Problem**: RAG API can't connect to Ollama at `http://ollama:11434`

**Solutions**:
```bash
# 1. Check Ollama is running
docker ps | grep ollama

# 2. Check Ollama health
docker exec ollama ollama list

# 3. Check network connectivity
docker exec rag-api ping ollama

# 4. Check Ollama logs
docker logs ollama

# 5. Restart containers in order
docker-compose -f docker-compose.advanced.yml restart ollama
docker-compose -f docker-compose.advanced.yml restart rag-api
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
✅ **Ollama** → Separate container (docker-compose.advanced.yml)
✅ **RAG API** → Separate container with system dependencies
✅ **Windows development** → Install poppler/tesseract manually OR skip (reduced quality)

---

**Quick Commands Cheatsheet**:
```bash
# Start everything
docker-compose -f docker-compose.advanced.yml up -d

# Pull model
docker exec ollama ollama pull llama3:8b

# Check health
curl http://localhost:8000/health

# View logs
docker-compose -f docker-compose.advanced.yml logs -f

# Stop everything
docker-compose -f docker-compose.advanced.yml down

# Stop and remove volumes (fresh start)
docker-compose -f docker-compose.advanced.yml down -v
```
