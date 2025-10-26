# Getting Started - Advanced RAG System

Quick start guide for your specific setup (Ollama already running separately).

## Your Current Setup

```
✅ Ollama Container - Already running (docker-compose.ollama.yml)
🔧 RAG API - Debug locally first, then containerize when ready
```

## Step 1: Local Development (Debug First)

### Install Dependencies

```bash
# Python packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab')"

# System dependencies (Windows - optional for now)
# choco install poppler tesseract
# OR skip and test with reduced PDF quality
```

### Run Locally

```bash
# Ensure Ollama is running
docker ps | grep ollama

# Start the RAG API
python run_advanced.py
```

You should see:
```
Advanced Air-Gapped RAG System v2.0
Data directory: /data/airgapped_rag
Ollama host: http://localhost:11434
LLM model: llama3:8b
Starting server on http://0.0.0.0:8000
```

### Test It

```bash
# Health check
curl http://localhost:8000/health

# Upload a test document
curl -X POST http://localhost:8000/upload-document \
  -F "file=@test.pdf" \
  -F "source_url=https://test.com/test.pdf"

# Query it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the main topic of the document?",
    "top_k": 10,
    "parent_limit": 3
  }'
```

### What to Expect

**First Document Upload** (slow - this is normal!):
- Docling extraction: 5-30 seconds
- Section cleaning: 1-3 seconds
- **Metadata extraction (LLM): 10-30 seconds** ⏰
- Semantic chunking: 1-5 seconds
- Embedding generation: 5-30 seconds
- **Total: 30-90 seconds per document**

**Subsequent Queries** (fast):
- Retrieval + LLM answer: 2-5 seconds ⚡

### Debug & Iterate

- Check logs in console
- Modify code (edit Python files)
- Restart: Ctrl+C, then `python run_advanced.py`
- Repeat until happy with results

### Customize Cleaning

Edit `cleaning_config.yaml` to add your company's specific noise patterns:

```yaml
patterns_to_remove:
  - pattern: "YOUR_COMPANY_BANNER"
    description: "Company specific header"
```

## Step 2: Containerize (When Ready)

Once you're satisfied with local testing:

```bash
# Build and start RAG API container
docker-compose -f docker-compose.advanced.yml up -d

# Check logs
docker logs -f rag-api-advanced

# Test health
curl http://localhost:8000/health

# Upload a document to the container
curl -X POST http://localhost:8000/upload-document \
  -F "file=@test.pdf" \
  -F "source_url=https://test.com/test.pdf"
```

## Architecture

```
┌─────────────────────────────────────────┐
│ Your Setup                              │
├─────────────────────────────────────────┤
│                                         │
│ Ollama Container                        │
│ └─> docker-compose.ollama.yml           │
│     Port: 11434                         │
│     ↓                                   │
│     │ host.docker.internal              │
│     ↓                                   │
│ RAG API (choose one):                   │
│ ├─> Local: python run_advanced.py      │
│ │   Connects to localhost:11434         │
│ │                                       │
│ └─> Container: docker-compose up        │
│     Connects to host.docker.internal    │
│                                         │
└─────────────────────────────────────────┘
```

## Tips

### Windows + WSL2 + Docker Desktop

If using WSL2, `host.docker.internal` should work automatically.

### Windows + Docker Desktop (no WSL2)

Should also work with `host.docker.internal`.

### Linux

You may need to uncomment this in `docker-compose.advanced.yml`:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

## Common Issues

### "Cannot connect to Ollama"

```bash
# Check Ollama is running
docker ps | grep ollama

# Check Ollama is accessible
curl http://localhost:11434/api/tags

# If running locally, check:
python -c "import ollama; print(ollama.Client(host='http://localhost:11434').list())"
```

### "Docling extraction failed"

You're probably missing poppler/tesseract. Either:
1. Install them: `choco install poppler tesseract`
2. Or accept reduced quality (basic extraction still works)

### "Slow first upload"

This is **expected and normal**! The LLM metadata extraction takes 10-30 seconds. You said you don't mind the latency during upload. Queries will be fast (2-5s).

## Next Steps

1. ✅ Test locally with a few documents
2. ✅ Verify cleaning is removing your company's noise patterns
3. ✅ Test queries and check answer quality
4. ✅ Adjust `cleaning_config.yaml` if needed
5. ✅ Adjust chunking parameters if needed (in `airgapped_rag_advanced.py`)
6. ✅ When happy, containerize with docker-compose

## Files to Know

| File | Purpose | When to Edit |
|------|---------|--------------|
| `run_advanced.py` | Runner script | Never (just run it) |
| `airgapped_rag_advanced.py` | Main pipeline | To adjust chunking params |
| `cleaning_config.yaml` | Noise patterns | To add company-specific patterns |
| `requirements.txt` | Python deps | When adding new libraries |
| `Dockerfile.advanced` | Container image | When adding system deps |
| `docker-compose.advanced.yml` | Container config | To change env vars |

## Quick Reference

```bash
# Local development
python run_advanced.py

# Docker deployment
docker-compose -f docker-compose.advanced.yml up -d

# View logs
docker logs -f rag-api-advanced

# Stop container (keep data)
docker-compose -f docker-compose.advanced.yml down

# Stop container (delete data - fresh start)
docker-compose -f docker-compose.advanced.yml down -v

# Rebuild after code changes
docker-compose -f docker-compose.advanced.yml up -d --build
```

Happy RAG building! 🚀
