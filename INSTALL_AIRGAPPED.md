# Quick Installation Guide - Air-Gapped RAG API

This is a streamlined installation guide. For complete documentation, see [README_AIRGAPPED_RAG.md](README_AIRGAPPED_RAG.md).

## 5-Minute Quick Start

### 1. Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:** Download from https://ollama.com/download

### 2. Install Required Models

```bash
# Start Ollama (if not auto-started)
ollama serve &

# Pull models (this will download ~5GB)
ollama pull nomic-embed-text
ollama pull llama3:8b
```

### 3. Install Python Dependencies

```bash
# Navigate to project directory
cd Adam-api

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Start the API

**Option A: Using the startup script (recommended)**
```bash
./start_airgapped_rag.sh
```

**Option B: Direct Python**
```bash
python airgapped_rag.py
```

### 5. Test It!

Open another terminal:

```bash
# Check health
curl http://localhost:8000/health

# Or use the example script
python example_usage.py --health
```

## First Document Upload

```bash
# Upload a PDF
python example_usage.py --upload /path/to/document.pdf --url http://source.url

# Query it
python example_usage.py --query "What is this document about?"
```

## API Documentation

Once running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## Common Issues

### "Ollama connection failed"
```bash
# Start Ollama
ollama serve
```

### "Model not found"
```bash
# Pull the models
ollama pull nomic-embed-text
ollama pull llama3:8b
```

### "Port already in use"
```bash
# Use a different port
PORT=8001 python airgapped_rag.py
```

## System Requirements

- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 10GB for models
- **Python**: 3.10 or higher
- **OS**: Linux, macOS, or Windows

## What's Different from the Original System?

This air-gapped RAG system:
- ✅ Uses **Ollama** instead of HuggingFace models (fully local)
- ✅ Uses **ChromaDB** instead of FAISS (simpler setup)
- ✅ Retrieves **full documents** instead of chunks (better context)
- ✅ Uses **topic-based indexing** (one embedding per document)
- ✅ Provides **inline citations** with source URLs and excerpts

The original system (`api/main.py`) remains functional and uses:
- HuggingFace Granite models
- FAISS + TF-IDF hybrid search
- Document chunking strategy

## Next Steps

1. Read the full [README_AIRGAPPED_RAG.md](README_AIRGAPPED_RAG.md)
2. Upload your documents
3. Start querying!

## Getting Help

- Check the [Troubleshooting section](README_AIRGAPPED_RAG.md#-troubleshooting) in the full README
- Review Ollama docs: https://ollama.com/docs
- Check API docs: http://localhost:8000/docs

---

**Ready to use!** The system is fully air-gapped and requires no external API calls.
