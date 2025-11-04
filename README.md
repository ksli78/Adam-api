# Adam - Amentum Document Assistant and Manager

Production-ready AI-powered RAG (Retrieval-Augmented Generation) system for intelligent document search and question answering.

## 🎯 Overview

**Adam** is a sophisticated, fully air-gapped RAG system that helps users find information in company documents using advanced AI techniques. Built with production-grade components and designed for enterprise use.

### Key Features

✅ **Intelligent Hybrid Search** - Combines BM25 keyword matching with semantic AI understanding
✅ **Parent-Child Chunking** - Retrieves precise chunks, provides rich context
✅ **Feedback Learning** - Continuously improves from user ratings
✅ **System Query Handling** - Automatically detects and answers meta-queries
✅ **Semantic Relevance Filtering** - Prevents irrelevant keyword-only matches
✅ **100% Air-Gapped** - No external API calls, fully local processing
✅ **Document Citations** - Every answer includes source references

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Query                           │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  Classifier   │ → System Query → Direct Response
         │ (LLM-based)   │
         └───────┬───────┘
                 │
                 │ Document Query
                 ▼
         ┌──────────────┐
         │ Hybrid Search│
         │ BM25 + AI    │
         └──────┬───────┘
                │
                ▼
    ┌────────────────────────┐
    │ Child Chunks (precise) │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ Parent Chunks (context)│
    └────────┬───────────────┘
             │
             ▼
         ┌────────┐
         │  LLM   │ → Answer + Citations
         │(Ollama)│
         └────────┘
```

## 📦 Components

### Core Modules

- **`airgapped_rag_advanced.py`** - Main FastAPI application
- **`parent_child_store.py`** - Hybrid search with feedback weighting
- **`query_classifier.py`** - System vs document query detection
- **`feedback_store.py`** - User feedback storage and analytics
- **`semantic_chunker.py`** - Intelligent document chunking
- **`document_cleaner.py`** - Document cleaning and preprocessing
- **`metadata_extractor.py`** - LLM-based metadata extraction

### Key Technologies

- **FastAPI** - Modern async REST API framework
- **ChromaDB** - Vector database for embeddings
- **Ollama** - Local LLM inference (Llama 3)
- **Sentence Transformers** - Local embedding models
- **Docling** - Structure-aware PDF extraction
- **BM25** - Keyword-based retrieval
- **SQLite** - Feedback and analytics storage

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (for Ollama)
- 8GB+ RAM recommended

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run downloads models (~500MB):
- `all-MiniLM-L6-v2` for embeddings (local AI)
- Ollama models pulled separately

### 2. Start Ollama (if not running)

```bash
# Already running on your system
# Ensure llama3:8b model is available:
docker exec <ollama-container> ollama pull llama3:8b
```

### 3. Run the API

```bash
python run_advanced.py
```

API available at: **http://localhost:8000**
Interactive docs: **http://localhost:8000/docs**

## 🐳 Docker Deployment

See **[DOCKER_BUILD.md](DOCKER_BUILD.md)** for complete Docker build and deployment instructions.

Quick start:
```bash
docker-compose up -d
```

## 📖 Documentation

- **[README_ADVANCED_RAG.md](README_ADVANCED_RAG.md)** - Complete system documentation
- **[SYSTEM_QUERIES.md](SYSTEM_QUERIES.md)** - System query handling guide
- **[DOCKER_BUILD.md](DOCKER_BUILD.md)** - Docker deployment guide
- **[SHAREPOINT_INTEGRATION_SUMMARY.md](SHAREPOINT_INTEGRATION_SUMMARY.md)** - SharePoint integration

## 🔧 Configuration

Environment variables:

```bash
DATA_DIR=/data/airgapped_rag         # Data storage directory
OLLAMA_HOST=http://localhost:11434   # Ollama server URL
LLM_MODEL=llama3:8b                  # LLM model to use
```

## 📊 API Endpoints

### Document Management

- `POST /upload-document` - Upload and process PDF documents
- `GET /documents` - List all indexed documents
- `DELETE /documents/{id}` - Delete a specific document
- `DELETE /documents` - Clear all documents

### Query & Search

- `POST /query` - Ask questions about documents
  - Automatic system vs document query classification
  - Hybrid search with relevance filtering
  - Returns answer with citations

### Feedback System

- `POST /feedback` - Submit user feedback (good/bad)
- `GET /feedback/analytics` - View satisfaction metrics
- `GET /feedback/recent` - View recent feedback

### Utilities

- `GET /statistics` - System statistics
- `GET /health` - Health check
- `GET /debug/document/{id}` - Inspect document chunks

## 💡 Usage Examples

### Upload a Document

```bash
curl -X POST "http://localhost:8000/upload-document" \
  -F "file=@policy.pdf" \
  -F "source_url=https://portal.example.com/policy.pdf"
```

### Ask a Question

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the PTO policy?",
    "use_hybrid": true,
    "bm25_weight": 0.5
  }'
```

### Submit Feedback

```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the PTO policy?",
    "answer": "PTO varies by years of service...",
    "feedback_type": "good",
    "retrieval_stats": {...}
  }'
```

## 🎓 How It Works

### 1. Document Ingestion

```
PDF → Docling Extraction → Cleaning → Semantic Chunking → Metadata Extraction → ChromaDB Storage
```

- **Docling** preserves document structure
- **Cleaning** removes headers, footers, noise
- **Chunking** creates parent (context) and child (precise) chunks
- **Metadata** extracted with LLM for better retrieval
- **Storage** in dual ChromaDB collections

### 2. Query Processing

```
Query → Classification → Routing → Retrieval → Answer Generation
```

**System Queries** (e.g., "What can you do?"):
- Detected by LLM classifier
- Responded directly with system info
- No document retrieval

**Document Queries** (e.g., "What is the PTO policy?"):
- Hybrid search (BM25 + semantic)
- Relevance filtering (BM25 >= 0.95, semantic >= 0.2)
- Feedback weighting (15% boost/demote based on history)
- Parent chunk expansion for context
- LLM generates answer with citations

### 3. Feedback Learning

```
User Rates → Store Feedback → Update Chunk Scores → Influence Future Retrievals
```

- Tracks good/bad feedback per chunk
- Calculates quality scores (-1.0 to +1.0)
- Applies 15% adjustment in future searches
- Provides analytics on best/worst content

## 🔒 Security & Privacy

- ✅ **100% Air-Gapped** - No external API calls
- ✅ **Local Processing** - All AI runs locally
- ✅ **Data Privacy** - Documents never leave your infrastructure
- ✅ **Persistent Storage** - Data survives restarts

## 📈 Performance

- **Ingestion**: ~10-30 seconds per PDF (depending on size)
- **Query**: ~2-5 seconds (including LLM generation)
- **Concurrent Users**: Supports multiple simultaneous queries
- **Scalability**: Single worker recommended for air-gapped use

## 🤝 Contributing

This system is built for enterprise use. For issues or enhancements:

1. Check existing documentation
2. Review logs for error details
3. Submit feedback through the API

## 📄 License

Proprietary - Amentum Corporation

## 🆘 Support

- Documentation: See files in this repository
- API Docs: http://localhost:8000/docs (when running)
- Logs: Check console output and application logs

---

**Adam v2.0** - Built with ❤️ for intelligent document search
