# Air-Gapped RAG API with Ollama and ChromaDB

A robust, production-ready Retrieval-Augmented Generation (RAG) system designed for air-gapped environments. This system uses **Ollama** for local LLM inference, **ChromaDB** for vector storage, and retrieves **full documents** based on topic similarity to provide accurate answers with proper citations.

## 🎯 Key Features

- **100% Air-Gapped Compatible**: All processing happens locally with no external API calls
- **Ollama Integration**: Uses local Ollama for embeddings and text generation
- **Full Document Retrieval**: Stores and retrieves complete documents (no chunking)
- **Topic-Based Indexing**: One embedding per document based on generated topic/summary
- **Accurate Citations**: Provides source URLs and relevant excerpts with every answer
- **FastAPI Backend**: Modern, async Python API with automatic documentation
- **ChromaDB Vector Store**: Efficient, embedded vector database for similarity search

## 🏗️ Architecture

```
┌─────────────┐
│   Upload    │
│     PDF     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│  PDF → Markdown Conversion  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Generate Topic Summary     │
│  (Ollama Llama 3)           │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Generate Topic Embedding   │
│  (Ollama nomic-embed-text)  │
└──────────┬──────────────────┘
           │
           ├─────► Store Full Document (JSON + .md files)
           │
           └─────► Store Topic Embedding (ChromaDB)


┌─────────────┐
│    Query    │
│  Question   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│  Generate Query Embedding   │
│  (Ollama nomic-embed-text)  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Search Similar Topics      │
│  (ChromaDB Similarity)      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Retrieve Full Documents    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Generate Answer            │
│  (Ollama Llama 3)           │
│  with Citations             │
└──────────┬──────────────────┘
           │
           ▼
      ┌────────┐
      │ Answer │
      │   +    │
      │Citations│
      └────────┘
```

## 📋 Prerequisites

### Required Software

1. **Python 3.10+**
2. **Ollama** (for local LLM inference)

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 10GB for models + storage for documents
- **CPU**: Modern multi-core processor (GPU optional but recommended)

## 🚀 Installation

### Step 1: Install Ollama

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### macOS
```bash
brew install ollama
```

#### Windows
Download from: https://ollama.com/download

### Step 2: Pull Required Ollama Models

```bash
# Embedding model (required)
ollama pull nomic-embed-text

# LLM for generation (required)
ollama pull llama3:8b

# Optional: Use a smaller/larger model
# ollama pull llama3:13b
# ollama pull llama2:7b
```

### Step 3: Install Python Dependencies

```bash
# Clone the repository (if not already done)
cd Adam-api

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Verify Ollama Installation

```bash
# Check Ollama is running
ollama list

# Should show:
# NAME                    ID              SIZE    MODIFIED
# nomic-embed-text:latest xxxxx          274MB   X days ago
# llama3:8b              xxxxx          4.7GB   X days ago
```

## ⚙️ Configuration

The system can be configured using environment variables:

```bash
# Optional: Create a .env file
cat > .env.airgapped << EOF
# Data storage directory
DATA_DIR=/data/airgapped_rag

# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3:8b

# Server configuration
HOST=0.0.0.0
PORT=8000
EOF
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/data/airgapped_rag` | Directory for storing documents and ChromaDB |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Model for generating embeddings |
| `OLLAMA_LLM_MODEL` | `llama3:8b` | Model for text generation |
| `HOST` | `0.0.0.0` | API server host |
| `PORT` | `8000` | API server port |

## 🏃 Running the Application

### Method 1: Direct Python

```bash
# Start Ollama (if not already running)
ollama serve &

# Run the API
python airgapped_rag.py
```

### Method 2: Using Uvicorn

```bash
# Start Ollama
ollama serve &

# Run with Uvicorn
uvicorn airgapped_rag:app --host 0.0.0.0 --port 8000 --reload
```

### Method 3: Production Deployment

```bash
# Start Ollama as a service
sudo systemctl start ollama  # Linux with systemd

# Run API with Gunicorn
pip install gunicorn
gunicorn airgapped_rag:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 API Documentation

Once running, access the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "ollama_base_url": "http://localhost:11434",
  "embed_model": "nomic-embed-text",
  "llm_model": "llama3:8b",
  "documents_count": 5
}
```

#### 2. Upload Document
```bash
POST /upload-document
Content-Type: multipart/form-data

file: [PDF file]
source_url: "http://company.com/hr/pto-policy"
```

**Response:**
```json
{
  "document_id": "a3f2c8d9e1b4a5c6d7e8f9a0b1c2d3e4",
  "message": "Document indexed successfully. Topic: PTO Policy 2024",
  "topic": "Annual Paid Time Off Policy for Full-Time Employees 2024",
  "source_url": "http://company.com/hr/pto-policy"
}
```

#### 3. Query Documents
```bash
POST /query
Content-Type: application/json

{
  "prompt": "What is the current PTO policy and how do I apply?",
  "top_k": 2
}
```

**Response:**
```json
{
  "answer": "According to the PTO policy, full-time employees receive 20 days of paid time off annually [Document 1: \"Employees are granted twenty (20) days of PTO on an annual basis\"]. To apply for PTO, employees must submit Form HR-501 at least 2 weeks in advance through the employee portal [Document 1: \"PTO requests require Form HR-501 submission minimum 14 days prior\"].",
  "citations": [
    {
      "source_url": "http://company.com/hr/pto-policy",
      "excerpt": "Employees are granted twenty (20) days of PTO on an annual basis, accrued monthly at 1.67 days per month..."
    },
    {
      "source_url": "http://company.com/hr/pto-policy",
      "excerpt": "PTO requests require Form HR-501 submission minimum 14 days prior to the requested start date via the employee self-service portal..."
    }
  ]
}
```

#### 4. List Documents
```bash
GET /documents
```

**Response:**
```json
[
  {
    "document_id": "a3f2c8d9e1b4a5c6d7e8f9a0b1c2d3e4",
    "topic": "Annual Paid Time Off Policy for Full-Time Employees 2024",
    "source_url": "http://company.com/hr/pto-policy",
    "filename": "pto_policy_2024.pdf",
    "created_at": "2024-10-24T10:30:00",
    "char_count": 15432
  }
]
```

#### 5. Delete Document
```bash
DELETE /documents/{document_id}
```

## 🧪 Testing

### Using cURL

```bash
# 1. Upload a test document
curl -X POST "http://localhost:8000/upload-document" \
  -F "file=@test_document.pdf" \
  -F "source_url=http://example.com/test"

# 2. Query the system
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is this document about?",
    "top_k": 1
  }'

# 3. List all documents
curl "http://localhost:8000/documents"
```

### Using Python

```python
import requests

# Upload document
with open("test.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload-document",
        files={"file": f},
        data={"source_url": "http://example.com/test"}
    )
    print(response.json())

# Query
response = requests.post(
    "http://localhost:8000/query",
    json={
        "prompt": "What are the key points?",
        "top_k": 1
    }
)
print(response.json())
```

### Using HTTPie

```bash
# Upload
http -f POST localhost:8000/upload-document \
  file@test.pdf \
  source_url="http://example.com/test"

# Query
http POST localhost:8000/query \
  prompt="What is this about?" \
  top_k:=1
```

## 🔍 How It Works

### Document Ingestion Flow

1. **PDF Upload**: User uploads a PDF with a `source_url`
2. **Conversion**: PDF is converted to Markdown format (preserving structure)
3. **Topic Generation**: Ollama Llama 3 generates a concise topic/summary from the document's first portion
4. **Embedding**: Ollama generates an embedding vector for the topic using `nomic-embed-text`
5. **Storage**:
   - Full Markdown content stored in local file system
   - Metadata stored in JSON index
   - Topic embedding stored in ChromaDB

### Query Flow

1. **Question Embedding**: User's question is embedded using `nomic-embed-text`
2. **Similarity Search**: ChromaDB finds top-K documents with most similar topic embeddings
3. **Document Retrieval**: Full document contents are retrieved from storage
4. **Answer Generation**:
   - Ollama Llama 3 receives the full documents as context
   - Generates an answer using ONLY the provided documents
   - Includes inline citations in format `[Document N: "excerpt"]`
5. **Citation Extraction**: Citations are parsed and formatted with source URLs and excerpts

## 🎛️ Advanced Usage

### Custom Models

You can use different Ollama models:

```bash
# Use a larger/smaller LLM
export OLLAMA_LLM_MODEL=llama3:13b
# or
export OLLAMA_LLM_MODEL=llama2:7b

# Use different embedding model
export OLLAMA_EMBED_MODEL=all-minilm
```

### Adjusting Retrieval

Modify the `top_k` parameter in queries to retrieve more documents:

```json
{
  "prompt": "Your question",
  "top_k": 3  // Retrieve top 3 most relevant documents
}
```

### Batch Upload

Create a script to upload multiple documents:

```python
import os
import requests

docs_dir = "path/to/pdfs"
base_url = "http://localhost:8000"

for filename in os.listdir(docs_dir):
    if filename.endswith('.pdf'):
        with open(os.path.join(docs_dir, filename), 'rb') as f:
            response = requests.post(
                f"{base_url}/upload-document",
                files={"file": f},
                data={"source_url": f"http://internal/{filename}"}
            )
            print(f"Uploaded {filename}: {response.json()['topic']}")
```

## 🐛 Troubleshooting

### Ollama Connection Failed

**Symptom**: `Failed to generate embedding: Connection refused`

**Solution**:
```bash
# Make sure Ollama is running
ollama serve

# Check Ollama is accessible
curl http://localhost:11434/api/tags
```

### Models Not Found

**Symptom**: `model 'llama3:8b' not found`

**Solution**:
```bash
# Pull the required models
ollama pull nomic-embed-text
ollama pull llama3:8b
```

### Out of Memory

**Symptom**: System crashes or becomes unresponsive

**Solution**:
1. Use a smaller model: `ollama pull llama2:7b`
2. Reduce concurrent requests
3. Increase system RAM or use swap

### Slow Performance

**Solution**:
1. Use GPU acceleration (Ollama auto-detects CUDA/Metal)
2. Reduce document size/count
3. Lower `top_k` in queries
4. Use smaller models for faster inference

### ChromaDB Errors

**Symptom**: `Collection not found` or database errors

**Solution**:
```bash
# Delete and reinitialize ChromaDB
rm -rf /data/airgapped_rag/chromadb
# Restart the API - it will recreate the collection
```

## 📊 Performance Considerations

### Model Selection

| Model | Size | Speed | Quality | RAM Required |
|-------|------|-------|---------|--------------|
| llama3:8b | 4.7GB | Medium | High | 8GB |
| llama3:13b | 7.4GB | Slow | Very High | 16GB |
| llama2:7b | 3.8GB | Fast | Good | 6GB |

### Embedding Model

- `nomic-embed-text` (274MB): Fast, high-quality, recommended for most use cases
- `all-minilm` (120MB): Faster but lower quality

### Optimization Tips

1. **Use GPU**: Ollama automatically uses GPU if available (NVIDIA CUDA, Apple Metal)
2. **Batch uploads**: Upload documents during off-peak hours
3. **Cache warmup**: First query after startup is slower (model loading)
4. **Document size**: Smaller documents = faster processing
5. **Concurrent requests**: Limit based on available RAM

## 🔒 Security Considerations

### Air-Gapped Deployment

This system is designed for air-gapped environments:

- ✅ No external API calls
- ✅ All models run locally
- ✅ Data never leaves your infrastructure
- ✅ No telemetry or tracking

### Recommended Security Practices

1. **Network isolation**: Deploy in isolated network segment
2. **Access control**: Use reverse proxy with authentication
3. **Input validation**: System validates file types and inputs
4. **Resource limits**: Set ulimits and container resource constraints
5. **Audit logging**: Monitor API access and document uploads

## 📁 Data Storage

### Directory Structure

```
/data/airgapped_rag/
├── documents/
│   ├── index.json                    # Document metadata index
│   ├── {doc_id}.md                   # Full document content
│   ├── a3f2c8d9e1b4a5c6.md
│   └── b5d7e9f1a3c5b7d9.md
└── chromadb/
    └── {chroma_internal_files}       # ChromaDB vector store
```

### Backup

To backup your data:

```bash
# Backup documents and embeddings
tar -czf airgapped_rag_backup_$(date +%Y%m%d).tar.gz /data/airgapped_rag/

# Restore
tar -xzf airgapped_rag_backup_20241024.tar.gz -C /
```

## 🔄 Migration from Existing System

If you're migrating from the existing chunked-based system:

```python
# Export existing documents
# Then re-upload through the new /upload-document endpoint
# The new system will re-index them with topic-based approach
```

## 📈 Monitoring

### Logging

The system logs to stdout. Capture logs:

```bash
# Run with log file
python airgapped_rag.py 2>&1 | tee -a airgapped_rag.log

# Monitor logs
tail -f airgapped_rag.log
```

### Metrics

Monitor these key metrics:
- Document count: `GET /health`
- Query latency: Time API responses
- Memory usage: Monitor process RSS
- Disk usage: Monitor `/data/airgapped_rag` size

## 🤝 Contributing

This is a production system. If you encounter issues:

1. Check logs for error messages
2. Verify Ollama models are installed
3. Check system resources (RAM, disk)
4. Review configuration settings

## 📜 License

[Your License Here]

## 🙏 Acknowledgments

- **Ollama**: Local LLM inference
- **ChromaDB**: Vector database
- **FastAPI**: Modern Python web framework
- **PyMuPDF**: PDF processing

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review Ollama documentation: https://ollama.com/docs
- Check ChromaDB docs: https://docs.trychroma.com

---

**Note**: This system is designed for offline/air-gapped environments. All processing happens locally. No data is sent to external services.
