# Air-Gapped RAG API

A production-ready, air-gapped-compatible Retrieval-Augmented Generation (RAG) system using **Ollama** (Llama 3), **ChromaDB**, and **FastAPI**.

## 🚀 Quick Start

### Windows Development

```powershell
# 1. Start Ollama in Docker
docker-compose -f docker-compose.ollama.yml up -d

# 2. Pull models
bash setup-ollama-models.sh

# 3. Install Python dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 4. Run API
python airgapped_rag.py
```

**API runs at:** http://localhost:8000/docs

### RHEL9 Production

```bash
# 1. Export from Windows
.\export-for-rhel9.ps1

# 2. Transfer rhel9-deployment/ to RHEL9

# 3. Deploy on RHEL9
cd rhel9-deployment
sudo bash deploy-rhel9.sh
```

## 📁 Project Structure

```
Adam-api/
├── 📄 airgapped_rag.py              # Main API application
├── 📄 example_usage.py              # CLI testing tool
├── 📄 requirements.txt              # Python dependencies
│
├── 🐳 docker-compose.ollama.yml     # Dev: Ollama only
├── 🐳 docker-compose.airgapped.yml  # Prod: Full stack
├── 🐳 Dockerfile.airgapped          # API container (RHEL UBI9)
│
├── 🔧 setup-ollama-models.sh        # Pull Ollama models
├── 🔧 start_airgapped_rag.sh        # Start API locally
├── 🔧 export-for-rhel9.ps1          # Export for RHEL9 (PowerShell)
├── 🔧 deploy-rhel9.sh               # Deploy on RHEL9 (Bash)
│
├── 📖 README.md                     # This file
├── 📖 DOCKER_QUICK_START.md         # Quick reference
├── 📖 WINDOWS_DEVELOPMENT.md        # Detailed Windows guide
├── 📖 README_AIRGAPPED_RAG.md       # Complete API docs
└── 📖 INSTALL_AIRGAPPED.md          # 5-minute install
```

## 📚 Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[README.md](README.md)** | Project overview | Everyone |
| **[DOCKER_QUICK_START.md](DOCKER_QUICK_START.md)** | Quick commands | Daily use |
| **[WINDOWS_DEVELOPMENT.md](WINDOWS_DEVELOPMENT.md)** | Full dev guide | Windows developers |
| **[INSTALL_AIRGAPPED.md](INSTALL_AIRGAPPED.md)** | Quick install | First-time setup |
| **[README_AIRGAPPED_RAG.md](README_AIRGAPPED_RAG.md)** | Complete docs | Reference |

## 🎯 Features

- ✅ **100% Air-Gapped**: No external API calls, all processing local
- ✅ **Ollama Integration**: Local LLMs (Llama 3) via Ollama
- ✅ **Full Document Retrieval**: No chunking, complete context
- ✅ **Topic-Based Indexing**: One embedding per document
- ✅ **Accurate Citations**: Source URLs and excerpts
- ✅ **Docker Support**: Windows dev → RHEL9 production
- ✅ **FastAPI**: Modern async Python API
- ✅ **ChromaDB**: Embedded vector database

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI | REST API server |
| **LLM** | Ollama (Llama 3) | Text generation |
| **Embeddings** | nomic-embed-text | Vector embeddings |
| **Vector DB** | ChromaDB | Similarity search |
| **PDF Processing** | PyMuPDF/pypdf | Document ingestion |
| **Container** | Docker | Deployment |
| **Base Image** | RHEL UBI9 | Production (RHEL9) |

## 📋 Prerequisites

### Development (Windows)
- Docker Desktop with WSL2
- Python 3.10+
- Git

### Production (RHEL9)
- Docker
- docker-compose
- RHEL9 / Rocky Linux 9

## 🔄 Development Workflow

```
1. Edit code in VS Code (Windows)
        ↓
2. Debug with Ollama in Docker
        ↓
3. Test with example_usage.py
        ↓
4. Build Docker image
        ↓
5. Export for RHEL9
        ↓
6. Transfer to RHEL9
        ↓
7. Deploy with deploy-rhel9.sh
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/upload-document` | POST | Upload PDF with source URL |
| `/query` | POST | Query documents |
| `/documents` | GET | List all documents |
| `/documents/{id}` | DELETE | Delete document |

## 🧪 Testing

```powershell
# Check health
python example_usage.py --health

# Upload document
python example_usage.py --upload test.pdf --url http://example.com/doc

# Query
python example_usage.py --query "What is the PTO policy?"

# Interactive mode
python example_usage.py --interactive
```

## 🔍 Common Commands

### Development

```powershell
# Start Ollama
docker-compose -f docker-compose.ollama.yml up -d

# Run API
python airgapped_rag.py

# Stop Ollama
docker-compose -f docker-compose.ollama.yml down
```

### Production

```bash
# Start services
docker-compose -f docker-compose.airgapped.yml up -d

# View logs
docker-compose -f docker-compose.airgapped.yml logs -f

# Stop services
docker-compose -f docker-compose.airgapped.yml down
```

## 🆘 Troubleshooting

### Cannot connect to Ollama

```powershell
# Check if running
docker ps | grep ollama

# Restart
docker restart ollama-airgapped-rag
```

### Models not found

```bash
# Check models
docker exec ollama-airgapped-rag ollama list

# Pull models
docker exec ollama-airgapped-rag ollama pull nomic-embed-text
docker exec ollama-airgapped-rag ollama pull llama3:8b
```

### Port already in use

```powershell
# Check what's using port 8000
netstat -ano | findstr :8000

# Change port
PORT=8001 python airgapped_rag.py
```

## 🔒 Security

- ✅ Runs as non-root user in containers
- ✅ No external network calls
- ✅ SELinux compatible (RHEL9)
- ✅ Firewall configuration included
- ✅ Data encrypted at rest (optional)

## 📦 System Requirements

### Minimum
- **RAM**: 8 GB
- **Disk**: 10 GB (models + documents)
- **CPU**: 4 cores

### Recommended
- **RAM**: 16 GB
- **Disk**: 50 GB
- **CPU**: 8 cores
- **GPU**: NVIDIA (optional, for faster inference)

## 🤝 Contributing

This is an enterprise RAG system. For issues:
1. Check documentation
2. Review logs
3. Test with minimal example
4. Document steps to reproduce

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- **Ollama** - Local LLM runtime
- **ChromaDB** - Vector database
- **FastAPI** - Web framework
- **PyMuPDF** - PDF processing

## 📞 Support

For detailed documentation, see:
- [Complete API Documentation](README_AIRGAPPED_RAG.md)
- [Windows Development Guide](WINDOWS_DEVELOPMENT.md)
- [Docker Quick Reference](DOCKER_QUICK_START.md)
- [Installation Guide](INSTALL_AIRGAPPED.md)

---

**Version**: 1.0.0
**Last Updated**: 2024-10-24
**Target Environment**: Windows Development → RHEL9 Production
**Air-Gapped**: Yes ✅
