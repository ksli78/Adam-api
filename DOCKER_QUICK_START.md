# Docker Quick Start - Air-Gapped RAG API

Quick reference for Windows development and RHEL9 deployment.

## Windows Development (3 Steps)

### 1. Start Ollama in Docker

```powershell
# Start Ollama
docker-compose -f docker-compose.ollama.yml up -d

# Pull models (using Git Bash)
bash setup-ollama-models.sh

# OR manually (PowerShell)
docker exec -it ollama-airgapped-rag ollama pull nomic-embed-text
docker exec -it ollama-airgapped-rag ollama pull llama3:8b
```

### 2. Run API Locally

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run API (connects to Docker Ollama)
python airgapped_rag.py
```

### 3. Test It

```powershell
# New terminal
python example_usage.py --health
python example_usage.py --upload test.pdf --url http://test.com
python example_usage.py --query "What is this about?"
```

**API Docs:** http://localhost:8000/docs

---

## Export for RHEL9

```powershell
# 1. Make sure Ollama is running with models
docker ps  # Should see ollama container

# 2. Run export script
.\export-for-rhel9.ps1

# 3. Transfer rhel9-deployment folder to RHEL9
# (USB drive, SCP, network share, etc.)
```

---

## RHEL9 Deployment (1 Command)

```bash
cd rhel9-deployment
sudo bash deploy-rhel9.sh
```

**Done!** API runs at http://localhost:8000

---

## Quick Commands Reference

### Windows - Development

```powershell
# Start/Stop Ollama only
docker-compose -f docker-compose.ollama.yml up -d
docker-compose -f docker-compose.ollama.yml down

# Check Ollama
docker exec ollama-airgapped-rag ollama list
curl http://localhost:11434/api/tags

# Run API locally
python airgapped_rag.py

# Debug in VS Code
# Press F5 (after setting up launch.json)
```

### Windows - Full Stack Test

```powershell
# Build and start everything
docker build -f Dockerfile.airgapped -t airgapped-rag-api:latest .
docker-compose -f docker-compose.airgapped.yml up -d

# Pull models into production Ollama
docker exec -it ollama-service ollama pull nomic-embed-text
docker exec -it ollama-service ollama pull llama3:8b

# Check logs
docker-compose -f docker-compose.airgapped.yml logs -f

# Stop
docker-compose -f docker-compose.airgapped.yml down
```

### RHEL9 - Management

```bash
# Start/Stop services
docker-compose -f docker-compose.airgapped.yml up -d
docker-compose -f docker-compose.airgapped.yml down

# View logs
docker-compose -f docker-compose.airgapped.yml logs -f
docker-compose -f docker-compose.airgapped.yml logs -f airgapped-rag-api

# Check status
docker-compose -f docker-compose.airgapped.yml ps
curl http://localhost:8000/health

# Restart single service
docker-compose -f docker-compose.airgapped.yml restart airgapped-rag-api
```

### Useful Docker Commands

```bash
# List images
docker images

# List volumes
docker volume ls

# Inspect volume
docker volume inspect airgapped_ollama_models

# Remove volume (careful!)
docker volume rm airgapped_ollama_models

# Check logs
docker logs ollama-service
docker logs airgapped-rag-api

# Execute command in container
docker exec -it ollama-service bash
docker exec airgapped-rag-api curl localhost:8000/health
```

---

## Troubleshooting

### Ollama not responding

```bash
# Check if running
docker ps | grep ollama

# Restart
docker restart ollama-airgapped-rag  # (dev)
docker restart ollama-service        # (production)

# Check logs
docker logs ollama-airgapped-rag
```

### API can't connect to Ollama

Check environment variable:
```bash
# Should be http://localhost:11434 (local dev)
# Should be http://ollama:11434 (in Docker)
echo $OLLAMA_BASE_URL
```

### Models missing after restart

```bash
# Models are in Docker volumes - they persist
docker volume ls | grep ollama

# Check if models are there
docker exec ollama-service ollama list

# If empty, pull again
docker exec ollama-service ollama pull nomic-embed-text
docker exec ollama-service ollama pull llama3:8b
```

### Port already in use

```bash
# Find what's using port
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # Linux/Mac

# Change port in .env or docker-compose
PORT=8001 python airgapped_rag.py
```

---

## Architecture

```
Windows Development:
┌─────────────────┐
│ VS Code/Python  │  <- Debug here
│ airgapped_rag.py│
│ (localhost:8000)│
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│ Docker:         │
│ Ollama          │  <- Models here
│ (localhost:11434)│
└─────────────────┘

RHEL9 Production:
┌─────────────────┐
│ Docker:         │
│ airgapped-rag-api│
│ (localhost:8000)│
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│ Docker:         │
│ Ollama          │
│ (internal:11434)│
└─────────────────┘
```

---

## File Organization

```
Adam-api/
├── airgapped_rag.py                    # Main API application
├── example_usage.py                    # CLI testing tool
├── requirements.txt                    # Python dependencies
│
├── docker-compose.ollama.yml          # Ollama only (dev)
├── docker-compose.airgapped.yml       # Full stack (production)
├── Dockerfile.airgapped               # API container
│
├── setup-ollama-models.sh             # Pull models (Bash)
├── export-for-rhel9.ps1               # Export package (PowerShell)
├── deploy-rhel9.sh                    # Deploy on RHEL9 (Bash)
│
├── README_AIRGAPPED_RAG.md            # Full documentation
├── INSTALL_AIRGAPPED.md               # Quick start
├── WINDOWS_DEVELOPMENT.md             # This guide (detailed)
└── DOCKER_QUICK_START.md              # This file
```

---

## Common Workflows

### 1. Daily Development

```powershell
# Morning
docker-compose -f docker-compose.ollama.yml up -d
.\venv\Scripts\Activate.ps1
python airgapped_rag.py

# Work, edit, test, debug (F5 in VS Code)

# Evening
docker-compose -f docker-compose.ollama.yml down
```

### 2. Test Full Stack

```powershell
docker build -f Dockerfile.airgapped -t airgapped-rag-api:latest .
docker-compose -f docker-compose.airgapped.yml up -d
# Test...
docker-compose -f docker-compose.airgapped.yml down
```

### 3. Prepare for Deployment

```powershell
# 1. Ensure models are downloaded
docker exec ollama-airgapped-rag ollama list

# 2. Build latest API image
docker build -f Dockerfile.airgapped -t airgapped-rag-api:latest .

# 3. Export
.\export-for-rhel9.ps1

# 4. Transfer rhel9-deployment/ to RHEL9
```

### 4. Deploy on RHEL9

```bash
# 1. Copy deployment package
scp -r rhel9-deployment/ user@rhel9:/home/user/

# 2. SSH and deploy
ssh user@rhel9
cd rhel9-deployment
sudo bash deploy-rhel9.sh

# 3. Verify
curl http://localhost:8000/health
python3 example_usage.py --health
```

---

## Environment Variables

| Variable | Dev (Windows) | Prod (RHEL9) |
|----------|--------------|--------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `http://ollama:11434` |
| `DATA_DIR` | `C:\Users\...\data` | `/data/airgapped_rag` |
| `HOST` | `127.0.0.1` | `0.0.0.0` |
| `PORT` | `8000` | `8000` |

---

## Performance Tips

- **Use GPU**: Uncomment GPU sections in docker-compose files
- **Increase RAM**: Allocate more to Docker Desktop (Settings → Resources)
- **Use smaller model**: Try `llama2:7b` instead of `llama3:8b`
- **Limit top_k**: Use `top_k=1` in queries for faster responses

---

## Security Checklist

### Windows Development
- [ ] Use virtual environment
- [ ] Don't commit `.env` files
- [ ] Keep Docker Desktop updated

### RHEL9 Production
- [ ] Run as non-root user (already configured)
- [ ] Configure firewalld appropriately
- [ ] Keep SELinux enforcing
- [ ] Regular backups of `/data` volume
- [ ] Monitor disk space (models + documents)
- [ ] Set up log rotation

---

## Support

- **Full docs**: [README_AIRGAPPED_RAG.md](README_AIRGAPPED_RAG.md)
- **Installation**: [INSTALL_AIRGAPPED.md](INSTALL_AIRGAPPED.md)
- **Windows dev**: [WINDOWS_DEVELOPMENT.md](WINDOWS_DEVELOPMENT.md)
- **This guide**: [DOCKER_QUICK_START.md](DOCKER_QUICK_START.md)

---

**Last Updated:** 2024-10-24
