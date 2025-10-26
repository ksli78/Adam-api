# Adam RAG System - Quick Deployment Guide

Complete guide for building, exporting, and deploying Adam in air-gapped environments.

## 🚀 Quick Start (Development Environment)

### One-Command Build

```bash
./build-all.sh
```

This single script:
1. ✅ Pulls Ollama image
2. ✅ Builds Adam API image
3. ✅ Creates Docker network
4. ✅ Creates volumes
5. ✅ Starts Ollama container
6. ✅ Pulls llama3:8b model
7. ✅ Starts Adam API container

**Total time:** 10-15 minutes (first run, includes model download)

### Verify Installation

```bash
# Check health
curl http://localhost:8000/health

# Test system query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "What is your name?"}'
```

## 📦 Export for Air-Gapped Deployment

### Export Images and Create Deployment Package

```bash
./export-images.sh
```

This creates:
- `docker-export/ollama-image.tar.gz` - Ollama image (~1.5GB)
- `docker-export/adam-api-image.tar.gz` - Adam API image (~2GB)
- `docker-export/adam-deployment-YYYYMMDD_HHMMSS.tar.gz` - **Complete package**
- `docker-export/adam-deployment-YYYYMMDD_HHMMSS.sha256` - Checksums

**What's in the deployment package:**
- Compressed Docker images
- Import script (`import-images.sh`)
- Start script (`start-services.sh`)
- Stop script (`stop-services.sh`)
- README with instructions

### Transfer to Production

```bash
# Using SCP
scp docker-export/adam-deployment-*.tar.gz user@production:/opt/adam/

# Using USB drive
cp docker-export/adam-deployment-*.tar.gz /mnt/usb/

# Verify integrity
sha256sum -c adam-deployment-*.sha256
```

## 🎯 Production Deployment (Air-Gapped)

On your production server:

### 1. Extract Package

```bash
tar xzf adam-deployment-YYYYMMDD_HHMMSS.tar.gz
cd adam-deployment-YYYYMMDD_HHMMSS
```

### 2. Make Scripts Executable

```bash
chmod +x *.sh
```

### 3. Import Images

```bash
./import-images.sh
```

This loads both Docker images and creates networks/volumes.

### 4. Start Services

```bash
./start-services.sh
```

This:
- Starts Ollama container
- Pulls/verifies LLM model
- Starts Adam API container
- Verifies both services are healthy

### 5. Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Check services
docker ps

# View logs
docker logs -f adam-api
```

## 🔧 Management Commands

### Development Environment

```bash
# Full rebuild
./build-all.sh

# Export for production
./export-images.sh

# View logs
docker logs -f adam-api
docker logs -f ollama

# Restart services
docker restart adam-api ollama

# Stop services
docker stop adam-api ollama

# Remove everything (DESTRUCTIVE)
docker stop adam-api ollama
docker rm adam-api ollama
docker volume rm adam-data ollama-data
docker network rm adam-network
```

### Production Environment

```bash
# Start services
./start-services.sh

# Stop services
./stop-services.sh

# View logs
docker logs -f adam-api

# Restart API only
docker restart adam-api

# Check status
docker ps
curl http://localhost:8000/health
```

## 📁 File Structure

### Development Environment

```
Adam-api/
├── build-all.sh              # Complete build script
├── export-images.sh          # Export images script
├── Dockerfile                # Adam API Dockerfile
├── docker-compose.yml        # Alternative: docker-compose
├── airgapped_rag_advanced.py # Main API
├── parent_child_store.py     # Hybrid search
└── ...                       # Other modules
```

### Export Directory (after export)

```
docker-export/
├── ollama-image.tar.gz                      # Ollama image
├── adam-api-image.tar.gz                    # Adam API image
├── adam-deployment-YYYYMMDD_HHMMSS.tar.gz   # Complete package
├── adam-deployment-YYYYMMDD_HHMMSS.sha256   # Checksums
└── adam-deployment-YYYYMMDD_HHMMSS/         # Extracted package
    ├── ollama-image.tar.gz
    ├── adam-api-image.tar.gz
    ├── import-images.sh
    ├── start-services.sh
    ├── stop-services.sh
    └── README.txt
```

### Production Environment (after import)

```
adam-deployment-YYYYMMDD_HHMMSS/
├── import-images.sh          # Import script (run once)
├── start-services.sh         # Start services
├── stop-services.sh          # Stop services
├── README.txt                # Instructions
├── ollama-image.tar.gz       # Images (can delete after import)
└── adam-api-image.tar.gz
```

## 🔄 Update Workflow

### Development to Production

```bash
# 1. In development: Rebuild with latest code
./build-all.sh

# 2. Export new images
./export-images.sh

# 3. Transfer to production
scp docker-export/adam-deployment-*.tar.gz user@production:/opt/adam/

# 4. On production: Stop services
./stop-services.sh

# 5. Extract new package
tar xzf adam-deployment-YYYYMMDD_HHMMSS.tar.gz
cd adam-deployment-YYYYMMDD_HHMMSS

# 6. Import updated images
./import-images.sh

# 7. Start services
./start-services.sh
```

## 🗄️ Data Persistence

### Docker Volumes

Data is stored in Docker volumes (survives container recreation):

```bash
# List volumes
docker volume ls | grep adam

# Inspect volume
docker volume inspect adam-data

# Backup data
docker run --rm \
  -v adam-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/adam-backup-$(date +%Y%m%d).tar.gz /data

# Restore data
docker run --rm \
  -v adam-data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/adam-backup-YYYYMMDD.tar.gz -C /
```

### What's Stored

- `adam-data` volume contains:
  - `/data/airgapped_rag/documents/` - Uploaded PDFs
  - `/data/airgapped_rag/chromadb_advanced/` - Vector database
  - `/data/airgapped_rag/feedback.db` - User feedback SQLite

- `ollama-data` volume contains:
  - `/root/.ollama/` - LLM models and configs

## 🛠️ Troubleshooting

### Build Issues

```bash
# Docker not running
sudo systemctl start docker

# Insufficient disk space
docker system prune -a

# Network issues during build
# Check internet connection for model download
```

### Export Issues

```bash
# Export failed
# Check disk space for export directory
df -h

# Missing images
docker images | grep -E "ollama|adam"
# If missing, run: ./build-all.sh
```

### Import/Deployment Issues

```bash
# Import failed
# Check Docker is running
docker ps

# Port already in use
sudo netstat -tulpn | grep 8000
# Stop conflicting service or change port

# Services won't start
docker logs adam-api
docker logs ollama
# Check logs for specific errors
```

### Runtime Issues

```bash
# API not responding
docker logs adam-api
curl http://localhost:8000/health

# Ollama connection issues
docker exec adam-api curl http://ollama:11434/api/version

# Out of memory
docker stats
# Increase Docker memory in settings

# Model not found
docker exec ollama ollama list
docker exec ollama ollama pull llama3:8b
```

## 📊 System Requirements

### Minimum

- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 15GB free
- **OS**: Linux, macOS, Windows (with WSL2)
- **Docker**: 20.10+

### Recommended

- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Disk**: 50GB+ (for multiple models and documents)
- **GPU**: Optional (for faster inference)

## 🔐 Production Checklist

Before deploying to production:

- [ ] Test deployment package in staging environment
- [ ] Verify checksums after transfer (`sha256sum -c`)
- [ ] Configure firewall rules (ports 8000, 11434)
- [ ] Set up reverse proxy with SSL (nginx/traefik)
- [ ] Configure backup strategy for data volumes
- [ ] Set up monitoring (Prometheus, etc.)
- [ ] Document network configuration
- [ ] Plan update/rollback procedure
- [ ] Test disaster recovery process

## 📞 Common Scenarios

### Scenario 1: Fresh Development Setup

```bash
# Clone repository
git clone <repo-url>
cd Adam-api

# Build everything
./build-all.sh

# Test
curl http://localhost:8000/health
```

### Scenario 2: Export for Production

```bash
# In development environment
./export-images.sh

# Transfer
scp docker-export/adam-deployment-*.tar.gz user@prod:/opt/adam/
```

### Scenario 3: Deploy in Air-Gapped Production

```bash
# On production server
cd /opt/adam
tar xzf adam-deployment-*.tar.gz
cd adam-deployment-*/
chmod +x *.sh
./import-images.sh
./start-services.sh
```

### Scenario 4: Update Production

```bash
# Stop current services
./stop-services.sh

# Import new images
./import-images.sh

# Start with new version
./start-services.sh

# Verify
docker ps
curl http://localhost:8000/health
```

## 📝 Environment Variables

Can be set in scripts or passed to containers:

```bash
OLLAMA_HOST=http://ollama:11434    # Ollama service URL
LLM_MODEL=llama3:8b                # LLM model to use
DATA_DIR=/data/airgapped_rag       # Data storage path
```

## 🎓 Additional Resources

- **Full Documentation**: [README.md](README.md)
- **Docker Details**: [DOCKER_BUILD.md](DOCKER_BUILD.md)
- **System Queries**: [SYSTEM_QUERIES.md](SYSTEM_QUERIES.md)
- **Advanced RAG**: [README_ADVANCED_RAG.md](README_ADVANCED_RAG.md)

---

**Quick Reference Card:**

```bash
# Development
./build-all.sh              # Build everything
./export-images.sh          # Export for production
docker logs -f adam-api     # View logs

# Production
./import-images.sh          # Import images (once)
./start-services.sh         # Start services
./stop-services.sh          # Stop services
docker logs -f adam-api     # View logs
```
