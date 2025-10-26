# Docker Build and Deployment Guide

Complete guide for building and deploying the Adam RAG API using Docker.

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM recommended
- 10GB+ free disk space (for models and data)

## 🎯 Quick Start (Ollama Already Running)

If you already have Ollama running in a container, use this approach:

### 1. Build the Adam API Image

```bash
docker build -t adam-api:latest .
```

### 2. Connect to Existing Ollama Container

```bash
# Find your Ollama container name
docker ps | grep ollama

# Create a shared network (if not exists)
docker network create adam-network

# Connect your existing Ollama container to the network
docker network connect adam-network <your-ollama-container-name>

# Run Adam API connected to existing Ollama
docker run -d \
  --name adam-api \
  --network adam-network \
  -p 8000:8000 \
  -e OLLAMA_HOST=http://<your-ollama-container-name>:11434 \
  -e LLM_MODEL=llama3:8b \
  -v adam-data:/data/airgapped_rag \
  adam-api:latest
```

### 3. Verify

```bash
# Check logs
docker logs -f adam-api

# Test API
curl http://localhost:8000/health
```

## 🚀 Full Stack Deployment (New Setup)

If you don't have Ollama running or want a fresh deployment:

### 1. Build and Start All Services

```bash
docker-compose up -d
```

This starts:
- **adam-api** on port 8000
- **ollama** on port 11434

### 2. Pull the LLM Model

```bash
# Wait for Ollama to start (check logs)
docker-compose logs -f ollama

# Pull the model (first time only, ~4.5GB)
docker exec ollama ollama pull llama3:8b
```

### 3. Verify Deployment

```bash
# Check all services are running
docker-compose ps

# View Adam API logs
docker-compose logs -f adam-api

# Test the API
curl http://localhost:8000/health
```

### 4. Access the API

- **API Endpoint**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📂 Directory Structure

```
/data/airgapped_rag/
├── documents/           # Uploaded PDFs
├── chromadb_advanced/   # Vector database
└── feedback.db          # User feedback SQLite
```

## 🔧 Configuration

### Environment Variables

Edit `docker-compose.yml` or pass via command line:

```yaml
environment:
  - DATA_DIR=/data/airgapped_rag          # Data storage location
  - OLLAMA_HOST=http://ollama:11434       # Ollama service URL
  - LLM_MODEL=llama3:8b                   # Model to use
```

### Custom Configuration

```bash
docker run -d \
  --name adam-api \
  -p 8000:8000 \
  -e OLLAMA_HOST=http://custom-ollama:11434 \
  -e LLM_MODEL=llama3.1:8b \
  -e DATA_DIR=/custom/data/path \
  -v /host/data:/custom/data/path \
  adam-api:latest
```

## 🗄️ Data Persistence

### Using Docker Volumes (Recommended)

```bash
# Data persists in Docker volumes
docker volume ls | grep adam

# Inspect volume
docker volume inspect adam-data

# Backup volume
docker run --rm -v adam-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/adam-data-backup.tar.gz /data
```

### Using Host Directories

Edit `docker-compose.yml`:

```yaml
volumes:
  # Replace named volume with host path
  - /path/on/host:/data/airgapped_rag
```

## 🔄 Updates and Maintenance

### Update the API

```bash
# Pull latest code
git pull

# Rebuild image
docker-compose build adam-api

# Restart with new image
docker-compose up -d adam-api
```

### Update Ollama Model

```bash
# Pull new model version
docker exec ollama ollama pull llama3:8b

# Or use a different model
docker exec ollama ollama pull llama3.1:8b

# Update environment variable
docker-compose down
# Edit docker-compose.yml: LLM_MODEL=llama3.1:8b
docker-compose up -d
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f adam-api
docker-compose logs -f ollama

# Last 100 lines
docker-compose logs --tail=100 adam-api
```

## 🛠️ Troubleshooting

### API Won't Start

```bash
# Check logs for errors
docker logs adam-api

# Common issues:
# 1. Ollama not accessible
docker exec adam-api curl http://ollama:11434

# 2. Port already in use
docker ps | grep 8000
# Change port in docker-compose.yml if needed

# 3. Volume permissions
docker exec adam-api ls -la /data/airgapped_rag
```

### Ollama Connection Issues

```bash
# Verify Ollama is running
docker ps | grep ollama

# Test Ollama from API container
docker exec adam-api curl http://ollama:11434/api/version

# Check network connectivity
docker network inspect adam-network
```

### Models Not Downloading

```bash
# Check Ollama logs
docker logs ollama

# Manually pull model
docker exec -it ollama ollama pull llama3:8b

# Verify model is available
docker exec ollama ollama list
```

### Out of Memory

```bash
# Check memory usage
docker stats

# Increase Docker memory limit (Docker Desktop)
# Settings → Resources → Memory → 8GB+

# Or limit service memory in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 6G
```

## 🧪 Testing the Deployment

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "version": "2.0.0"}
```

### 2. Upload a Test Document

```bash
curl -X POST "http://localhost:8000/upload-document" \
  -F "file=@test.pdf" \
  -F "source_url=https://example.com/test.pdf"
```

### 3. Query the System

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is your name?",
    "use_hybrid": true
  }'
```

### 4. Check Statistics

```bash
curl http://localhost:8000/statistics
```

## 🔐 Production Considerations

### Security

1. **Reverse Proxy**: Use nginx/traefik for SSL/TLS
2. **Authentication**: Add API key or OAuth
3. **Network Isolation**: Use Docker networks
4. **Firewall Rules**: Restrict access to port 8000

Example with nginx:
```nginx
server {
    listen 443 ssl;
    server_name adam.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Resource Limits

```yaml
# In docker-compose.yml
services:
  adam-api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          memory: 2G
```

### Backup Strategy

```bash
# Automated daily backup
0 2 * * * docker run --rm -v adam-data:/data -v /backup:/backup \
  alpine tar czf /backup/adam-$(date +\%Y\%m\%d).tar.gz /data
```

### Monitoring

```yaml
# Add to docker-compose.yml
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## 📊 Performance Tuning

### For High-Load Environments

```bash
# Multiple workers (not recommended for air-gapped due to model loading)
# Edit run_advanced.py:
# workers=4

# Or use gunicorn
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker airgapped_rag_advanced:app
```

### GPU Acceleration (Optional)

```yaml
# In docker-compose.yml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 🗑️ Cleanup

### Stop Services

```bash
# Stop but keep data
docker-compose down

# Stop and remove volumes (DELETES ALL DATA)
docker-compose down -v
```

### Remove Everything

```bash
# Remove containers, networks, volumes
docker-compose down -v

# Remove images
docker rmi adam-api:latest

# Remove orphaned volumes
docker volume prune
```

## 📝 Development Setup

### Local Development with Hot Reload

```bash
# Build image
docker build -t adam-api:dev .

# Run with code mounted (for development)
docker run -d \
  --name adam-api-dev \
  -p 8000:8000 \
  -v $(pwd):/app \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  adam-api:dev \
  python -m uvicorn airgapped_rag_advanced:app --host 0.0.0.0 --reload
```

## 🎓 Common Use Cases

### Use Case 1: Connect to Existing Ollama

Your scenario - Ollama already running:

```bash
# Build API
docker build -t adam-api:latest .

# Find Ollama container name
OLLAMA_CONTAINER=$(docker ps --filter "ancestor=ollama/ollama" --format "{{.Names}}")

# Create network
docker network create adam-network
docker network connect adam-network $OLLAMA_CONTAINER

# Run API
docker run -d \
  --name adam-api \
  --network adam-network \
  -p 8000:8000 \
  -e OLLAMA_HOST=http://$OLLAMA_CONTAINER:11434 \
  -v adam-data:/data/airgapped_rag \
  adam-api:latest
```

### Use Case 2: Fresh Installation

```bash
docker-compose up -d
docker exec ollama ollama pull llama3:8b
```

### Use Case 3: Multiple Environments

```bash
# Development
docker-compose -f docker-compose.dev.yml up -d

# Production
docker-compose -f docker-compose.prod.yml up -d
```

## 📞 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify connectivity: `docker exec adam-api curl http://ollama:11434`
3. Check resources: `docker stats`
4. Review documentation: http://localhost:8000/docs

---

**Adam v2.0** - Dockerized deployment guide
