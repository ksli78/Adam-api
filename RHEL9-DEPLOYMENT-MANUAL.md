# RHEL9 Production Deployment - Manual Guide

This guide shows how to deploy Adam RAG System on RHEL9 after uploading .tar.gz files via Portainer.

**No file uploads needed** - Choose either SSH commands OR Portainer UI.

---

## Option 1: Direct SSH Commands (Recommended)

SSH into your RHEL9 server and run these commands in order.

### Prerequisites

Assume your .tar.gz files are in `/opt/adam` (adjust path as needed):
- `ollama-image.tar.gz`
- `adam-api-image.tar.gz`

### Step 1: Import Docker Images

```bash
# Navigate to directory with .tar.gz files
cd /opt/adam

# Import Ollama image
gunzip -c ollama-image.tar.gz | docker load

# Import Adam API image
gunzip -c adam-api-image.tar.gz | docker load

# Verify images loaded
docker images | grep -E "ollama|adam"
```

Expected output:
```
ollama/ollama    latest    ...
adam-api         latest    ...
```

### Step 2: Create Docker Network

```bash
docker network create adam-network
```

### Step 3: Create Docker Volumes

```bash
docker volume create adam-data
docker volume create ollama-data
```

### Step 4: Start Ollama Container

```bash
docker run -d \
  --name ollama \
  --network adam-network \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  --restart unless-stopped \
  ollama/ollama:latest
```

Wait for Ollama to be ready:
```bash
# Wait ~10 seconds
sleep 10

# Check if ready
curl http://localhost:11434/api/version
```

### Step 5: Pull LLM Model

```bash
# Pull llama3:8b model (~4.5GB, takes several minutes)
docker exec ollama ollama pull llama3:8b

# Verify model is available
docker exec ollama ollama list
```

### Step 6: Start Adam API Container

```bash
docker run -d \
  --name adam-api \
  --network adam-network \
  -p 8000:8000 \
  -e OLLAMA_HOST=http://ollama:11434 \
  -e LLM_MODEL=llama3:8b \
  -e DATA_DIR=/data/airgapped_rag \
  -v adam-data:/data/airgapped_rag \
  --restart unless-stopped \
  adam-api:latest
```

### Step 7: Configure RHEL9 Firewall

```bash
# Open required ports
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=11434/tcp
sudo firewall-cmd --reload

# Verify
sudo firewall-cmd --list-ports
```

### Step 8: Enable Docker on Boot

```bash
sudo systemctl enable docker
```

### Step 9: Verify Deployment

```bash
# Check containers are running
docker ps

# Test API health
curl http://localhost:8000/health

# Test with query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "What is your name?"}'
```

### Step 10: Get Server IP for Remote Access

```bash
# Get server IP
hostname -I | awk '{print $1}'
```

Access from remote browser:
- API: `http://<server-ip>:8000`
- API Docs: `http://<server-ip>:8000/docs`
- Ollama: `http://<server-ip>:11434`

---

## Option 2: Portainer UI Setup

### Step 1: Import Images

1. **Navigate to Images**
   - Left menu → Images
   - Click "Import" button

2. **Import Ollama Image**
   - Click "Upload"
   - Select `ollama-image.tar.gz`
   - Wait for upload and import (may take several minutes)
   - Verify `ollama/ollama:latest` appears in images list

3. **Import Adam API Image**
   - Click "Import" again
   - Select `adam-api-image.tar.gz`
   - Wait for upload and import
   - Verify `adam-api:latest` appears in images list

### Step 2: Create Network

1. **Navigate to Networks**
   - Left menu → Networks
   - Click "Add network"

2. **Configure Network**
   - Name: `adam-network`
   - Driver: `bridge`
   - Click "Create the network"

### Step 3: Create Volumes

1. **Navigate to Volumes**
   - Left menu → Volumes
   - Click "Add volume"

2. **Create adam-data Volume**
   - Name: `adam-data`
   - Driver: `local`
   - Click "Create the volume"

3. **Create ollama-data Volume**
   - Name: `ollama-data`
   - Driver: `local`
   - Click "Create the volume"

### Step 4: Create Ollama Container

1. **Navigate to Containers**
   - Left menu → Containers
   - Click "Add container"

2. **Basic Configuration**
   - Name: `ollama`
   - Image: `ollama/ollama:latest`

3. **Network & Ports**
   - Network: `adam-network`
   - Port mapping:
     - Host: `11434` → Container: `11434`

4. **Volumes**
   - Click "map additional volume"
   - Container: `/root/.ollama`
   - Volume: `ollama-data`

5. **Restart Policy**
   - Restart policy: `Unless stopped`

6. **Deploy**
   - Click "Deploy the container"
   - Wait for container to start (green status)

### Step 5: Pull LLM Model in Ollama

1. **Access Ollama Console**
   - In Containers list, click `ollama`
   - Click "Console" tab
   - Click "Connect" (use `/bin/sh` or `/bin/bash`)

2. **Pull Model**
   ```bash
   ollama pull llama3:8b
   ```
   Wait for download to complete (~4.5GB)

3. **Verify**
   ```bash
   ollama list
   ```
   Should show `llama3:8b`

4. **Disconnect Console**

### Step 6: Create Adam API Container

1. **Navigate to Containers**
   - Left menu → Containers
   - Click "Add container"

2. **Basic Configuration**
   - Name: `adam-api`
   - Image: `adam-api:latest`

3. **Network & Ports**
   - Network: `adam-network`
   - Port mapping:
     - Host: `8000` → Container: `8000`

4. **Environment Variables**
   Click "add environment variable" for each:
   - Name: `OLLAMA_HOST` → Value: `http://ollama:11434`
   - Name: `LLM_MODEL` → Value: `llama3:8b`
   - Name: `DATA_DIR` → Value: `/data/airgapped_rag`

5. **Volumes**
   - Click "map additional volume"
   - Container: `/data/airgapped_rag`
   - Volume: `adam-data`

6. **Restart Policy**
   - Restart policy: `Unless stopped`

7. **Deploy**
   - Click "Deploy the container"
   - Wait for container to start (green status)

### Step 7: Verify in Portainer

1. **Check Container Status**
   - Both containers should show green "running" status
   - Check logs for any errors

2. **View Adam API Logs**
   - Click `adam-api` container
   - Click "Logs" tab
   - Should see: "Application startup complete"
   - Should see: "Uvicorn running on http://0.0.0.0:8000"

3. **View Ollama Logs**
   - Click `ollama` container
   - Click "Logs" tab
   - Should see Ollama service running

### Step 8: Test Deployment

Open browser and navigate to:
- Health Check: `http://<server-ip>:8000/health`
- API Documentation: `http://<server-ip>:8000/docs`

Or use curl from SSH:
```bash
curl http://localhost:8000/health
```

### Step 9: Configure Firewall (via SSH)

Still need to open firewall ports via SSH:
```bash
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=11434/tcp
sudo firewall-cmd --reload
```

---

## Management Commands (SSH)

After deployment, manage via SSH:

### View Logs
```bash
# API logs
docker logs -f adam-api

# Ollama logs
docker logs -f ollama

# Last 100 lines
docker logs --tail 100 adam-api
```

### Restart Services
```bash
# Restart API
docker restart adam-api

# Restart Ollama
docker restart ollama

# Restart both
docker restart adam-api ollama
```

### Stop Services
```bash
docker stop adam-api ollama
```

### Start Services
```bash
docker start ollama
sleep 5
docker start adam-api
```

### Check Status
```bash
# Container status
docker ps

# Health check
curl http://localhost:8000/health

# Ollama version
curl http://localhost:11434/api/version
```

### Backup Data
```bash
# Create backup directory
mkdir -p /backup/adam

# Backup adam-data volume
docker run --rm \
  -v adam-data:/data \
  -v /backup/adam:/backup \
  alpine tar czf /backup/adam-backup-$(date +%Y%m%d).tar.gz /data

# Backup ollama-data volume (includes models)
docker run --rm \
  -v ollama-data:/data \
  -v /backup/adam:/backup \
  alpine tar czf /backup/ollama-backup-$(date +%Y%m%d).tar.gz /data
```

### Restore Data
```bash
# Restore adam-data
docker run --rm \
  -v adam-data:/data \
  -v /backup/adam:/backup \
  alpine tar xzf /backup/adam-backup-YYYYMMDD.tar.gz -C /

# Restore ollama-data
docker run --rm \
  -v ollama-data:/data \
  -v /backup/adam:/backup \
  alpine tar xzf /backup/ollama-backup-YYYYMMDD.tar.gz -C /
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs adam-api
docker logs ollama

# Check if port is already in use
sudo netstat -tulpn | grep 8000
sudo netstat -tulpn | grep 11434

# Remove and recreate container
docker stop adam-api
docker rm adam-api
# Then run the docker run command again
```

### API Returns "Ollama Connection Error"

```bash
# Check Ollama is running
docker ps | grep ollama

# Check Ollama is responding
curl http://localhost:11434/api/version

# Check they're on same network
docker network inspect adam-network

# Restart API
docker restart adam-api
```

### Model Not Found

```bash
# Check model is pulled
docker exec ollama ollama list

# If not, pull it
docker exec ollama ollama pull llama3:8b
```

### Cannot Access from Remote Browser

```bash
# Check firewall
sudo firewall-cmd --list-ports

# Should show: 8000/tcp 11434/tcp

# If not, add them
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=11434/tcp
sudo firewall-cmd --reload
```

### Disk Space Issues

```bash
# Check disk usage
df -h

# Clean up unused Docker resources
docker system prune -a

# Check volume sizes
docker system df -v
```

---

## Quick Reference

### Where Files Are Stored

- **Documents**: In `adam-data` volume at `/data/airgapped_rag/documents/`
- **Vector DB**: In `adam-data` volume at `/data/airgapped_rag/chromadb_advanced/`
- **Feedback DB**: In `adam-data` volume at `/data/airgapped_rag/feedback.db`
- **LLM Models**: In `ollama-data` volume at `/root/.ollama/`

### Upload Documents

```bash
# Access adam-data volume
docker run -it --rm -v adam-data:/data alpine sh

# Inside container, navigate to documents
cd /data/airgapped_rag/documents
ls -la

# Exit container
exit
```

Or use Portainer:
1. Click `adam-api` container
2. Click "Volumes" tab
3. Browse volume contents

### Complete Removal (Start Fresh)

```bash
# Stop and remove containers
docker stop adam-api ollama
docker rm adam-api ollama

# Remove volumes (⚠️ DELETES ALL DATA)
docker volume rm adam-data ollama-data

# Remove network
docker network rm adam-network

# Remove images
docker rmi adam-api:latest ollama/ollama:latest
```

---

## Summary

**Option 1 (SSH)**: Copy-paste commands from "Option 1" section - takes ~10 minutes

**Option 2 (Portainer)**: Follow step-by-step UI guide - takes ~15 minutes

Both methods result in the same deployment:
- Ollama running on port 11434
- Adam API running on port 8000
- Data persisted in volumes
- Auto-restart enabled
- Firewall configured

**Access Points**:
- API: `http://<server-ip>:8000`
- API Docs: `http://<server-ip>:8000/docs`
- Health: `http://<server-ip>:8000/health`
