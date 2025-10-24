# Windows Development Guide - Air-Gapped RAG API

This guide walks you through setting up Ollama in Docker on Windows for local development, then packaging everything for deployment on RHEL9.

## Prerequisites

### Required Software on Windows

1. **Docker Desktop for Windows** (with WSL2 backend)
   - Download: https://www.docker.com/products/docker-desktop/
   - Make sure WSL2 integration is enabled

2. **Python 3.10+**
   - Download: https://www.python.org/downloads/
   - Add to PATH during installation

3. **Git**
   - Download: https://git-scm.com/download/win
   - Git Bash will be useful for running shell scripts

4. **VS Code** (already installed)
   - Install Python extension
   - Install Docker extension

## Part 1: Local Development Setup (Windows)

### Step 1: Start Ollama in Docker

Open PowerShell or Command Prompt in your project directory:

```powershell
# Start Ollama container
docker-compose -f docker-compose.ollama.yml up -d

# Check if it's running
docker ps
```

You should see a container named `ollama-airgapped-rag` running on port 11434.

### Step 2: Pull Ollama Models

**Option A: Using Git Bash**
```bash
bash setup-ollama-models.sh
```

**Option B: Manual commands (PowerShell/CMD)**
```powershell
# Pull embedding model (~274 MB)
docker exec -it ollama-airgapped-rag ollama pull nomic-embed-text

# Pull LLM model (~4.7 GB) - this will take a while
docker exec -it ollama-airgapped-rag ollama pull llama3:8b

# Verify models are installed
docker exec ollama-airgapped-rag ollama list
```

**Expected output:**
```
NAME                    ID              SIZE    MODIFIED
nomic-embed-text:latest xxxxx          274MB   X seconds ago
llama3:8b              xxxxx          4.7GB   X seconds ago
```

### Step 3: Set Up Python Environment

In VS Code terminal or PowerShell:

```powershell
# Create virtual environment
python -m venv venv

# Activate it (PowerShell)
.\venv\Scripts\Activate.ps1

# Or if using Command Prompt
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Configure Environment for Local Development

Create a `.env.local` file in your project root:

```bash
# Ollama in Docker
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3:8b

# Local data directory (Windows path)
DATA_DIR=C:\Users\YourUsername\airgapped_rag_data

# API configuration
HOST=127.0.0.1
PORT=8000
```

**Note:** Update `DATA_DIR` to a real path on your system.

### Step 5: Run the API Locally

```powershell
# Make sure venv is activated
python airgapped_rag.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 6: Test the API

Open a new terminal and test:

```powershell
# Check health
curl http://localhost:8000/health

# Or use the example script
python example_usage.py --health
```

Open in browser: http://localhost:8000/docs

### Step 7: Debug in VS Code

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Air-Gapped RAG API",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/airgapped_rag.py",
      "console": "integratedTerminal",
      "env": {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "DATA_DIR": "C:\\Users\\YourUsername\\airgapped_rag_data"
      },
      "justMyCode": true
    }
  ]
}
```

Now you can:
1. Set breakpoints in `airgapped_rag.py`
2. Press F5 to start debugging
3. Test API calls and step through code

## Part 2: Containerize the API for Production

### Step 1: Build the API Docker Image

```powershell
# Build the image
docker build -f Dockerfile.airgapped -t airgapped-rag-api:latest .

# Verify it was built
docker images | grep airgapped-rag-api
```

### Step 2: Test Full Stack Locally

```powershell
# Stop the local API if running
# Stop Ollama development container
docker-compose -f docker-compose.ollama.yml down

# Start both services together
docker-compose -f docker-compose.airgapped.yml up -d

# Check logs
docker-compose -f docker-compose.airgapped.yml logs -f

# Test the API
curl http://localhost:8000/health
```

### Step 3: Models are Already in Ollama Volume

The models you pulled in Part 1 are stored in a Docker volume. When you start the production stack, they need to be pulled again into the new Ollama container.

```powershell
# Pull models into the production Ollama container
docker exec -it ollama-service ollama pull nomic-embed-text
docker exec -it ollama-service ollama pull llama3:8b

# Verify
docker exec ollama-service ollama list
```

## Part 3: Export for RHEL9 Air-Gapped Deployment

### Step 1: Save Docker Images

```powershell
# Create export directory
mkdir docker-images

# Save Ollama image
docker pull ollama/ollama:latest
docker save ollama/ollama:latest -o docker-images/ollama.tar

# Save your API image
docker save airgapped-rag-api:latest -o docker-images/airgapped-rag-api.tar

# Save Python base image (for RHEL9 compatibility)
docker pull registry.access.redhat.com/ubi9/python-311:latest
docker save registry.access.redhat.com/ubi9/python-311:latest -o docker-images/ubi9-python.tar
```

### Step 2: Export Ollama Models

```powershell
# Get the volume path
docker volume inspect airgapped_ollama_models

# Copy models from Docker volume
# Note: On Windows with WSL2, Docker volumes are in WSL2 filesystem
# Access via \\wsl$\docker-desktop-data\data\docker\volumes\

# Create models export directory
mkdir ollama-models-export

# Export using a temporary container
docker run --rm -v airgapped_ollama_models:/models -v ${PWD}/ollama-models-export:/backup alpine tar czf /backup/ollama-models.tar.gz -C /models .
```

### Step 3: Package Everything

```powershell
# Create deployment package
mkdir rhel9-deployment
cd rhel9-deployment

# Copy Docker images
cp ../docker-images/*.tar .

# Copy Ollama models
cp ../ollama-models-export/ollama-models.tar.gz .

# Copy deployment files
cp ../docker-compose.airgapped.yml .
cp ../Dockerfile.airgapped .
cp ../airgapped_rag.py .
cp ../requirements.txt .
cp ../example_usage.py .

# Create README for RHEL9 deployment
```

### Step 4: Transfer to RHEL9

Use one of these methods:

**Option A: USB Drive**
```powershell
# Compress everything
Compress-Archive -Path rhel9-deployment -DestinationPath airgapped-rag-rhel9.zip

# Copy to USB drive
# Then physically transfer to RHEL9 system
```

**Option B: Secure Copy (if network available)**
```powershell
scp -r rhel9-deployment user@rhel9-server:/home/user/
```

**Option C: Internal File Share**
```powershell
# Copy to network share accessible from RHEL9
Copy-Item -Recurse rhel9-deployment \\internal-share\deployments\
```

## Part 4: Deploy on RHEL9 (Air-Gapped)

SSH into your RHEL9 system and follow these steps:

### Step 1: Prepare RHEL9 System

```bash
# Check Docker is installed
docker --version

# If not installed:
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect

# Install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Step 2: Load Docker Images

```bash
cd /home/user/rhel9-deployment

# Load images
docker load -i ollama.tar
docker load -i airgapped-rag-api.tar
docker load -i ubi9-python.tar

# Verify
docker images
```

### Step 3: Restore Ollama Models

```bash
# Create volume
docker volume create airgapped_ollama_models

# Extract models into volume
docker run --rm -v airgapped_ollama_models:/models -v $(pwd):/backup alpine tar xzf /backup/ollama-models.tar.gz -C /models

# Verify volume
docker volume inspect airgapped_ollama_models
```

### Step 4: Start Services

```bash
# Create data directory
sudo mkdir -p /data/airgapped_rag
sudo chown $USER:$USER /data/airgapped_rag

# Start services
docker-compose -f docker-compose.airgapped.yml up -d

# Check status
docker-compose -f docker-compose.airgapped.yml ps

# View logs
docker-compose -f docker-compose.airgapped.yml logs -f
```

### Step 5: Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Test with example script
python3 example_usage.py --health

# Upload a test document
python3 example_usage.py --upload test.pdf --url http://example.com/test

# Query
python3 example_usage.py --query "What is this document about?"
```

## Troubleshooting

### Windows Development Issues

**Issue: Docker Desktop not starting**
- Enable WSL2 in Windows Features
- Update Windows to latest version
- Check Docker Desktop settings → Resources → WSL Integration

**Issue: Permission denied on scripts**
```powershell
# Use Git Bash or WSL to run .sh scripts
# Or use PowerShell equivalents shown above
```

**Issue: Port 11434 already in use**
```powershell
# Check what's using the port
netstat -ano | findstr :11434

# Stop any existing Ollama process
# Or change port in docker-compose.ollama.yml
```

**Issue: Cannot connect to Ollama**
```powershell
# Check if container is running
docker ps

# Check container logs
docker logs ollama-airgapped-rag

# Restart container
docker restart ollama-airgapped-rag
```

### RHEL9 Deployment Issues

**Issue: Docker daemon not running**
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

**Issue: Permission denied**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

**Issue: Models not found**
```bash
# Check if models were restored
docker exec ollama-service ollama list

# If empty, models didn't restore properly
# Re-extract from tar.gz or pull models again if internet available
```

**Issue: SELinux denials**
```bash
# Check SELinux status
getenforce

# If Enforcing and causing issues, check logs
sudo ausearch -m avc -ts recent

# Temporary workaround (not recommended for production)
sudo setenforce 0

# Better: Create proper SELinux policy or use semanage
```

## Performance Tuning

### Enable GPU Support (if available)

**Windows (NVIDIA GPU with WSL2):**
1. Install NVIDIA drivers for Windows
2. Install NVIDIA Container Toolkit in WSL2
3. Uncomment GPU sections in docker-compose files

**RHEL9 (NVIDIA GPU):**
```bash
# Install NVIDIA container toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
  sudo tee /etc/yum.repos.d/nvidia-docker.repo

sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is accessible
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Memory Limits

Add to docker-compose.airgapped.yml:

```yaml
services:
  airgapped-rag-api:
    # ... other config ...
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

## Development Workflow

1. **Edit code** in VS Code on Windows
2. **Debug locally** with Ollama in Docker (F5 in VS Code)
3. **Test** with example_usage.py or Swagger UI
4. **Commit changes** to git
5. **Rebuild Docker image** when ready
6. **Test full stack** with docker-compose
7. **Export** Docker images and models
8. **Transfer** to RHEL9 (USB/network)
9. **Deploy** on RHEL9

## Quick Reference

### Windows Development Commands

```powershell
# Start Ollama only
docker-compose -f docker-compose.ollama.yml up -d

# Pull models
bash setup-ollama-models.sh

# Run API locally (for debugging)
python airgapped_rag.py

# Stop Ollama
docker-compose -f docker-compose.ollama.yml down
```

### Build & Test

```powershell
# Build API image
docker build -f Dockerfile.airgapped -t airgapped-rag-api:latest .

# Test full stack
docker-compose -f docker-compose.airgapped.yml up -d
docker-compose -f docker-compose.airgapped.yml logs -f
docker-compose -f docker-compose.airgapped.yml down
```

### Export for RHEL9

```powershell
# Save images
docker save ollama/ollama:latest -o ollama.tar
docker save airgapped-rag-api:latest -o airgapped-rag-api.tar

# Export models
docker run --rm -v airgapped_ollama_models:/models -v ${PWD}:/backup alpine tar czf /backup/ollama-models.tar.gz -C /models .
```

### RHEL9 Deployment

```bash
# Load images
docker load -i ollama.tar
docker load -i airgapped-rag-api.tar

# Restore models
docker volume create airgapped_ollama_models
docker run --rm -v airgapped_ollama_models:/models -v $(pwd):/backup alpine tar xzf /backup/ollama-models.tar.gz -C /models

# Deploy
docker-compose -f docker-compose.airgapped.yml up -d
```

## Security Considerations

### Windows Development
- ✅ Run Docker Desktop with WSL2 (more secure than Hyper-V)
- ✅ Use virtual environment for Python
- ✅ Don't commit `.env` files with sensitive data

### RHEL9 Production
- ✅ Keep SELinux enforcing (create proper policies)
- ✅ Use firewalld to limit access to ports
- ✅ Run containers as non-root (already configured in Dockerfile)
- ✅ Regularly update base images
- ✅ Use secrets management for any credentials

## Next Steps

1. ✅ Complete Windows development setup
2. ✅ Debug and test your API locally
3. ✅ Build Docker images
4. ✅ Export for RHEL9
5. ✅ Deploy on production RHEL9 system
6. 📝 Document any custom configurations for your environment
7. 📝 Create backup/restore procedures
8. 📝 Set up monitoring and logging

---

**Need Help?**
- Check the main [README_AIRGAPPED_RAG.md](README_AIRGAPPED_RAG.md)
- Review Docker logs: `docker logs <container-name>`
- Check Ollama API: `curl http://localhost:11434/api/tags`
