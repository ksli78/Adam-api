#!/bin/bash
# Production Deployment Script for RHEL9
# Imports Docker images, creates network/volumes, and starts Adam RAG System

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_IMAGE="ollama/ollama:latest"
ADAM_API_IMAGE="adam-api:latest"
NETWORK_NAME="adam-network"
ADAM_VOLUME="adam-data"
OLLAMA_VOLUME="ollama-data"
LLM_MODEL="llama3:8b"

echo "================================================================"
echo -e "${BLUE}Adam RAG System - Production Deployment for RHEL9${NC}"
echo "================================================================"
echo ""

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]]; then
   DOCKER_CMD="docker"
else
   echo -e "${YELLOW}Note: Running with current user. May need sudo for Docker commands.${NC}"
   DOCKER_CMD="docker"
fi

# Verify Docker is installed and running
echo -e "${BLUE}Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    echo "Install Docker first:"
    echo "  sudo dnf config-manager --add-repo=https://download.docker.com/linux/rhel/docker-ce.repo"
    echo "  sudo dnf install docker-ce docker-ce-cli containerd.io"
    echo "  sudo systemctl start docker"
    echo "  sudo systemctl enable docker"
    exit 1
fi

if ! sudo systemctl is-active --quiet docker; then
    echo -e "${YELLOW}Docker is not running, starting...${NC}"
    sudo systemctl start docker
    sleep 3
fi
echo -e "${GREEN}✓ Docker is running${NC}"
echo ""

# Find image files
echo -e "${BLUE}Locating image files...${NC}"
echo "Please enter the directory containing the .tar.gz files"
echo "(or press Enter to use current directory):"
read -r IMAGE_DIR

if [ -z "$IMAGE_DIR" ]; then
    IMAGE_DIR="."
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo -e "${RED}✗ Directory not found: $IMAGE_DIR${NC}"
    exit 1
fi

# Look for the files
OLLAMA_FILE=$(find "$IMAGE_DIR" -name "ollama-image.tar.gz" -o -name "ollama*.tar.gz" | head -1)
ADAM_FILE=$(find "$IMAGE_DIR" -name "adam-api-image.tar.gz" -o -name "adam*.tar.gz" | head -1)

if [ -z "$OLLAMA_FILE" ]; then
    echo -e "${RED}✗ Ollama image file not found in $IMAGE_DIR${NC}"
    echo "Looking for: ollama-image.tar.gz or ollama*.tar.gz"
    exit 1
fi

if [ -z "$ADAM_FILE" ]; then
    echo -e "${RED}✗ Adam API image file not found in $IMAGE_DIR${NC}"
    echo "Looking for: adam-api-image.tar.gz or adam*.tar.gz"
    exit 1
fi

echo -e "${GREEN}✓ Found images:${NC}"
echo "  Ollama: $OLLAMA_FILE"
echo "  Adam:   $ADAM_FILE"
echo ""

# Step 1: Import Ollama image
echo "================================================================"
echo -e "${BLUE}[Step 1/6] Importing Ollama image...${NC}"
echo "================================================================"
echo "This may take several minutes..."
echo ""

if docker image inspect $OLLAMA_IMAGE >/dev/null 2>&1; then
    echo -e "${YELLOW}Ollama image already exists. Remove it first? (y/N)${NC}"
    read -r REMOVE_OLLAMA
    if [[ $REMOVE_OLLAMA =~ ^[Yy]$ ]]; then
        docker rmi $OLLAMA_IMAGE
        gunzip -c "$OLLAMA_FILE" | docker load
    else
        echo "Keeping existing image..."
    fi
else
    gunzip -c "$OLLAMA_FILE" | docker load
fi

echo -e "${GREEN}✓ Ollama image imported${NC}"
echo ""

# Step 2: Import Adam API image
echo "================================================================"
echo -e "${BLUE}[Step 2/6] Importing Adam API image...${NC}"
echo "================================================================"
echo "This may take several minutes..."
echo ""

if docker image inspect $ADAM_API_IMAGE >/dev/null 2>&1; then
    echo -e "${YELLOW}Adam API image already exists. Remove it first? (y/N)${NC}"
    read -r REMOVE_ADAM
    if [[ $REMOVE_ADAM =~ ^[Yy]$ ]]; then
        docker rmi $ADAM_API_IMAGE
        gunzip -c "$ADAM_FILE" | docker load
    else
        echo "Keeping existing image..."
    fi
else
    gunzip -c "$ADAM_FILE" | docker load
fi

echo -e "${GREEN}✓ Adam API image imported${NC}"
echo ""

# Verify images
echo "Imported images:"
docker images | grep -E "REPOSITORY|ollama|adam-api"
echo ""

# Step 3: Create Docker network
echo "================================================================"
echo -e "${BLUE}[Step 3/6] Creating Docker network...${NC}"
echo "================================================================"

if docker network inspect $NETWORK_NAME >/dev/null 2>&1; then
    echo -e "${YELLOW}Network '$NETWORK_NAME' already exists${NC}"
else
    docker network create $NETWORK_NAME
    echo -e "${GREEN}✓ Network '$NETWORK_NAME' created${NC}"
fi
echo ""

# Step 4: Create Docker volumes
echo "================================================================"
echo -e "${BLUE}[Step 4/6] Creating Docker volumes...${NC}"
echo "================================================================"

if docker volume inspect $ADAM_VOLUME >/dev/null 2>&1; then
    echo -e "${YELLOW}Volume '$ADAM_VOLUME' already exists${NC}"
else
    docker volume create $ADAM_VOLUME
    echo -e "${GREEN}✓ Volume '$ADAM_VOLUME' created${NC}"
fi

if docker volume inspect $OLLAMA_VOLUME >/dev/null 2>&1; then
    echo -e "${YELLOW}Volume '$OLLAMA_VOLUME' already exists${NC}"
else
    docker volume create $OLLAMA_VOLUME
    echo -e "${GREEN}✓ Volume '$OLLAMA_VOLUME' created${NC}"
fi
echo ""

# Step 5: Start Ollama container
echo "================================================================"
echo -e "${BLUE}[Step 5/6] Starting Ollama container...${NC}"
echo "================================================================"

# Stop and remove if exists
if docker ps -a --format '{{.Names}}' | grep -q "^ollama$"; then
    echo "Stopping existing Ollama container..."
    docker stop ollama 2>/dev/null || true
    echo "Removing existing Ollama container..."
    docker rm ollama 2>/dev/null || true
fi

# Start Ollama
docker run -d \
    --name ollama \
    --network $NETWORK_NAME \
    -p 11434:11434 \
    -v $OLLAMA_VOLUME:/root/.ollama \
    --restart unless-stopped \
    $OLLAMA_IMAGE

echo -e "${GREEN}✓ Ollama container started${NC}"

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
sleep 5
OLLAMA_READY=0
for i in {1..30}; do
    if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is ready${NC}"
        OLLAMA_READY=1
        break
    fi
    echo -n "."
    sleep 2
done

if [ $OLLAMA_READY -eq 0 ]; then
    echo -e "${RED}✗ Ollama failed to start${NC}"
    echo "Check logs with: docker logs ollama"
    exit 1
fi
echo ""

# Check if model exists, if not pull it
echo "Checking for LLM model ($LLM_MODEL)..."
if docker exec ollama ollama list | grep -q "$LLM_MODEL"; then
    echo -e "${GREEN}✓ Model '$LLM_MODEL' already exists${NC}"
else
    echo -e "${YELLOW}Model not found. Pulling $LLM_MODEL (~4.5GB)...${NC}"
    echo "This will take several minutes..."
    docker exec ollama ollama pull $LLM_MODEL
    echo -e "${GREEN}✓ Model pulled${NC}"
fi
echo ""

# Step 6: Start Adam API container
echo "================================================================"
echo -e "${BLUE}[Step 6/6] Starting Adam API container...${NC}"
echo "================================================================"

# Stop and remove if exists
if docker ps -a --format '{{.Names}}' | grep -q "^adam-api$"; then
    echo "Stopping existing Adam API container..."
    docker stop adam-api 2>/dev/null || true
    echo "Removing existing Adam API container..."
    docker rm adam-api 2>/dev/null || true
fi

# Start Adam API
docker run -d \
    --name adam-api \
    --network $NETWORK_NAME \
    -p 8000:8000 \
    -e OLLAMA_HOST=http://ollama:11434 \
    -e LLM_MODEL=$LLM_MODEL \
    -e DATA_DIR=/data/airgapped_rag \
    -v $ADAM_VOLUME:/data/airgapped_rag \
    --restart unless-stopped \
    $ADAM_API_IMAGE

echo -e "${GREEN}✓ Adam API container started${NC}"

# Wait for API to be ready
echo "Waiting for Adam API to be ready..."
sleep 5
API_READY=0
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Adam API is ready${NC}"
        API_READY=1
        break
    fi
    echo -n "."
    sleep 2
done

if [ $API_READY -eq 0 ]; then
    echo -e "${RED}✗ Adam API failed to start${NC}"
    echo "Check logs with: docker logs adam-api"
    exit 1
fi
echo ""

# Configure firewall for RHEL9
echo "================================================================"
echo -e "${BLUE}Configuring firewall...${NC}"
echo "================================================================"

if command -v firewall-cmd &> /dev/null; then
    echo "Opening ports in firewall..."
    sudo firewall-cmd --permanent --add-port=8000/tcp 2>/dev/null || echo "Port 8000 already open"
    sudo firewall-cmd --permanent --add-port=11434/tcp 2>/dev/null || echo "Port 11434 already open"
    sudo firewall-cmd --reload 2>/dev/null
    echo -e "${GREEN}✓ Firewall configured${NC}"
else
    echo -e "${YELLOW}firewalld not found, skipping firewall configuration${NC}"
fi
echo ""

# Enable Docker to start on boot
echo -e "${BLUE}Enabling Docker to start on boot...${NC}"
sudo systemctl enable docker
echo -e "${GREEN}✓ Docker enabled on boot${NC}"
echo ""

# Summary
echo "================================================================"
echo -e "${GREEN}Deployment Complete!${NC}"
echo "================================================================"
echo ""
echo "Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "Docker resources:"
echo "  Network: $NETWORK_NAME"
docker network inspect $NETWORK_NAME --format "    Subnet: {{range .IPAM.Config}}{{.Subnet}}{{end}}"
echo ""
echo "  Volumes:"
docker volume ls | grep -E "adam|ollama"
echo ""
echo "Access Points:"
echo "  Adam API:          http://$(hostname -I | awk '{print $1}'):8000"
echo "  API Documentation: http://$(hostname -I | awk '{print $1}'):8000/docs"
echo "  Health Check:      http://$(hostname -I | awk '{print $1}'):8000/health"
echo "  Ollama:            http://$(hostname -I | awk '{print $1}'):11434"
echo ""
echo "Local access:"
echo "  API:               http://localhost:8000"
echo "  API Docs:          http://localhost:8000/docs"
echo ""
echo "Management Commands:"
echo "  View API logs:     docker logs -f adam-api"
echo "  View Ollama logs:  docker logs ollama"
echo "  Restart API:       docker restart adam-api"
echo "  Stop all:          docker stop adam-api ollama"
echo "  Start all:         docker start ollama adam-api"
echo ""
echo "Testing:"
echo "  curl http://localhost:8000/health"
echo ""
echo "  curl -X POST http://localhost:8000/query \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"What is your name?\"}'"
echo ""

# Test the deployment
echo -e "${BLUE}Running health check...${NC}"
HEALTH_CHECK=$(curl -s http://localhost:8000/health)
if [[ $HEALTH_CHECK == *"healthy"* ]]; then
    echo -e "${GREEN}✓ System is healthy!${NC}"
    echo "$HEALTH_CHECK"
else
    echo -e "${YELLOW}⚠ Health check returned unexpected response${NC}"
    echo "$HEALTH_CHECK"
fi
echo ""

echo "================================================================"
echo -e "${GREEN}Adam RAG System is ready for use!${NC}"
echo "================================================================"
