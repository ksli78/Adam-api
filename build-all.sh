#!/bin/bash
# Complete build script for Adam RAG System
# Builds Ollama, Adam API, creates networks, volumes, and pulls LLM model

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
echo -e "${BLUE}Adam RAG System - Complete Build Script${NC}"
echo "================================================================"
echo ""

# Step 1: Pull Ollama image
echo -e "${BLUE}[Step 1/7] Pulling Ollama image...${NC}"
docker pull $OLLAMA_IMAGE
echo -e "${GREEN}✓ Ollama image pulled${NC}"
echo ""

# Step 2: Build Adam API image
echo -e "${BLUE}[Step 2/7] Building Adam API image...${NC}"
docker build -t $ADAM_API_IMAGE .
echo -e "${GREEN}✓ Adam API image built${NC}"
echo ""

# Step 3: Create Docker network
echo -e "${BLUE}[Step 3/7] Creating Docker network...${NC}"
if docker network inspect $NETWORK_NAME >/dev/null 2>&1; then
    echo -e "${YELLOW}Network '$NETWORK_NAME' already exists, skipping...${NC}"
else
    docker network create $NETWORK_NAME
    echo -e "${GREEN}✓ Network '$NETWORK_NAME' created${NC}"
fi
echo ""

# Step 4: Create Docker volumes
echo -e "${BLUE}[Step 4/7] Creating Docker volumes...${NC}"
if docker volume inspect $ADAM_VOLUME >/dev/null 2>&1; then
    echo -e "${YELLOW}Volume '$ADAM_VOLUME' already exists, skipping...${NC}"
else
    docker volume create $ADAM_VOLUME
    echo -e "${GREEN}✓ Volume '$ADAM_VOLUME' created${NC}"
fi

if docker volume inspect $OLLAMA_VOLUME >/dev/null 2>&1; then
    echo -e "${YELLOW}Volume '$OLLAMA_VOLUME' already exists, skipping...${NC}"
else
    docker volume create $OLLAMA_VOLUME
    echo -e "${GREEN}✓ Volume '$OLLAMA_VOLUME' created${NC}"
fi
echo ""

# Step 5: Start Ollama container
echo -e "${BLUE}[Step 5/7] Starting Ollama container...${NC}"
if docker ps -a --format '{{.Names}}' | grep -q "^ollama$"; then
    echo -e "${YELLOW}Ollama container already exists${NC}"
    if docker ps --format '{{.Names}}' | grep -q "^ollama$"; then
        echo -e "${GREEN}Ollama is already running${NC}"
    else
        echo "Starting existing Ollama container..."
        docker start ollama
        echo -e "${GREEN}✓ Ollama container started${NC}"
    fi
else
    docker run -d \
        --name ollama \
        --network $NETWORK_NAME \
        -p 11434:11434 \
        -v $OLLAMA_VOLUME:/root/.ollama \
        --restart unless-stopped \
        $OLLAMA_IMAGE
    echo -e "${GREEN}✓ Ollama container started${NC}"
fi

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
sleep 5
for i in {1..30}; do
    if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Ollama failed to start${NC}"
        exit 1
    fi
    echo -n "."
    sleep 2
done
echo ""

# Step 6: Pull LLM model
echo -e "${BLUE}[Step 6/7] Pulling LLM model ($LLM_MODEL)...${NC}"
echo "This may take several minutes (model is ~4.5GB)..."
if docker exec ollama ollama list | grep -q "$LLM_MODEL"; then
    echo -e "${YELLOW}Model '$LLM_MODEL' already exists, skipping pull...${NC}"
else
    docker exec ollama ollama pull $LLM_MODEL
    echo -e "${GREEN}✓ LLM model pulled${NC}"
fi
echo ""

# Step 7: Start Adam API container
echo -e "${BLUE}[Step 7/7] Starting Adam API container...${NC}"
if docker ps -a --format '{{.Names}}' | grep -q "^adam-api$"; then
    echo -e "${YELLOW}Adam API container already exists${NC}"
    if docker ps --format '{{.Names}}' | grep -q "^adam-api$"; then
        echo "Stopping existing container..."
        docker stop adam-api
    fi
    echo "Removing old container..."
    docker rm adam-api
fi

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
echo ""

# Wait for API to be ready
echo "Waiting for Adam API to be ready..."
sleep 5
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Adam API is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Adam API failed to start${NC}"
        echo "Check logs with: docker logs adam-api"
        exit 1
    fi
    echo -n "."
    sleep 2
done
echo ""

# Summary
echo "================================================================"
echo -e "${GREEN}Build Complete!${NC}"
echo "================================================================"
echo ""
echo "Containers running:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "ollama|adam-api"
echo ""
echo "Resources created:"
echo "  Network: $NETWORK_NAME"
echo "  Volumes: $ADAM_VOLUME, $OLLAMA_VOLUME"
echo ""
echo "Access points:"
echo "  Adam API:          http://localhost:8000"
echo "  API Documentation: http://localhost:8000/docs"
echo "  Ollama:            http://localhost:11434"
echo ""
echo "Useful commands:"
echo "  View logs:         docker logs -f adam-api"
echo "  Stop services:     docker stop adam-api ollama"
echo "  Start services:    docker start ollama adam-api"
echo "  Export images:     ./export-images.sh"
echo ""
echo -e "${YELLOW}Next: Test the API with a query!${NC}"
echo "  curl -X POST http://localhost:8000/query \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"What is your name?\"}'"
echo ""
