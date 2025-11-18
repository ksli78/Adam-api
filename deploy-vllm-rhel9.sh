#!/bin/bash
#
# vLLM Deployment Script for RHEL9
#
# This script deploys vLLM on RHEL9 with NVIDIA GPUs.
# Run this script ON THE RHEL9 SERVER (not locally).
#
# Prerequisites:
#   - NVIDIA drivers installed
#   - nvidia-docker/nvidia-container-toolkit installed
#   - Docker and docker-compose installed
#   - 2x NVIDIA Quadro RTX 5000 GPUs
#
# Usage:
#   sudo bash deploy-vllm-rhel9.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "vLLM Deployment for RHEL9"
echo "========================================${NC}"
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}This script requires root privileges.${NC}"
  echo "Please run with sudo:"
  echo "  sudo bash deploy-vllm-rhel9.sh"
  exit 1
fi

# Get the actual user (not root when using sudo)
ACTUAL_USER="${SUDO_USER:-$USER}"
echo "Running as: $ACTUAL_USER"
echo ""

#
# Step 1: Check prerequisites
#
echo -e "${BLUE}Step 1: Checking prerequisites...${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    echo "Install Docker with:"
    echo "  sudo dnf install -y docker"
    echo "  sudo systemctl enable --now docker"
    exit 1
fi
echo -e "${GREEN}✓ Docker is installed${NC}"

# Check if Docker is running
if ! systemctl is-active --quiet docker; then
    echo -e "${YELLOW}⚠ Docker is not running. Starting...${NC}"
    systemctl start docker
fi
echo -e "${GREEN}✓ Docker is running${NC}"

# Check for docker-compose (v1) or docker compose (v2 plugin)
COMPOSE_CMD=""
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
    echo -e "${GREEN}✓ docker compose (plugin) is available${NC}"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    echo -e "${GREEN}✓ docker-compose (standalone) is available${NC}"
else
    echo -e "${YELLOW}⚠ docker-compose not found. Installing standalone version...${NC}"
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    COMPOSE_CMD="docker-compose"
    echo -e "${GREEN}✓ docker-compose installed${NC}"
fi

# Check for NVIDIA drivers
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ NVIDIA drivers not found${NC}"
    echo "Install NVIDIA drivers first. See: GPU_SETUP_RHEL9_PORTAINER.md"
    exit 1
fi
echo -e "${GREEN}✓ NVIDIA drivers installed${NC}"

# Check GPUs
echo ""
echo "Detected GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo ""

if [ "$GPU_COUNT" -lt 2 ]; then
    echo -e "${YELLOW}⚠ Warning: Expected 2 GPUs, found $GPU_COUNT${NC}"
    echo "vLLM is configured for 2 GPUs (tensor-parallel-size=2)"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Found $GPU_COUNT GPUs${NC}"
fi

# Check for nvidia-container-toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-container-toolkit not working${NC}"
    echo "Install with:"
    echo "  sudo dnf install -y nvidia-container-toolkit"
    echo "  sudo systemctl restart docker"
    exit 1
fi
echo -e "${GREEN}✓ nvidia-container-toolkit is working${NC}"

# Check for required files
echo ""
echo "Checking for required files..."
if [ ! -f "docker-compose.vllm.yml" ]; then
    echo -e "${RED}✗ Missing: docker-compose.vllm.yml${NC}"
    echo "Are you in the correct directory?"
    exit 1
fi
echo -e "${GREEN}✓ Found: docker-compose.vllm.yml${NC}"

if [ ! -f "vllm_client.py" ]; then
    echo -e "${RED}✗ Missing: vllm_client.py${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found: vllm_client.py${NC}"

echo ""

#
# Step 2: Pull vLLM image
#
echo -e "${BLUE}Step 2: Pulling vLLM image...${NC}"
echo "This will download ~8GB from Docker Hub (first time only)"
echo ""

if docker pull vllm/vllm-openai:latest; then
    echo -e "${GREEN}✓ vLLM image pulled successfully${NC}"
else
    echo -e "${RED}✗ Failed to pull vLLM image${NC}"
    echo "Is the server connected to the internet?"
    exit 1
fi

echo ""

#
# Step 3: Start vLLM container
#
echo -e "${BLUE}Step 3: Starting vLLM container...${NC}"
echo "This will download Mistral-Small model (~24GB) on first run"
echo "Expected time: 10-30 minutes depending on internet speed"
echo ""

# Stop any existing vLLM container
if docker ps -a | grep -q vllm-server; then
    echo "Stopping existing vLLM container..."
    $COMPOSE_CMD -f docker-compose.vllm.yml down vllm
fi

# Start vLLM
$COMPOSE_CMD -f docker-compose.vllm.yml up -d vllm

echo -e "${GREEN}✓ vLLM container started${NC}"
echo ""

#
# Step 4: Monitor startup
#
echo -e "${BLUE}Step 4: Monitoring startup...${NC}"
echo "Waiting for vLLM to download model and initialize..."
echo "This may take 10-30 minutes on first run."
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop monitoring (vLLM will continue in background)${NC}"
echo ""

# Follow logs
docker logs -f vllm-server &
LOG_PID=$!

# Wait for health check to pass (with timeout)
TIMEOUT=1800  # 30 minutes
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if docker ps | grep vllm-server | grep -q "healthy"; then
        kill $LOG_PID 2>/dev/null || true
        echo ""
        echo -e "${GREEN}✓ vLLM is healthy and ready!${NC}"
        break
    fi

    sleep 10
    ELAPSED=$((ELAPSED + 10))

    # Check if container crashed
    if ! docker ps | grep -q vllm-server; then
        kill $LOG_PID 2>/dev/null || true
        echo ""
        echo -e "${RED}✗ vLLM container stopped unexpectedly${NC}"
        echo "Check logs with: docker logs vllm-server"
        exit 1
    fi
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    kill $LOG_PID 2>/dev/null || true
    echo ""
    echo -e "${YELLOW}⚠ Timeout waiting for vLLM to become healthy${NC}"
    echo "Container is still starting. Check logs with:"
    echo "  docker logs -f vllm-server"
fi

echo ""

#
# Step 5: Test vLLM
#
echo -e "${BLUE}Step 5: Testing vLLM...${NC}"

# Wait a bit for API to be fully ready
sleep 5

# Test health endpoint
if curl -s http://localhost:8001/health | grep -q "ok"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${YELLOW}⚠ Health check failed, but container is running${NC}"
    echo "Try manually: curl http://localhost:8001/health"
fi

# Test completion endpoint
echo ""
echo "Testing text generation..."
RESPONSE=$(curl -s http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-Small-Instruct-2409",
    "prompt": "2+2=",
    "max_tokens": 10,
    "temperature": 0.1
  }')

if echo "$RESPONSE" | grep -q "choices"; then
    echo -e "${GREEN}✓ Text generation working!${NC}"
    echo "Sample response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    echo -e "${YELLOW}⚠ Text generation test failed${NC}"
    echo "Response: $RESPONSE"
fi

echo ""

#
# Summary
#
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}vLLM Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "vLLM is running and accessible at:"
echo "  - Health: http://localhost:8001/health"
echo "  - API:    http://localhost:8001/v1/completions"
echo ""
echo "GPU usage:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""
echo "Container status:"
docker ps | grep vllm-server
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Review VLLM_MIGRATION_GUIDE.md for code changes"
echo "2. Update Python files to use vLLM client"
echo "3. Test with: curl http://localhost:8001/v1/completions"
echo ""
echo "To view logs:"
echo "  docker logs -f vllm-server"
echo ""
echo "To stop vLLM:"
echo "  docker-compose -f docker-compose.vllm.yml down vllm"
echo ""
