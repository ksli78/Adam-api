#!/bin/bash
#
# Build Adam API with vLLM Support
# Run this on your LOCAL machine (no GPU needed)
#
# This script:
# 1. Builds adam-api Docker image with vLLM client
# 2. Exports both adam-api and vLLM images
# 3. Creates deployment package for RHEL9
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
VLLM_IMAGE="vllm/vllm-openai:latest"
ADAM_API_IMAGE="adam-api-vllm:latest"
EXPORT_DIR="./vllm-deployment-package"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================"
echo -e "${BLUE}Build Adam API with vLLM Support${NC}"
echo "================================================================"
echo ""
echo "This script prepares a complete deployment package for RHEL9:"
echo "  1. Builds adam-api Docker image (with vLLM client)"
echo "  2. Pulls vLLM server image from Docker Hub"
echo "  3. Exports both images for air-gapped deployment"
echo "  4. Packages everything for RHEL9 transfer"
echo ""

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}✗ Docker is not running${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"
echo ""

# Check if vllm_client.py exists
if [ ! -f "vllm_client.py" ]; then
    echo -e "${RED}✗ vllm_client.py not found${NC}"
    echo "Are you in the Adam-api directory?"
    exit 1
fi
echo -e "${GREEN}✓ Found vllm_client.py${NC}"
echo ""

#
# Step 1: Build Adam API image
#
echo -e "${BLUE}[Step 1/5] Building Adam API Docker image...${NC}"
echo "Building: $ADAM_API_IMAGE"
echo ""

# Build using standard Dockerfile (vllm_client.py will be included)
docker build -t $ADAM_API_IMAGE -f Dockerfile .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Adam API image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build Adam API image${NC}"
    exit 1
fi

# Get image size
ADAM_SIZE=$(docker image inspect $ADAM_API_IMAGE --format='{{.Size}}' | awk '{print int($1/1024/1024)}')
echo "  Image size: ${ADAM_SIZE}MB"
echo ""

#
# Step 2: Pull vLLM image
#
echo -e "${BLUE}[Step 2/5] Pulling vLLM image from Docker Hub...${NC}"
echo "Pulling: $VLLM_IMAGE (~8GB download)"
echo ""

docker pull $VLLM_IMAGE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ vLLM image pulled successfully${NC}"
else
    echo -e "${RED}✗ Failed to pull vLLM image${NC}"
    echo "Check internet connection"
    exit 1
fi

# Get vLLM image size
VLLM_SIZE=$(docker image inspect $VLLM_IMAGE --format='{{.Size}}' | awk '{print int($1/1024/1024)}')
echo "  Image size: ${VLLM_SIZE}MB"
echo ""

TOTAL_SIZE=$((ADAM_SIZE + VLLM_SIZE))
echo "Total uncompressed: ${TOTAL_SIZE}MB"
echo ""

#
# Step 3: Create export directory
#
echo -e "${BLUE}[Step 3/5] Preparing export directory...${NC}"
rm -rf $EXPORT_DIR
mkdir -p $EXPORT_DIR
echo -e "${GREEN}✓ Created: $EXPORT_DIR${NC}"
echo ""

#
# Step 4: Export images
#
echo -e "${BLUE}[Step 4/5] Exporting Docker images...${NC}"
echo "This may take several minutes..."
echo ""

# Export Adam API
echo "Exporting Adam API image..."
ADAM_FILE="$EXPORT_DIR/adam-api-vllm.tar"
docker save $ADAM_API_IMAGE -o $ADAM_FILE
echo "Compressing..."
gzip -f $ADAM_FILE
ADAM_EXPORT_SIZE=$(ls -lh "${ADAM_FILE}.gz" | awk '{print $5}')
echo -e "${GREEN}✓ Adam API exported: ${ADAM_FILE}.gz (${ADAM_EXPORT_SIZE})${NC}"
echo ""

# Export vLLM
echo "Exporting vLLM image..."
VLLM_FILE="$EXPORT_DIR/vllm-openai.tar"
docker save $VLLM_IMAGE -o $VLLM_FILE
echo "Compressing..."
gzip -f $VLLM_FILE
VLLM_EXPORT_SIZE=$(ls -lh "${VLLM_FILE}.gz" | awk '{print $5}')
echo -e "${GREEN}✓ vLLM exported: ${VLLM_FILE}.gz (${VLLM_EXPORT_SIZE})${NC}"
echo ""

#
# Step 5: Copy required files
#
echo -e "${BLUE}[Step 5/5] Packaging deployment files...${NC}"

# Copy configuration files
cp docker-compose.vllm.yml $EXPORT_DIR/
cp vllm_client.py $EXPORT_DIR/
cp .env.vllm.example $EXPORT_DIR/.env.example
cp VLLM_MIGRATION_GUIDE.md $EXPORT_DIR/ 2>/dev/null || true

# Copy SQL config if exists
if [ -d "config" ]; then
    cp -r config $EXPORT_DIR/
    echo "✓ Copied config directory"
fi

echo -e "${GREEN}✓ Configuration files copied${NC}"
echo ""

# Create deployment script for RHEL9
cat > $EXPORT_DIR/deploy-on-rhel9.sh << 'DEPLOY_SCRIPT'
#!/bin/bash
#
# Deploy Adam API with vLLM on RHEL9
# Run this script ON THE RHEL9 SERVER
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "Deploy Adam API with vLLM on RHEL9"
echo "========================================${NC}"
echo ""

# Check prerequisites
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}This script requires sudo privileges.${NC}"
  echo "Run: sudo bash deploy-on-rhel9.sh"
  exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

# Check NVIDIA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ NVIDIA drivers not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ NVIDIA drivers installed${NC}"

# Check GPU
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${GREEN}✓ Found ${GPU_COUNT} GPU(s)${NC}"
echo ""

# Load images
echo -e "${BLUE}Loading Docker images...${NC}"
echo ""

if [ -f "vllm-openai.tar.gz" ]; then
    echo "Loading vLLM image..."
    gunzip -c vllm-openai.tar.gz | docker load
    echo -e "${GREEN}✓ vLLM image loaded${NC}"
else
    echo -e "${YELLOW}⚠ vllm-openai.tar.gz not found, will pull from Docker Hub${NC}"
    docker pull vllm/vllm-openai:latest
fi

if [ -f "adam-api-vllm.tar.gz" ]; then
    echo "Loading Adam API image..."
    gunzip -c adam-api-vllm.tar.gz | docker load
    echo -e "${GREEN}✓ Adam API image loaded${NC}"
else
    echo -e "${RED}✗ adam-api-vllm.tar.gz not found${NC}"
    exit 1
fi

echo ""

# Check for docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Installing docker-compose...${NC}"
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Setup environment
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${YELLOW}⚠ Please edit .env and set your configuration${NC}"
fi

# Start services
echo -e "${BLUE}Starting vLLM and Adam API...${NC}"
echo ""

# Start vLLM first
docker-compose -f docker-compose.vllm.yml up -d vllm

echo "Waiting for vLLM to initialize (may take 10-30 minutes on first run)..."
echo "vLLM will download Mistral-Small model (~24GB) from HuggingFace"
echo ""
echo "Monitor with: docker logs -f vllm-server"
echo ""

# Wait a bit
sleep 10

# Start Adam API
docker-compose -f docker-compose.vllm.yml up -d adam-api

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Services are starting. Monitor with:"
echo "  docker-compose -f docker-compose.vllm.yml logs -f"
echo ""
echo "Check status with:"
echo "  docker-compose -f docker-compose.vllm.yml ps"
echo ""
echo "Test vLLM health:"
echo "  curl http://localhost:8001/health"
echo ""
echo "Test Adam API:"
echo "  curl http://localhost:8000/health"
echo ""
DEPLOY_SCRIPT

chmod +x $EXPORT_DIR/deploy-on-rhel9.sh
echo -e "${GREEN}✓ Created deployment script: deploy-on-rhel9.sh${NC}"
echo ""

# Create README
cat > $EXPORT_DIR/README.txt << 'README'
Adam API with vLLM - RHEL9 Deployment Package
==============================================

This package contains everything needed to deploy Adam API with vLLM
on your RHEL9 server with NVIDIA GPUs.

Package Contents:
-----------------
  adam-api-vllm.tar.gz       - Adam API Docker image (with vLLM client)
  vllm-openai.tar.gz         - vLLM inference server image
  docker-compose.vllm.yml    - Docker Compose configuration
  deploy-on-rhel9.sh         - Automated deployment script
  vllm_client.py             - vLLM Python client (included in image)
  .env.example               - Environment configuration template
  config/                    - Configuration files (if included)

Quick Start:
-----------

1. Transfer this entire directory to your RHEL9 server:
   scp -r vllm-deployment-package/ user@rhel9-server:/tmp/

2. SSH to RHEL9 server:
   ssh user@rhel9-server

3. Run deployment script:
   cd /tmp/vllm-deployment-package
   sudo bash deploy-on-rhel9.sh

4. Monitor startup:
   docker logs -f vllm-server    # vLLM (downloads model first time)
   docker logs -f adam-api       # Adam API

5. Test:
   curl http://localhost:8001/health    # vLLM health
   curl http://localhost:8000/health    # Adam API health

Expected Timeline:
-----------------
  - Image loading: 2-5 minutes
  - vLLM model download (first time): 10-30 minutes
  - Total first deployment: 15-40 minutes
  - Subsequent startups: 1-2 minutes

Requirements:
------------
  - RHEL9 with NVIDIA GPUs (2x Quadro RTX 5000 recommended)
  - NVIDIA drivers installed
  - nvidia-container-toolkit installed
  - Docker and docker-compose
  - 50GB free disk space (for model cache)

Performance:
-----------
  Expected improvements vs Ollama:
    - RAG queries: 45s → 13-18s (2.5-3.5x faster)
    - Better concurrent request handling (3-5 users)
    - More consistent latency

Troubleshooting:
---------------
  - "No GPU available": Install nvidia-container-toolkit
  - "Model download fails": Check internet or pre-download model
  - "Out of memory": Reduce --gpu-memory-utilization in docker-compose.vllm.yml

For detailed troubleshooting, see VLLM_MIGRATION_GUIDE.md

Support:
--------
See VLLM_MIGRATION_GUIDE.md for complete documentation.
README

echo -e "${GREEN}✓ Created README.txt${NC}"
echo ""

# Create final tarball
echo "Creating final deployment package..."
PACKAGE_NAME="adam-vllm-deployment-${TIMESTAMP}.tar.gz"
tar -czf $PACKAGE_NAME -C $EXPORT_DIR .

if [ $? -eq 0 ]; then
    PACKAGE_SIZE=$(ls -lh "$PACKAGE_NAME" | awk '{print $5}')
    echo -e "${GREEN}✓ Package created: ${PACKAGE_NAME} (${PACKAGE_SIZE})${NC}"
else
    echo -e "${RED}✗ Failed to create package${NC}"
    exit 1
fi

echo ""

# Summary
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "Deployment package created:"
echo "  ${PACKAGE_NAME}"
echo "  Size: ${PACKAGE_SIZE}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo ""
echo "1. Transfer to RHEL9 server:"
echo "   scp ${PACKAGE_NAME} user@rhel9-server:/tmp/"
echo ""
echo "2. On RHEL9 server:"
echo "   cd /tmp"
echo "   tar -xzf ${PACKAGE_NAME}"
echo "   sudo bash deploy-on-rhel9.sh"
echo ""
echo "3. Monitor deployment:"
echo "   docker logs -f vllm-server"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC}"
echo "  - First deployment will download Mistral-Small model (~24GB)"
echo "  - This requires internet on RHEL9 server"
echo "  - Allow 10-30 minutes for first startup"
echo ""
echo "See README.txt in the package for complete instructions."
echo ""
