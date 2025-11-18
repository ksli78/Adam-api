#!/bin/bash
# Export vLLM image for air-gapped deployment to RHEL9
# Run this on a machine WITH internet access

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
VLLM_IMAGE="vllm/vllm-openai:latest"
EXPORT_DIR="./vllm-export"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================"
echo -e "${BLUE}vLLM Image Export for Air-Gapped Deployment${NC}"
echo "================================================================"
echo ""

# Create export directory
echo -e "${BLUE}Creating export directory...${NC}"
mkdir -p $EXPORT_DIR
echo -e "${GREEN}✓ Directory created: $EXPORT_DIR${NC}"
echo ""

# Pull vLLM image
echo -e "${BLUE}Pulling vLLM image from Docker Hub...${NC}"
echo "Image: $VLLM_IMAGE (~8GB download)"
echo ""

if docker pull $VLLM_IMAGE; then
    echo -e "${GREEN}✓ Image pulled successfully${NC}"
else
    echo -e "${RED}✗ Failed to pull image${NC}"
    exit 1
fi

echo ""

# Get image size
IMAGE_SIZE=$(docker image inspect $VLLM_IMAGE --format='{{.Size}}' | awk '{print int($1/1024/1024)}')
echo "Image size: ${IMAGE_SIZE}MB (uncompressed)"
echo ""

# Export image
echo -e "${BLUE}Exporting vLLM image...${NC}"
VLLM_FILE="$EXPORT_DIR/vllm-openai.tar"
docker save $VLLM_IMAGE -o $VLLM_FILE
echo -e "${GREEN}✓ Image saved: ${VLLM_FILE}${NC}"
echo ""

# Compress
echo -e "${BLUE}Compressing...${NC}"
gzip -f $VLLM_FILE
echo -e "${GREEN}✓ Compressed: ${VLLM_FILE}.gz${NC}"

EXPORT_SIZE=$(ls -lh "${VLLM_FILE}.gz" | awk '{print $5}')
echo "  Compressed size: $EXPORT_SIZE"
echo ""

# Copy required files
echo -e "${BLUE}Copying required files...${NC}"
cp docker-compose.vllm.yml $EXPORT_DIR/
cp vllm_client.py $EXPORT_DIR/
cp VLLM_MIGRATION_GUIDE.md $EXPORT_DIR/
cp .env.vllm.example $EXPORT_DIR/
echo -e "${GREEN}✓ Files copied${NC}"
echo ""

# Create deployment instructions
cat > $EXPORT_DIR/DEPLOY_INSTRUCTIONS.txt << 'EOF'
vLLM Air-Gapped Deployment Instructions
========================================

Files in this package:
  - vllm-openai.tar.gz          vLLM Docker image
  - docker-compose.vllm.yml     Docker Compose configuration
  - vllm_client.py              Python client library
  - VLLM_MIGRATION_GUIDE.md     Complete migration guide
  - .env.vllm.example           Environment configuration

Transfer these files to your RHEL9 server.

On RHEL9 Server:
---------------

1. Load the vLLM image:
   gunzip vllm-openai.tar.gz
   sudo docker load -i vllm-openai.tar

2. Verify image loaded:
   sudo docker images | grep vllm

3. Start vLLM:
   sudo docker-compose -f docker-compose.vllm.yml up -d vllm

4. Monitor startup (first time downloads model ~24GB):
   sudo docker logs -f vllm-server

5. Test vLLM:
   curl http://localhost:8001/health
   curl http://localhost:8001/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "mistralai/Mistral-Small-Instruct-2409",
       "prompt": "What is 2+2?",
       "max_tokens": 50
     }'

6. Follow VLLM_MIGRATION_GUIDE.md for code changes

Notes:
------
- vLLM requires NVIDIA GPUs (2x Quadro RTX 5000 recommended)
- First startup downloads Mistral-Small model (~24GB) from HuggingFace
  * If server has NO internet, you must pre-download the model
  * See VLLM_MIGRATION_GUIDE.md "Issue 2: Model download fails"
- Requires nvidia-container-toolkit installed on RHEL9
- Allow 30GB disk space for model cache

Troubleshooting:
---------------
See VLLM_MIGRATION_GUIDE.md for detailed troubleshooting.

Common issues:
- "No GPU available" → Install nvidia-container-toolkit
- "CUDA out of memory" → Reduce --gpu-memory-utilization
- Model download fails → Pre-download model manually
EOF

echo -e "${GREEN}✓ Created DEPLOY_INSTRUCTIONS.txt${NC}"
echo ""

# Create package
echo -e "${BLUE}Creating deployment package...${NC}"
PACKAGE_NAME="vllm-deployment-${TIMESTAMP}.tar.gz"
tar -czf $PACKAGE_NAME -C $EXPORT_DIR .
PACKAGE_SIZE=$(ls -lh "$PACKAGE_NAME" | awk '{print $5}')
echo -e "${GREEN}✓ Package created: ${PACKAGE_NAME}${NC}"
echo "  Size: $PACKAGE_SIZE"
echo ""

# Summary
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}Export Complete!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "Transfer this file to your RHEL9 server:"
echo "  ${PACKAGE_NAME}"
echo ""
echo "On RHEL9, extract with:"
echo "  tar -xzf ${PACKAGE_NAME}"
echo "  cd vllm-deployment-${TIMESTAMP}"
echo "  cat DEPLOY_INSTRUCTIONS.txt"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC} vLLM will still need to download the model (~24GB)"
echo "from HuggingFace on first run. If your RHEL9 server has NO internet,"
echo "you must pre-download the model. See VLLM_MIGRATION_GUIDE.md"
echo ""
