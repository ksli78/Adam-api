#!/bin/bash
# Export Docker images for air-gapped deployment
# Creates .tar.gz files of Ollama and Adam API images

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
EXPORT_DIR="./docker-export"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="adam-deployment-${TIMESTAMP}"

echo "================================================================"
echo -e "${BLUE}Adam RAG System - Image Export Script${NC}"
echo "================================================================"
echo ""

# Create export directory
echo -e "${BLUE}Creating export directory...${NC}"
mkdir -p $EXPORT_DIR
echo -e "${GREEN}✓ Directory created: $EXPORT_DIR${NC}"
echo ""

# Verify images exist
echo -e "${BLUE}Verifying images exist...${NC}"
if ! docker image inspect $OLLAMA_IMAGE >/dev/null 2>&1; then
    echo -e "${RED}✗ Error: Ollama image not found${NC}"
    echo "Run './build-all.sh' first to build the images"
    exit 1
fi

if ! docker image inspect $ADAM_API_IMAGE >/dev/null 2>&1; then
    echo -e "${RED}✗ Error: Adam API image not found${NC}"
    echo "Run './build-all.sh' first to build the images"
    exit 1
fi
echo -e "${GREEN}✓ Both images found${NC}"
echo ""

# Get image sizes
OLLAMA_SIZE=$(docker image inspect $OLLAMA_IMAGE --format='{{.Size}}' | awk '{print int($1/1024/1024)}')
ADAM_SIZE=$(docker image inspect $ADAM_API_IMAGE --format='{{.Size}}' | awk '{print int($1/1024/1024)}')
TOTAL_SIZE=$((OLLAMA_SIZE + ADAM_SIZE))

echo "Image sizes:"
echo "  Ollama:    ${OLLAMA_SIZE}MB"
echo "  Adam API:  ${ADAM_SIZE}MB"
echo "  Total:     ${TOTAL_SIZE}MB (uncompressed)"
echo ""
echo -e "${YELLOW}Export will take several minutes...${NC}"
echo ""

# Export Ollama image
echo -e "${BLUE}[1/2] Exporting Ollama image...${NC}"
OLLAMA_FILE="$EXPORT_DIR/ollama-image.tar"
docker save $OLLAMA_IMAGE -o $OLLAMA_FILE
echo "Compressing..."
gzip -f $OLLAMA_FILE
echo -e "${GREEN}✓ Ollama image exported: ${OLLAMA_FILE}.gz${NC}"
OLLAMA_EXPORT_SIZE=$(ls -lh "${OLLAMA_FILE}.gz" | awk '{print $5}')
echo "  Compressed size: $OLLAMA_EXPORT_SIZE"
echo ""

# Export Adam API image
echo -e "${BLUE}[2/2] Exporting Adam API image...${NC}"
ADAM_FILE="$EXPORT_DIR/adam-api-image.tar"
docker save $ADAM_API_IMAGE -o $ADAM_FILE
echo "Compressing..."
gzip -f $ADAM_FILE
echo -e "${GREEN}✓ Adam API image exported: ${ADAM_FILE}.gz${NC}"
ADAM_EXPORT_SIZE=$(ls -lh "${ADAM_FILE}.gz" | awk '{print $5}')
echo "  Compressed size: $ADAM_EXPORT_SIZE"
echo ""

# Create deployment package
echo -e "${BLUE}Creating deployment package...${NC}"
PACKAGE_DIR="$EXPORT_DIR/$PACKAGE_NAME"
mkdir -p $PACKAGE_DIR

# Copy images
cp "${OLLAMA_FILE}.gz" "$PACKAGE_DIR/"
cp "${ADAM_FILE}.gz" "$PACKAGE_DIR/"

# Copy deployment scripts
cat > "$PACKAGE_DIR/import-images.sh" << 'EOF'
#!/bin/bash
# Import Docker images and deploy Adam RAG System
# Run this script on your production/air-gapped environment

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================"
echo -e "${BLUE}Adam RAG System - Image Import & Deployment${NC}"
echo "================================================================"
echo ""

# Configuration
NETWORK_NAME="adam-network"
ADAM_VOLUME="adam-data"
OLLAMA_VOLUME="ollama-data"
LLM_MODEL="llama3:8b"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    exit 1
fi

# Import Ollama image
echo -e "${BLUE}[1/4] Importing Ollama image...${NC}"
if [ -f "ollama-image.tar.gz" ]; then
    echo "Decompressing..."
    gunzip -c ollama-image.tar.gz | docker load
    echo -e "${GREEN}✓ Ollama image imported${NC}"
else
    echo -e "${RED}✗ ollama-image.tar.gz not found${NC}"
    exit 1
fi
echo ""

# Import Adam API image
echo -e "${BLUE}[2/4] Importing Adam API image...${NC}"
if [ -f "adam-api-image.tar.gz" ]; then
    echo "Decompressing..."
    gunzip -c adam-api-image.tar.gz | docker load
    echo -e "${GREEN}✓ Adam API image imported${NC}"
else
    echo -e "${RED}✗ adam-api-image.tar.gz not found${NC}"
    exit 1
fi
echo ""

# Create network
echo -e "${BLUE}[3/4] Creating Docker network...${NC}"
if docker network inspect $NETWORK_NAME >/dev/null 2>&1; then
    echo -e "${YELLOW}Network already exists, skipping...${NC}"
else
    docker network create $NETWORK_NAME
    echo -e "${GREEN}✓ Network created${NC}"
fi
echo ""

# Create volumes
echo -e "${BLUE}[4/4] Creating Docker volumes...${NC}"
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

echo "================================================================"
echo -e "${GREEN}Import Complete!${NC}"
echo "================================================================"
echo ""
echo "Images imported:"
docker images | grep -E "ollama|adam-api"
echo ""
echo "To start the services, run:"
echo "  ./start-services.sh"
echo ""
EOF

# Create start script
cat > "$PACKAGE_DIR/start-services.sh" << 'EOF'
#!/bin/bash
# Start Adam RAG System services

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
OLLAMA_IMAGE="ollama/ollama:latest"
ADAM_API_IMAGE="adam-api:latest"
NETWORK_NAME="adam-network"
ADAM_VOLUME="adam-data"
OLLAMA_VOLUME="ollama-data"
LLM_MODEL="llama3:8b"

echo "================================================================"
echo -e "${BLUE}Starting Adam RAG System${NC}"
echo "================================================================"
echo ""

# Start Ollama
echo -e "${BLUE}[1/3] Starting Ollama container...${NC}"
if docker ps -a --format '{{.Names}}' | grep -q "^ollama$"; then
    if docker ps --format '{{.Names}}' | grep -q "^ollama$"; then
        echo -e "${GREEN}Ollama is already running${NC}"
    else
        docker start ollama
        echo -e "${GREEN}✓ Ollama started${NC}"
    fi
else
    docker run -d \
        --name ollama \
        --network $NETWORK_NAME \
        -p 11434:11434 \
        -v $OLLAMA_VOLUME:/root/.ollama \
        --restart unless-stopped \
        $OLLAMA_IMAGE
    echo -e "${GREEN}✓ Ollama container created and started${NC}"
fi

# Wait for Ollama
echo "Waiting for Ollama to be ready..."
sleep 5
for i in {1..30}; do
    if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is ready${NC}"
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

# Pull LLM model if needed
echo -e "${BLUE}[2/3] Checking LLM model...${NC}"
if docker exec ollama ollama list | grep -q "$LLM_MODEL"; then
    echo -e "${GREEN}✓ Model '$LLM_MODEL' already available${NC}"
else
    echo "Pulling model (this will take several minutes)..."
    docker exec ollama ollama pull $LLM_MODEL
    echo -e "${GREEN}✓ Model pulled${NC}"
fi
echo ""

# Start Adam API
echo -e "${BLUE}[3/3] Starting Adam API container...${NC}"
if docker ps -a --format '{{.Names}}' | grep -q "^adam-api$"; then
    if docker ps --format '{{.Names}}' | grep -q "^adam-api$"; then
        echo "Restarting Adam API..."
        docker restart adam-api
        echo -e "${GREEN}✓ Adam API restarted${NC}"
    else
        docker start adam-api
        echo -e "${GREEN}✓ Adam API started${NC}"
    fi
else
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
    echo -e "${GREEN}✓ Adam API container created and started${NC}"
fi

# Wait for API
echo "Waiting for Adam API to be ready..."
sleep 5
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Adam API is ready${NC}"
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

echo "================================================================"
echo -e "${GREEN}System Started Successfully!${NC}"
echo "================================================================"
echo ""
echo "Services:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAME|ollama|adam-api"
echo ""
echo "Access:"
echo "  API:          http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo "  Ollama:       http://localhost:11434"
echo ""
echo "Test the system:"
echo "  curl http://localhost:8000/health"
echo ""
EOF

# Create stop script
cat > "$PACKAGE_DIR/stop-services.sh" << 'EOF'
#!/bin/bash
# Stop Adam RAG System services

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Stopping Adam RAG System...${NC}"
docker stop adam-api ollama 2>/dev/null || true
echo -e "${GREEN}✓ Services stopped${NC}"
EOF

# Create README
cat > "$PACKAGE_DIR/README.txt" << EOF
Adam RAG System - Deployment Package
=====================================

This package contains everything needed to deploy the Adam RAG system
in an air-gapped environment.

Contents:
---------
  ollama-image.tar.gz    - Ollama LLM service image
  adam-api-image.tar.gz  - Adam RAG API image
  import-images.sh       - Import Docker images
  start-services.sh      - Start all services
  stop-services.sh       - Stop all services
  README.txt             - This file

Deployment Instructions:
------------------------

1. Transfer this entire directory to your production server

2. Make scripts executable:
   chmod +x *.sh

3. Import the Docker images:
   ./import-images.sh

4. Start the services:
   ./start-services.sh

5. Verify deployment:
   curl http://localhost:8000/health

6. Access the API:
   - API Endpoint:    http://localhost:8000
   - Documentation:   http://localhost:8000/docs
   - Health Check:    http://localhost:8000/health

Management:
-----------
  Start services:    ./start-services.sh
  Stop services:     ./stop-services.sh
  View logs:         docker logs -f adam-api
  Restart:           docker restart adam-api

System Requirements:
--------------------
  - Docker 20.10+
  - 8GB+ RAM
  - 10GB+ disk space
  - Linux or Windows with WSL2

Troubleshooting:
----------------
  If services fail to start:
  1. Check Docker is running: docker ps
  2. Check logs: docker logs adam-api
  3. Verify ports are free: netstat -an | grep 8000

For more information, see the main repository documentation.

Package created: $(date)
EOF

# Make scripts executable
chmod +x "$PACKAGE_DIR"/*.sh

# Create tarball
echo -e "${BLUE}Creating deployment tarball...${NC}"
cd $EXPORT_DIR
tar czf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME"
cd - > /dev/null

PACKAGE_SIZE=$(ls -lh "$EXPORT_DIR/${PACKAGE_NAME}.tar.gz" | awk '{print $5}')
echo -e "${GREEN}✓ Deployment package created${NC}"
echo ""

# Calculate checksums
echo -e "${BLUE}Generating checksums...${NC}"
cd $EXPORT_DIR
sha256sum "${PACKAGE_NAME}.tar.gz" > "${PACKAGE_NAME}.sha256"
sha256sum ollama-image.tar.gz >> "${PACKAGE_NAME}.sha256"
sha256sum adam-api-image.tar.gz >> "${PACKAGE_NAME}.sha256"
cd - > /dev/null
echo -e "${GREEN}✓ Checksums saved to ${PACKAGE_NAME}.sha256${NC}"
echo ""

# Summary
echo "================================================================"
echo -e "${GREEN}Export Complete!${NC}"
echo "================================================================"
echo ""
echo "Files created in $EXPORT_DIR:"
echo "  1. ollama-image.tar.gz       ($OLLAMA_EXPORT_SIZE)"
echo "  2. adam-api-image.tar.gz     ($ADAM_EXPORT_SIZE)"
echo "  3. ${PACKAGE_NAME}.tar.gz    ($PACKAGE_SIZE) - COMPLETE PACKAGE"
echo "  4. ${PACKAGE_NAME}.sha256    - Checksums"
echo ""
echo -e "${YELLOW}For Production Deployment:${NC}"
echo ""
echo "Transfer the complete package to your production server:"
echo "  scp $EXPORT_DIR/${PACKAGE_NAME}.tar.gz user@production:/path/"
echo ""
echo "On production server:"
echo "  tar xzf ${PACKAGE_NAME}.tar.gz"
echo "  cd ${PACKAGE_NAME}"
echo "  chmod +x *.sh"
echo "  ./import-images.sh"
echo "  ./start-services.sh"
echo ""
echo -e "${GREEN}Package is ready for air-gapped deployment!${NC}"
echo ""
