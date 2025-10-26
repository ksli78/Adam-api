#!/bin/bash
#
# RHEL9 Deployment Script for Air-Gapped RAG API
#
# This script automates the deployment process on RHEL9 systems.
# It expects the following files in the current directory:
#   - ollama.tar
#   - airgapped-rag-api.tar
#   - ollama-models.tar.gz
#   - docker-compose.airgapped.yml
#
# Usage:
#   sudo bash deploy-rhel9.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="/data/airgapped_rag"
VOLUME_NAME="airgapped_ollama_models"

echo -e "${BLUE}========================================"
echo "Air-Gapped RAG API - RHEL9 Deployment"
echo "========================================${NC}"
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}This script requires root privileges.${NC}"
  echo "Please run with sudo:"
  echo "  sudo bash deploy-rhel9.sh"
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
    echo ""
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

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠ docker-compose not found. Installing...${NC}"
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}✓ docker-compose installed${NC}"
else
    echo -e "${GREEN}✓ docker-compose is installed${NC}"
fi

# Check for required files
echo ""
echo "Checking for required files..."
missing_files=0

for file in ollama.tar airgapped-rag-api.tar ollama-models.tar.gz docker-compose.airgapped.yml; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}✗ Missing: $file${NC}"
        missing_files=1
    else
        echo -e "${GREEN}✓ Found: $file${NC}"
    fi
done

if [ $missing_files -eq 1 ]; then
    echo ""
    echo -e "${RED}Please ensure all required files are in the current directory.${NC}"
    exit 1
fi

echo ""
read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

#
# Step 2: Load Docker images
#
echo ""
echo -e "${BLUE}Step 2: Loading Docker images...${NC}"

echo "Loading Ollama image..."
docker load -i ollama.tar
echo -e "${GREEN}✓ Ollama image loaded${NC}"

echo "Loading Air-Gapped RAG API image..."
docker load -i airgapped-rag-api.tar
echo -e "${GREEN}✓ API image loaded${NC}"

# Verify images
echo ""
echo "Loaded images:"
docker images | grep -E "ollama|airgapped-rag-api"

#
# Step 3: Create data directory
#
echo ""
echo -e "${BLUE}Step 3: Creating data directory...${NC}"

if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
    echo -e "${GREEN}✓ Created $DATA_DIR${NC}"
else
    echo -e "${YELLOW}⚠ $DATA_DIR already exists${NC}"
fi

# Set ownership to the actual user
chown -R "$ACTUAL_USER:$ACTUAL_USER" "$DATA_DIR"
echo -e "${GREEN}✓ Set ownership to $ACTUAL_USER${NC}"

#
# Step 4: Restore Ollama models
#
echo ""
echo -e "${BLUE}Step 4: Restoring Ollama models...${NC}"

# Check if volume already exists
if docker volume inspect "$VOLUME_NAME" &> /dev/null; then
    echo -e "${YELLOW}⚠ Volume $VOLUME_NAME already exists${NC}"
    read -p "Do you want to replace it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume rm "$VOLUME_NAME" || true
    else
        echo "Skipping model restoration."
    fi
fi

if ! docker volume inspect "$VOLUME_NAME" &> /dev/null; then
    echo "Creating volume..."
    docker volume create "$VOLUME_NAME"
    echo -e "${GREEN}✓ Volume created${NC}"

    echo "Extracting models into volume (this may take a few minutes)..."
    docker run --rm -v "$VOLUME_NAME":/models -v "$(pwd)":/backup alpine tar xzf /backup/ollama-models.tar.gz -C /models
    echo -e "${GREEN}✓ Models restored${NC}"
else
    echo -e "${GREEN}✓ Using existing models volume${NC}"
fi

#
# Step 5: Stop any existing services
#
echo ""
echo -e "${BLUE}Step 5: Stopping existing services...${NC}"

if docker-compose -f docker-compose.airgapped.yml ps 2>/dev/null | grep -q "Up"; then
    echo "Stopping existing containers..."
    docker-compose -f docker-compose.airgapped.yml down
    echo -e "${GREEN}✓ Stopped${NC}"
else
    echo -e "${GREEN}✓ No existing services running${NC}"
fi

#
# Step 6: Start services
#
echo ""
echo -e "${BLUE}Step 6: Starting services...${NC}"

docker-compose -f docker-compose.airgapped.yml up -d

echo "Waiting for services to be healthy..."
sleep 5

# Wait for Ollama to be ready
echo "Waiting for Ollama..."
for i in {1..30}; do
    if docker exec ollama-service curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Ollama failed to start${NC}"
        exit 1
    fi
    sleep 2
done

# Wait for API to be ready
echo "Waiting for API..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ API failed to start${NC}"
        exit 1
    fi
    sleep 2
done

#
# Step 7: Verify deployment
#
echo ""
echo -e "${BLUE}Step 7: Verifying deployment...${NC}"

# Check Ollama models
echo "Checking Ollama models..."
docker exec ollama-service ollama list

# Check API health
echo ""
echo "Checking API health..."
curl -s http://localhost:8000/health | python3 -m json.tool || echo "API response received"

#
# Step 8: Configure firewall (optional)
#
echo ""
echo -e "${BLUE}Step 8: Firewall configuration${NC}"
read -p "Do you want to configure firewalld to allow access to the API? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v firewall-cmd &> /dev/null; then
        firewall-cmd --permanent --add-port=8000/tcp
        firewall-cmd --reload
        echo -e "${GREEN}✓ Firewall configured (port 8000 open)${NC}"
    else
        echo -e "${YELLOW}⚠ firewalld not installed${NC}"
    fi
fi

#
# Step 9: Create systemd service (optional)
#
echo ""
echo -e "${BLUE}Step 9: Systemd service${NC}"
read -p "Do you want to create a systemd service for auto-start? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat > /etc/systemd/system/airgapped-rag.service <<EOF
[Unit]
Description=Air-Gapped RAG API
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/local/bin/docker-compose -f docker-compose.airgapped.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.airgapped.yml down
User=$ACTUAL_USER

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable airgapped-rag.service
    echo -e "${GREEN}✓ Systemd service created and enabled${NC}"
    echo "  Service will start automatically on boot"
    echo "  Manage with: systemctl {start|stop|restart|status} airgapped-rag"
fi

#
# Deployment Complete
#
echo ""
echo -e "${GREEN}========================================"
echo "Deployment Complete!"
echo "========================================${NC}"
echo ""
echo "Services running:"
docker-compose -f docker-compose.airgapped.yml ps
echo ""
echo "Access points:"
echo "  API: http://$(hostname):8000"
echo "  API Documentation: http://$(hostname):8000/docs"
echo "  Ollama: http://$(hostname):11434"
echo ""
echo "Test the deployment:"
echo "  curl http://localhost:8000/health"
echo ""
echo "View logs:"
echo "  docker-compose -f docker-compose.airgapped.yml logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose -f docker-compose.airgapped.yml down"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Upload documents using the API"
echo "  2. Test queries"
echo "  3. Set up monitoring and backups"
echo ""
