#!/bin/bash
#
# Reconfigure Docker Data Directory
# Moves Docker data to a partition with more space
#
# Usage:
#   sudo bash reconfigure-docker-data-dir.sh /new/path
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}This script requires root privileges.${NC}"
  echo "Please run with sudo"
  exit 1
fi

# Get new data directory from argument
NEW_DATA_DIR="$1"

if [ -z "$NEW_DATA_DIR" ]; then
    echo "Usage: sudo bash reconfigure-docker-data-dir.sh /new/path"
    echo ""
    echo "Examples:"
    echo "  sudo bash reconfigure-docker-data-dir.sh /data/docker"
    echo "  sudo bash reconfigure-docker-data-dir.sh /home/docker"
    exit 1
fi

echo -e "${BLUE}========================================"
echo "Reconfigure Docker Data Directory"
echo "========================================${NC}"
echo ""

# Get current Docker root
CURRENT_ROOT=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $4}')

if [ -z "$CURRENT_ROOT" ]; then
    CURRENT_ROOT="/var/lib/docker"
fi

echo "Current Docker Root: $CURRENT_ROOT"
echo "New Docker Root:     $NEW_DATA_DIR"
echo ""

# Check if new directory is on a partition with enough space
PARENT_DIR=$(dirname "$NEW_DATA_DIR")
if [ ! -d "$PARENT_DIR" ]; then
    echo -e "${RED}✗ Parent directory does not exist: $PARENT_DIR${NC}"
    exit 1
fi

AVAIL_SPACE=$(df -BG "$PARENT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available space on target: ${AVAIL_SPACE}GB"

if [ "$AVAIL_SPACE" -lt 42 ]; then
    echo -e "${YELLOW}⚠ Warning: Target has less than 42GB available${NC}"
    echo "vLLM requires ~42GB for images + model"
    read -p "Continue anyway? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        exit 0
    fi
fi

echo ""
echo -e "${YELLOW}WARNING: This will:${NC}"
echo "  1. Stop Docker"
echo "  2. Move Docker data from $CURRENT_ROOT to $NEW_DATA_DIR"
echo "  3. Reconfigure Docker"
echo "  4. Restart Docker"
echo ""
echo "Existing containers will be preserved but stopped."
echo ""

read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled"
    exit 0
fi

echo ""

#
# Step 1: Stop Docker
#
echo -e "${BLUE}[1/6] Stopping Docker...${NC}"
systemctl stop docker
systemctl stop docker.socket
echo -e "${GREEN}✓ Docker stopped${NC}"
echo ""

#
# Step 2: Create new directory
#
echo -e "${BLUE}[2/6] Creating new directory...${NC}"
mkdir -p "$NEW_DATA_DIR"
echo -e "${GREEN}✓ Directory created: $NEW_DATA_DIR${NC}"
echo ""

#
# Step 3: Move data
#
echo -e "${BLUE}[3/6] Moving Docker data...${NC}"
echo "This may take several minutes..."

if [ -d "$CURRENT_ROOT" ]; then
    rsync -aP "$CURRENT_ROOT/" "$NEW_DATA_DIR/"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Data moved successfully${NC}"
    else
        echo -e "${RED}✗ Failed to move data${NC}"
        echo "Restarting Docker with original configuration..."
        systemctl start docker
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Current Docker root not found, creating new${NC}"
fi

echo ""

#
# Step 4: Backup old Docker directory
#
echo -e "${BLUE}[4/6] Backing up old directory...${NC}"
if [ -d "$CURRENT_ROOT" ]; then
    mv "$CURRENT_ROOT" "${CURRENT_ROOT}.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${GREEN}✓ Old directory backed up${NC}"
fi
echo ""

#
# Step 5: Configure Docker
#
echo -e "${BLUE}[5/6] Configuring Docker...${NC}"

# Create Docker config directory if it doesn't exist
mkdir -p /etc/docker

# Create or update daemon.json
DAEMON_JSON="/etc/docker/daemon.json"

if [ -f "$DAEMON_JSON" ]; then
    echo "Backing up existing daemon.json..."
    cp "$DAEMON_JSON" "${DAEMON_JSON}.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Create new daemon.json
cat > "$DAEMON_JSON" << EOF
{
  "data-root": "$NEW_DATA_DIR",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

echo -e "${GREEN}✓ Docker configured to use $NEW_DATA_DIR${NC}"
echo ""

#
# Step 6: Start Docker
#
echo -e "${BLUE}[6/6] Starting Docker...${NC}"
systemctl start docker

# Wait for Docker to start
sleep 5

# Verify Docker is running
if systemctl is-active --quiet docker; then
    echo -e "${GREEN}✓ Docker started successfully${NC}"
else
    echo -e "${RED}✗ Docker failed to start${NC}"
    echo "Check logs: journalctl -u docker"
    exit 1
fi

echo ""

#
# Verify configuration
#
echo -e "${BLUE}Verifying configuration...${NC}"
NEW_ROOT=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $4}')

if [ "$NEW_ROOT" = "$NEW_DATA_DIR" ]; then
    echo -e "${GREEN}✓ Docker is now using: $NEW_ROOT${NC}"
else
    echo -e "${RED}✗ Configuration verification failed${NC}"
    echo "Expected: $NEW_DATA_DIR"
    echo "Got: $NEW_ROOT"
fi

echo ""

# Show disk usage
echo "New location disk usage:"
df -h "$NEW_DATA_DIR"
echo ""

# Show Docker disk usage
echo "Docker disk usage:"
docker system df
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Docker Data Directory Reconfigured!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Old Docker data (backed up):"
echo "  ${CURRENT_ROOT}.backup.*"
echo ""
echo "You can safely delete the backup after verifying everything works:"
echo "  sudo rm -rf ${CURRENT_ROOT}.backup.*"
echo ""
echo "Restart any containers that were running:"
echo "  docker ps -a"
echo "  docker start <container_name>"
echo ""
