#!/bin/bash
#
# Disk Space Diagnostic Script for vLLM Deployment
# Run this on RHEL9 to diagnose disk space issues
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "Disk Space Diagnostic for vLLM"
echo "========================================${NC}"
echo ""

echo -e "${BLUE}1. Overall disk usage:${NC}"
df -h | grep -E "Filesystem|/$|/usr|/var|/home|/data"
echo ""

echo -e "${BLUE}2. Docker root directory:${NC}"
DOCKER_ROOT=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $4}')
if [ -n "$DOCKER_ROOT" ]; then
    echo "Docker Root Dir: $DOCKER_ROOT"
    df -h "$DOCKER_ROOT" 2>/dev/null || echo "Cannot check Docker root"
else
    echo "Docker not running or not installed"
fi
echo ""

echo -e "${BLUE}3. Docker disk usage:${NC}"
docker system df 2>/dev/null || echo "Docker not running"
echo ""

echo -e "${BLUE}4. LVM information (if using LVM):${NC}"
if command -v vgs &> /dev/null; then
    echo "Volume Groups:"
    sudo vgs
    echo ""
    echo "Logical Volumes:"
    sudo lvs
    echo ""
    echo "Physical Volumes:"
    sudo pvs
else
    echo "LVM not available or not using LVM"
fi
echo ""

echo -e "${BLUE}5. Largest directories in /usr (top 10):${NC}"
sudo du -h /usr 2>/dev/null | sort -rh | head -n 10 || echo "Cannot read /usr"
echo ""

echo -e "${BLUE}6. Available space check for vLLM:${NC}"
echo "vLLM requirements:"
echo "  - Docker images: ~8GB"
echo "  - Model cache: ~24GB"
echo "  - Working space: ~10GB"
echo "  - Total needed: ~42GB"
echo ""

# Check each potential location
for dir in /var/lib/docker /data /home; do
    if [ -d "$dir" ]; then
        AVAIL=$(df -BG "$dir" | tail -1 | awk '{print $4}' | sed 's/G//')
        if [ "$AVAIL" -ge 42 ]; then
            echo -e "${GREEN}✓ $dir has ${AVAIL}GB available (sufficient)${NC}"
        elif [ "$AVAIL" -ge 25 ]; then
            echo -e "${YELLOW}⚠ $dir has ${AVAIL}GB available (marginal)${NC}"
        else
            echo -e "${RED}✗ $dir has ${AVAIL}GB available (insufficient)${NC}"
        fi
    fi
done
echo ""

echo -e "${BLUE}7. Recommendations:${NC}"
echo ""

# Determine best course of action
USR_AVAIL=$(df -BG /usr | tail -1 | awk '{print $4}' | sed 's/G//')
VAR_AVAIL=$(df -BG /var 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//' || echo "0")

if [ "$VAR_AVAIL" -ge 42 ]; then
    echo -e "${GREEN}Option 1 (Recommended): /var has enough space${NC}"
    echo "  Docker is likely already using /var/lib/docker (good!)"
    echo "  Clean up unused Docker resources (see below)"
elif command -v lvextend &> /dev/null; then
    echo -e "${YELLOW}Option 2: Extend LVM partition${NC}"
    echo "  You're using LVM - you can extend the partition"
    echo "  See: extend-lvm-partition.sh"
else
    echo -e "${YELLOW}Option 3: Move Docker data directory${NC}"
    echo "  Reconfigure Docker to use a larger partition"
    echo "  See: reconfigure-docker-data-dir.sh"
fi
echo ""

echo -e "${BLUE}Quick cleanup commands:${NC}"
echo ""
echo "# Remove unused Docker images"
echo "docker image prune -a"
echo ""
echo "# Remove unused containers"
echo "docker container prune"
echo ""
echo "# Remove unused volumes"
echo "docker volume prune"
echo ""
echo "# Clean DNF cache"
echo "sudo dnf clean all"
echo ""
