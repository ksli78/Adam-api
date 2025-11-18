#!/bin/bash
#
# Cleanup Disk Space for vLLM Deployment
# Run on RHEL9 to free up space
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "Disk Space Cleanup"
echo "========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}This script requires root privileges for some operations.${NC}"
  echo "Please run with sudo:"
  echo "  sudo bash cleanup-disk-space.sh"
  exit 1
fi

# Show current disk usage
echo -e "${BLUE}Current disk usage:${NC}"
df -h | grep -E "Filesystem|/$|/usr|/var|/home"
echo ""

TOTAL_FREED=0

#
# 1. Clean DNF cache
#
echo -e "${BLUE}[1/7] Cleaning DNF package cache...${NC}"
BEFORE=$(df -BM /usr | tail -1 | awk '{print $3}' | sed 's/M//')
dnf clean all -y
AFTER=$(df -BM /usr | tail -1 | awk '{print $3}' | sed 's/M//')
FREED=$((BEFORE - AFTER))
if [ $FREED -gt 0 ]; then
    echo -e "${GREEN}✓ Freed ${FREED}MB${NC}"
    TOTAL_FREED=$((TOTAL_FREED + FREED))
else
    echo "No space freed"
fi
echo ""

#
# 2. Remove old kernels
#
echo -e "${BLUE}[2/7] Checking for old kernels...${NC}"
KERNEL_COUNT=$(rpm -q kernel | wc -l)
CURRENT_KERNEL=$(uname -r)

echo "Installed kernels: $KERNEL_COUNT"
echo "Current kernel: $CURRENT_KERNEL"

if [ $KERNEL_COUNT -gt 2 ]; then
    echo "Removing old kernels (keeping current + 1 backup)..."
    dnf remove -y $(dnf repoquery --installonly --latest-limit=-2 -q)
    echo -e "${GREEN}✓ Old kernels removed${NC}"
else
    echo "Only 2 kernels installed (good)"
fi
echo ""

#
# 3. Clean Docker
#
echo -e "${BLUE}[3/7] Cleaning Docker resources...${NC}"

if command -v docker &> /dev/null && systemctl is-active --quiet docker; then
    echo "Docker system usage before cleanup:"
    docker system df
    echo ""

    # Remove stopped containers
    echo "Removing stopped containers..."
    CONTAINERS=$(docker container prune -f 2>&1 | grep "Total reclaimed space" | awk '{print $4}')
    if [ -n "$CONTAINERS" ]; then
        echo -e "${GREEN}✓ Freed: $CONTAINERS${NC}"
    fi

    # Remove unused images
    echo "Removing unused images..."
    IMAGES=$(docker image prune -a -f 2>&1 | grep "Total reclaimed space" | awk '{print $4}')
    if [ -n "$IMAGES" ]; then
        echo -e "${GREEN}✓ Freed: $IMAGES${NC}"
    fi

    # Remove unused volumes
    echo "Removing unused volumes..."
    VOLUMES=$(docker volume prune -f 2>&1 | grep "Total reclaimed space" | awk '{print $4}')
    if [ -n "$VOLUMES" ]; then
        echo -e "${GREEN}✓ Freed: $VOLUMES${NC}"
    fi

    # Remove build cache
    echo "Removing build cache..."
    CACHE=$(docker builder prune -a -f 2>&1 | grep "Total" | awk '{print $3}')
    if [ -n "$CACHE" ]; then
        echo -e "${GREEN}✓ Freed: $CACHE${NC}"
    fi

    echo ""
    echo "Docker system usage after cleanup:"
    docker system df
else
    echo "Docker not running or not installed"
fi
echo ""

#
# 4. Clean systemd journal logs
#
echo -e "${BLUE}[4/7] Cleaning old journal logs...${NC}"
BEFORE=$(journalctl --disk-usage | awk '{print $7}' | sed 's/M//')
journalctl --vacuum-time=7d --vacuum-size=500M
AFTER=$(journalctl --disk-usage | awk '{print $7}' | sed 's/M//')
echo -e "${GREEN}✓ Journal logs cleaned${NC}"
echo ""

#
# 5. Clean temporary files
#
echo -e "${BLUE}[5/7] Cleaning temporary files...${NC}"
rm -rf /tmp/*
rm -rf /var/tmp/*
echo -e "${GREEN}✓ Temporary files cleaned${NC}"
echo ""

#
# 6. Find and offer to remove large files
#
echo -e "${BLUE}[6/7] Finding large files...${NC}"
echo "Searching for files > 500MB (this may take a minute)..."
echo ""

LARGE_FILES=$(find /var /usr /home -type f -size +500M 2>/dev/null | head -10)

if [ -n "$LARGE_FILES" ]; then
    echo "Large files found:"
    for file in $LARGE_FILES; do
        SIZE=$(du -h "$file" | awk '{print $1}')
        echo "  $SIZE - $file"
    done
    echo ""
    echo "Review these files manually and delete if not needed"
else
    echo "No large files found"
fi
echo ""

#
# 7. Find large directories
#
echo -e "${BLUE}[7/7] Largest directories in /usr...${NC}"
du -h /usr 2>/dev/null | sort -rh | head -n 5
echo ""

#
# Summary
#
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo "Current disk usage:"
df -h | grep -E "Filesystem|/$|/usr|/var|/home"
echo ""

# Check if enough space for vLLM
VAR_AVAIL=$(df -BG /var 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//' || echo "0")

if [ "$VAR_AVAIL" -ge 42 ]; then
    echo -e "${GREEN}✓ Sufficient space available for vLLM deployment (${VAR_AVAIL}GB)${NC}"
elif [ "$VAR_AVAIL" -ge 25 ]; then
    echo -e "${YELLOW}⚠ Marginal space (${VAR_AVAIL}GB). vLLM needs ~42GB${NC}"
    echo "Consider extending partition or moving Docker data directory"
else
    echo -e "${RED}✗ Insufficient space (${VAR_AVAIL}GB). vLLM needs ~42GB${NC}"
    echo ""
    echo "Options:"
    echo "  1. Extend LVM partition: sudo bash extend-lvm-partition.sh"
    echo "  2. Move Docker to larger partition: sudo bash reconfigure-docker-data-dir.sh"
fi
echo ""

echo "Additional cleanup options:"
echo "  - Remove old log files in /var/log"
echo "  - Remove old backups if any"
echo "  - Remove unused applications: dnf remove <package>"
echo ""
