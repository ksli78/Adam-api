#!/bin/bash
#
# Quick Fix for docker-compose Command Issue
# Run this on RHEL9 if you're getting "bad command" error
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Docker Compose Command Fix${NC}"
echo ""

# Check which version is available
if docker compose version &> /dev/null; then
    echo -e "${GREEN}✓ You have Docker Compose v2 (plugin)${NC}"
    echo "  Command: docker compose"
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓ You have Docker Compose v1 (standalone)${NC}"
    echo "  Command: docker-compose"
    COMPOSE_CMD="docker-compose"
else
    echo -e "${RED}✗ No docker-compose found${NC}"
    echo "Installing docker-compose v1..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    COMPOSE_CMD="docker-compose"
    echo -e "${GREEN}✓ Installed docker-compose${NC}"
fi

echo ""
echo -e "${BLUE}Testing command...${NC}"
$COMPOSE_CMD --version

echo ""
echo -e "${GREEN}To deploy vLLM manually, use:${NC}"
echo "  $COMPOSE_CMD -f docker-compose.vllm.yml up -d vllm"
echo ""
echo "To check status:"
echo "  $COMPOSE_CMD -f docker-compose.vllm.yml ps"
echo ""
echo "To view logs:"
echo "  $COMPOSE_CMD -f docker-compose.vllm.yml logs -f vllm"
echo ""
