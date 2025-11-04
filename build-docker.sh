#!/bin/bash
# Build script for Adam API Docker image

set -e  # Exit on error

echo "========================================="
echo "Adam API - Docker Build Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build image
echo -e "${BLUE}Building Docker image...${NC}"
docker build -t adam-api:latest .

echo ""
echo -e "${GREEN}✓ Build complete!${NC}"
echo ""

# Show image info
echo -e "${BLUE}Image details:${NC}"
docker images adam-api:latest

echo ""
echo "========================================="
echo "Next steps:"
echo "========================================="
echo ""
echo "Option 1: Use with existing Ollama container"
echo "  See DOCKER_BUILD.md section 'Quick Start (Ollama Already Running)'"
echo ""
echo "Option 2: Use docker-compose (includes Ollama)"
echo "  $ docker-compose up -d"
echo "  $ docker exec ollama ollama pull llama3:8b"
echo ""
echo -e "${YELLOW}Documentation: DOCKER_BUILD.md${NC}"
echo ""
