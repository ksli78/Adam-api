#!/bin/bash
# Fix dual Ollama port mapping issue

set -e

echo "=========================================="
echo "Fixing Dual Ollama Port Mapping"
echo "=========================================="
echo ""

# Detect docker compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Error: Neither 'docker compose' nor 'docker-compose' found"
    exit 1
fi

echo "Using: $COMPOSE_CMD"
echo ""

# Check current port mappings
echo "Current port mappings:"
docker ps --filter "name=ollama" --format "table {{.Names}}\t{{.Ports}}" || true
echo ""

# Stop and remove existing containers
echo "Stopping and removing existing Ollama containers..."
$COMPOSE_CMD -f docker-compose.dual-ollama.yml down

# Wait for cleanup
sleep 3

echo ""
echo "Recreating containers with correct port mappings..."
echo ""

# Start fresh
$COMPOSE_CMD -f docker-compose.dual-ollama.yml up -d ollama-gpu0 ollama-gpu1

# Wait for startup
sleep 5

echo ""
echo "=========================================="
echo "Verifying Port Mappings"
echo "=========================================="
echo ""

# Check new port mappings
docker ps --filter "name=ollama" --format "table {{.Names}}\t{{.Ports}}"

echo ""
echo "Expected:"
echo "  ollama-gpu0: 0.0.0.0:11434->11434/tcp"
echo "  ollama-gpu1: 0.0.0.0:11435->11434/tcp"
echo ""

# Test connectivity
echo "=========================================="
echo "Testing Connectivity"
echo "=========================================="
echo ""

# Wait for services to be ready
echo "Waiting for Ollama services to start (30 seconds)..."
sleep 30

echo ""
echo "Testing GPU 0 (port 11434)..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ GPU 0 is accessible on port 11434"
else
    echo "❌ GPU 0 is NOT accessible on port 11434"
fi

echo ""
echo "Testing GPU 1 (port 11435)..."
if curl -s http://localhost:11435/api/tags > /dev/null 2>&1; then
    echo "✅ GPU 1 is accessible on port 11435"
else
    echo "❌ GPU 1 is NOT accessible on port 11435"
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo ""
echo "If both tests passed, your dual Ollama setup is working correctly."
echo ""
echo "Useful commands:"
echo "  - Check ports: docker ps --filter 'name=ollama'"
echo "  - Test GPU 0: curl http://localhost:11434/api/tags"
echo "  - Test GPU 1: curl http://localhost:11435/api/tags"
echo ""
