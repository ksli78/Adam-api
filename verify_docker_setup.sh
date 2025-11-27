#!/bin/bash
# Verification script for Docker Compose Ollama setup

set -e

echo "=========================================="
echo "Docker Compose Ollama Verification"
echo "=========================================="
echo ""

# Check Docker is running
echo "1. Checking Docker daemon..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    exit 1
fi
echo "✅ Docker is running"
echo ""

# Check docker compose command
echo "2. Checking Docker Compose..."
if docker compose version > /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose > /dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Docker Compose not found!"
    exit 1
fi
echo "✅ Using: $COMPOSE_CMD"
echo ""

# Check GPU availability
echo "3. Checking NVIDIA GPU..."
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "✅ Found $GPU_COUNT GPU(s)"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv
else
    echo "⚠️ nvidia-smi not found - GPU support may not work"
fi
echo ""

# Verify compose file syntax
echo "4. Validating docker-compose.dual-ollama.yml..."
if $COMPOSE_CMD -f docker-compose.dual-ollama.yml config > /dev/null 2>&1; then
    echo "✅ Compose file is valid"
else
    echo "❌ Compose file has errors:"
    $COMPOSE_CMD -f docker-compose.dual-ollama.yml config
    exit 1
fi
echo ""

# Check if services are running
echo "5. Checking running services..."
RUNNING_SERVICES=$($COMPOSE_CMD -f docker-compose.dual-ollama.yml ps --services --filter "status=running" 2>/dev/null || echo "")

if echo "$RUNNING_SERVICES" | grep -q "ollama-gpu0"; then
    echo "✅ ollama-gpu0 is running"
else
    echo "⚠️ ollama-gpu0 is NOT running"
fi

if echo "$RUNNING_SERVICES" | grep -q "ollama-gpu1"; then
    echo "✅ ollama-gpu1 is running"
else
    echo "⚠️ ollama-gpu1 is NOT running"
fi

if echo "$RUNNING_SERVICES" | grep -q "adam-api"; then
    echo "✅ adam-api is running"
else
    echo "⚠️ adam-api is NOT running"
fi
echo ""

# Test Ollama endpoints
echo "6. Testing Ollama endpoints..."

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama GPU 0 (port 11434) is responding"
    MODELS_GPU0=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; models=json.load(sys.stdin).get('models',[]); print(', '.join(m['name'] for m in models) if models else 'No models')" 2>/dev/null || echo "Unable to parse")
    echo "   Models: $MODELS_GPU0"
else
    echo "❌ Ollama GPU 0 (port 11434) is NOT responding"
fi

if curl -s http://localhost:11435/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama GPU 1 (port 11435) is responding"
    MODELS_GPU1=$(curl -s http://localhost:11435/api/tags | python3 -c "import sys,json; models=json.load(sys.stdin).get('models',[]); print(', '.join(m['name'] for m in models) if models else 'No models')" 2>/dev/null || echo "Unable to parse")
    echo "   Models: $MODELS_GPU1"
else
    echo "❌ Ollama GPU 1 (port 11435) is NOT responding"
fi
echo ""

# Test Adam API
echo "7. Testing Adam API..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    HEALTH=$(curl -s http://localhost:8000/health)
    echo "✅ Adam API is healthy"
    echo "   Response: $HEALTH"
else
    echo "❌ Adam API is NOT responding"
fi
echo ""

# Check model loading
echo "8. Checking model loading status..."
PS_GPU0=$(curl -s http://localhost:11434/api/ps 2>/dev/null || echo "{}")
PS_GPU1=$(curl -s http://localhost:11435/api/ps 2>/dev/null || echo "{}")

echo "GPU 0 loaded models: $(echo $PS_GPU0 | python3 -c "import sys,json; models=json.load(sys.stdin).get('models',[]); print(', '.join(m['name'] for m in models) if models else 'None')" 2>/dev/null || echo "Unable to check")"
echo "GPU 1 loaded models: $(echo $PS_GPU1 | python3 -c "import sys,json; models=json.load(sys.stdin).get('models',[]); print(', '.join(m['name'] for m in models) if models else 'None')" 2>/dev/null || echo "Unable to check")"
echo ""

echo "=========================================="
echo "Verification Complete"
echo "=========================================="
