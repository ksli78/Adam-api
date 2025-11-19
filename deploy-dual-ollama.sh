#!/bin/bash
# Deploy Dual Ollama Setup - One instance per GPU
# Simple, proven solution with 3x speedup using Mistral 7B

set -e

echo "=========================================="
echo "Dual Ollama Deployment (Mistral 7B)"
echo "=========================================="
echo ""

# Detect docker compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Error: Neither 'docker compose' nor 'docker-compose' found"
    echo "Please install Docker Compose"
    exit 1
fi

echo "Using: $COMPOSE_CMD"
echo ""

# Check for existing Ollama container to offer cleanup
EXISTING_OLLAMA=$(docker ps -a --filter "name=ollama" --format "{{.Names}}" | grep -E "^ollama$" | head -1 || true)

if [ -n "$EXISTING_OLLAMA" ]; then
    echo "Found existing Ollama container: $EXISTING_OLLAMA"
    echo ""

    # List current models
    if docker ps --filter "name=ollama" --format "{{.Names}}" | grep -q "^ollama$"; then
        echo "Current models:"
        docker exec ollama ollama list || true
        echo ""

        read -p "Would you like to remove old models to save disk space? (y/n): " CLEANUP_MODELS

        if [[ "$CLEANUP_MODELS" =~ ^[Yy]$ ]]; then
            echo ""
            echo "Recommended: Remove mistral-small:22b (saves ~12GB)"
            read -p "Remove mistral-small:22b? (y/n): " REMOVE_22B

            if [[ "$REMOVE_22B" =~ ^[Yy]$ ]]; then
                echo "Removing mistral-small:22b..."
                docker exec ollama ollama rm mistral-small:22b || echo "  (Model not found, skipping)"
                echo "✅ Model removed"
            fi
            echo ""
        fi
    fi
fi

# Stop current Ollama
echo "Stopping existing Ollama containers..."
docker stop ollama ollama-gpu0 ollama-gpu1 2>/dev/null || true
docker rm ollama ollama-gpu0 ollama-gpu1 2>/dev/null || true

# Wait for cleanup
sleep 3

# Check GPU availability
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
    echo ""
else
    echo "⚠️  Warning: nvidia-smi not found, cannot check GPU status"
    echo ""
fi

# Start dual Ollama
echo "Starting dual Ollama instances..."
echo ""

$COMPOSE_CMD -f docker-compose.dual-ollama.yml up -d ollama-gpu0 ollama-gpu1

echo ""
echo "Waiting for Ollama instances to start..."
sleep 5

# Check if both instances are running
echo ""
if docker ps | grep -q ollama-gpu0 && docker ps | grep -q ollama-gpu1; then
    echo "✅ Both Ollama instances started successfully"
else
    echo "❌ One or both Ollama instances failed to start"
    echo ""
    echo "Check logs:"
    echo "  docker logs ollama-gpu0"
    echo "  docker logs ollama-gpu1"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pulling Mistral 7B Model"
echo "=========================================="
echo ""
echo "This will download Mistral 7B on both instances (uses shared cache)"
echo "Download size: ~4GB (happens once)"
echo ""

# Pull model on GPU 0
echo "Pulling model on GPU 0..."
docker exec ollama-gpu0 ollama pull mistral:7b-instruct-v0.3

echo ""
echo "Pulling model on GPU 1..."
docker exec ollama-gpu1 ollama pull mistral:7b-instruct-v0.3

echo ""
echo "=========================================="
echo "Testing Both Instances"
echo "=========================================="
echo ""

# Test GPU 0
echo "Testing Ollama GPU 0 (port 11434)..."
TEST_RESPONSE_GPU0=$(curl -s http://localhost:11434/api/generate \
    -d '{
        "model": "mistral:7b-instruct-v0.3",
        "prompt": "2+2=",
        "stream": false,
        "options": {"num_predict": 5}
    }')

if echo "$TEST_RESPONSE_GPU0" | grep -q "response"; then
    echo "✅ GPU 0 test PASSED"
    ANSWER_GPU0=$(echo "$TEST_RESPONSE_GPU0" | python3 -c "import sys, json; print(json.load(sys.stdin)['response'])" 2>/dev/null || echo "")
    echo "   Answer: $ANSWER_GPU0"
else
    echo "❌ GPU 0 test FAILED"
    echo "   Response: $TEST_RESPONSE_GPU0"
fi

echo ""

# Test GPU 1
echo "Testing Ollama GPU 1 (port 11435)..."
TEST_RESPONSE_GPU1=$(curl -s http://localhost:11435/api/generate \
    -d '{
        "model": "mistral:7b-instruct-v0.3",
        "prompt": "3+3=",
        "stream": false,
        "options": {"num_predict": 5}
    }')

if echo "$TEST_RESPONSE_GPU1" | grep -q "response"; then
    echo "✅ GPU 1 test PASSED"
    ANSWER_GPU1=$(echo "$TEST_RESPONSE_GPU1" | python3 -c "import sys, json; print(json.load(sys.stdin)['response'])" 2>/dev/null || echo "")
    echo "   Answer: $ANSWER_GPU1"
else
    echo "❌ GPU 1 test FAILED"
    echo "   Response: $TEST_RESPONSE_GPU1"
fi

echo ""

# Show GPU memory usage
echo "=========================================="
echo "GPU Memory Usage"
echo "=========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
    echo ""
    echo "Expected: ~7-8GB per GPU"
else
    echo "  (nvidia-smi not available)"
fi

echo ""
echo "=========================================="
echo "✅ Dual Ollama Deployment Complete!"
echo "=========================================="
echo ""
echo "Architecture:"
echo "  - Ollama GPU 0: http://localhost:11434"
echo "  - Ollama GPU 1: http://localhost:11435"
echo "  - Model: mistral:7b-instruct-v0.3"
echo ""
echo "Performance:"
echo "  - Response time: ~15-20s (3x faster than Mistral-Small 22B)"
echo "  - Concurrent users: 2 requests in parallel (4 total with queuing)"
echo ""
echo "Next Steps:"
echo ""
echo "1. Update your Python code to use load balancing:"
echo "   from ollama_client_lb import OllamaClient"
echo "   import os"
echo "   hosts = os.getenv('OLLAMA_HOSTS', 'http://localhost:11434').split(',')"
echo "   client = OllamaClient(hosts=hosts)"
echo ""
echo "2. Set environment variable in .env:"
echo "   OLLAMA_HOSTS=http://ollama-gpu0:11434,http://ollama-gpu1:11434"
echo ""
echo "3. Update model in .env:"
echo "   LLM_MODEL=mistral:7b-instruct-v0.3"
echo ""
echo "4. Restart Adam API:"
echo "   $COMPOSE_CMD -f docker-compose.dual-ollama.yml up -d adam-api"
echo ""
echo "Useful commands:"
echo "  - View GPU 0 logs: docker logs -f ollama-gpu0"
echo "  - View GPU 1 logs: docker logs -f ollama-gpu1"
echo "  - Monitor GPUs: watch -n 1 nvidia-smi"
echo "  - List models GPU 0: docker exec ollama-gpu0 ollama list"
echo "  - List models GPU 1: docker exec ollama-gpu1 ollama list"
echo ""
