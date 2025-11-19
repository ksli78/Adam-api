#!/bin/bash
# Deploy Text Generation Inference (TGI) - Alternative to vLLM
# Better support for Quadro RTX 5000 (Turing architecture)

set -e

echo "=========================================="
echo "TGI Deployment for Quadro RTX 5000"
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

# Stop vLLM if running
echo "Stopping any existing vLLM containers..."
docker stop vllm-gpu0 vllm-gpu1 vllm-server 2>/dev/null || true
docker rm vllm-gpu0 vllm-gpu1 vllm-server 2>/dev/null || true

# Wait for GPUs to clear
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

# Start TGI
echo "Starting TGI server..."
echo "This will download the model on first run (5-10 minutes)"
echo ""

$COMPOSE_CMD -f docker-compose.tgi.yml up -d tgi-server

echo ""
echo "TGI server starting..."
echo ""
echo "Monitor startup with:"
echo "  docker logs -f tgi-server"
echo ""

# Wait for startup (with progress)
echo "Waiting for TGI to start (max 5 minutes)..."
echo ""

TIMEOUT=300
ELAPSED=0
SUCCESS=false

while [ $ELAPSED -lt $TIMEOUT ]; do
    # Check if container is running
    if ! docker ps | grep -q tgi-server; then
        echo "❌ TGI container stopped unexpectedly!"
        echo ""
        echo "Check logs with: docker logs tgi-server"
        exit 1
    fi

    # Check health endpoint
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        SUCCESS=true
        break
    fi

    # Progress indicator
    if [ $((ELAPSED % 10)) -eq 0 ]; then
        echo "  ... $ELAPSED seconds elapsed (downloading/loading model) ..."
    fi

    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

echo ""

if [ "$SUCCESS" = true ]; then
    echo "=========================================="
    echo "✅ TGI Server Started Successfully!"
    echo "=========================================="
    echo ""

    # Get server info
    echo "Server information:"
    curl -s http://localhost:8001/info | python3 -m json.tool 2>/dev/null || echo "  (info endpoint not available)"
    echo ""

    # Test generation
    echo "Testing generation..."
    TEST_RESPONSE=$(curl -s http://localhost:8001/generate \
        -H "Content-Type: application/json" \
        -d '{
            "inputs": "2+2=",
            "parameters": {
                "max_new_tokens": 10,
                "temperature": 0
            }
        }')

    if echo "$TEST_RESPONSE" | grep -q "generated_text"; then
        echo "✅ Generation test PASSED"
        echo ""
        echo "Sample response:"
        echo "$TEST_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$TEST_RESPONSE"
    else
        echo "⚠️  Generation test returned unexpected response:"
        echo "$TEST_RESPONSE"
    fi

    echo ""

    # Show GPU usage
    echo "GPU Memory Usage:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
    else
        echo "  (nvidia-smi not available)"
    fi

    echo ""
    echo "=========================================="
    echo "TGI Deployment Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Update your Python code to use TGI client:"
    echo "   from tgi_client import TGIClient as Client"
    echo "   client = Client(host=\"http://tgi-server:80\")"
    echo ""
    echo "2. Start Adam API:"
    echo "   $COMPOSE_CMD -f docker-compose.tgi.yml up -d adam-api"
    echo ""
    echo "3. Test the full stack:"
    echo "   curl http://localhost:8000/query -d '{...}'"
    echo ""
    echo "Useful commands:"
    echo "  - View logs: docker logs -f tgi-server"
    echo "  - Check health: curl http://localhost:8001/health"
    echo "  - Monitor GPUs: watch -n 1 nvidia-smi"
    echo ""

else
    echo "=========================================="
    echo "❌ TGI Failed to Start"
    echo "=========================================="
    echo ""
    echo "Timeout after $TIMEOUT seconds"
    echo ""
    echo "Check logs with:"
    echo "  docker logs tgi-server"
    echo ""
    echo "Common issues:"
    echo "1. Out of memory: Try smaller model (edit docker-compose.tgi.yml)"
    echo "2. Model download slow: Check internet connection"
    echo "3. GPU access: Ensure nvidia-docker is configured"
    echo ""

    exit 1
fi
