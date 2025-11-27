#!/bin/bash
# Start all Adam RAG services with proper ordering

set -e

# Detect compose command
if docker compose version > /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

echo "=========================================="
echo "Starting Adam RAG Services"
echo "=========================================="
echo ""

echo "Starting Ollama instances..."
$COMPOSE_CMD -f docker-compose.dual-ollama.yml up -d ollama-gpu0 ollama-gpu1

echo "Waiting for Ollama to be ready..."
sleep 10

# Wait for health checks
echo "Checking Ollama GPU 0..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama GPU 0 ready"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 5
done

echo "Checking Ollama GPU 1..."
for i in {1..30}; do
    if curl -s http://localhost:11435/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama GPU 1 ready"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 5
done

# Pull/verify model on both instances
MODEL="${LLM_MODEL:-mistral:latest}"
echo "Ensuring model '$MODEL' is available..."

docker exec ollama-gpu0 ollama pull $MODEL || true
docker exec ollama-gpu1 ollama pull $MODEL || true

# Warm up models (keep in memory)
echo "Warming up models..."
curl -s http://localhost:11434/api/generate -d "{\"model\": \"$MODEL\", \"prompt\": \"Hi\", \"stream\": false}" > /dev/null &
curl -s http://localhost:11435/api/generate -d "{\"model\": \"$MODEL\", \"prompt\": \"Hi\", \"stream\": false}" > /dev/null &
wait

echo "Starting Adam API..."
$COMPOSE_CMD -f docker-compose.dual-ollama.yml up -d adam-api

echo "Waiting for Adam API..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Adam API ready"
        break
    fi
    echo "  Waiting... ($i/60)"
    sleep 5
done

echo ""
echo "=========================================="
echo "All services started!"
echo "=========================================="
echo "Adam API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Ollama 0: http://localhost:11434"
echo "Ollama 1: http://localhost:11435"
