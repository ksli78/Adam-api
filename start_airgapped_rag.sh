#!/bin/bash
#
# Startup script for Air-Gapped RAG API
#
# This script:
# 1. Checks if Ollama is running
# 2. Verifies required models are installed
# 3. Starts the FastAPI application
#

set -e

echo "========================================"
echo "Air-Gapped RAG API Startup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
LLM_MODEL="${OLLAMA_LLM_MODEL:-llama3:8b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Function to check if Ollama is running
check_ollama() {
    echo "Checking Ollama connection..."
    if curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is running at $OLLAMA_URL${NC}"
        return 0
    else
        echo -e "${RED}✗ Ollama is not running or not accessible${NC}"
        echo ""
        echo "Please start Ollama first:"
        echo "  ollama serve"
        echo ""
        return 1
    fi
}

# Function to check if a model is installed
check_model() {
    local model=$1
    echo "Checking for model: $model..."

    if curl -s "$OLLAMA_URL/api/tags" | grep -q "\"name\":\"$model\""; then
        echo -e "${GREEN}✓ Model $model is installed${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Model $model is not installed${NC}"
        echo ""
        echo "Install it with:"
        echo "  ollama pull $model"
        echo ""
        return 1
    fi
}

# Main checks
echo "Step 1: Checking Ollama..."
if ! check_ollama; then
    exit 1
fi

echo ""
echo "Step 2: Checking required models..."
models_ok=true

if ! check_model "$EMBED_MODEL"; then
    models_ok=false
fi

if ! check_model "$LLM_MODEL"; then
    models_ok=false
fi

if [ "$models_ok" = false ]; then
    echo ""
    echo -e "${RED}Please install missing models before continuing${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All prerequisites satisfied${NC}"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠ No virtual environment detected${NC}"
    echo "Consider activating a virtual environment:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
fi

# Check if dependencies are installed
echo "Step 3: Checking Python dependencies..."
if ! python -c "import fastapi, chromadb, ollama" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Some dependencies are missing${NC}"
    echo "Install them with:"
    echo "  pip install -r requirements.txt"
    echo ""
    read -p "Install dependencies now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -r requirements.txt
    else
        exit 1
    fi
fi

echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo ""

# Start the API
echo "========================================"
echo "Starting Air-Gapped RAG API"
echo "========================================"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Embedding Model: $EMBED_MODEL"
echo "LLM Model: $LLM_MODEL"
echo ""
echo "API Documentation will be available at:"
echo "  http://localhost:$PORT/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Start the API with uvicorn
python -m uvicorn airgapped_rag:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info
