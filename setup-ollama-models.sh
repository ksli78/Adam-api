#!/bin/bash
#
# Setup script to pull Ollama models into Docker container
#
# Usage (after starting Ollama container):
#   ./setup-ollama-models.sh
#
# Or on Windows (Git Bash or WSL):
#   bash setup-ollama-models.sh
#

set -e

echo "========================================"
echo "Ollama Models Setup"
echo "========================================"
echo ""

# Configuration
CONTAINER_NAME="${OLLAMA_CONTAINER:-ollama-airgapped-rag}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
LLM_MODEL="${OLLAMA_LLM_MODEL:-llama3:8b}"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running"
    echo ""
    echo "Start it with:"
    echo "  docker-compose -f docker-compose.ollama.yml up -d"
    echo ""
    exit 1
fi

echo "✓ Container '${CONTAINER_NAME}' is running"
echo ""

# Function to pull a model
pull_model() {
    local model=$1
    echo "Pulling model: $model"
    echo "This may take several minutes depending on model size..."
    echo ""

    docker exec -it "$CONTAINER_NAME" ollama pull "$model"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Successfully pulled: $model"
        echo ""
    else
        echo ""
        echo "✗ Failed to pull: $model"
        echo ""
        return 1
    fi
}

# Pull embedding model
echo "Step 1: Pulling embedding model..."
echo "Model: $EMBED_MODEL (approx. 274 MB)"
echo ""
pull_model "$EMBED_MODEL"

# Pull LLM model
echo "Step 2: Pulling LLM model..."
echo "Model: $LLM_MODEL (approx. 4.7 GB)"
echo ""
pull_model "$LLM_MODEL"

# List installed models
echo "========================================"
echo "Installed Models"
echo "========================================"
docker exec "$CONTAINER_NAME" ollama list

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "You can now:"
echo "  1. Run the API locally: python airgapped_rag.py"
echo "  2. Or use Docker Compose: docker-compose -f docker-compose.airgapped.yml up"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
