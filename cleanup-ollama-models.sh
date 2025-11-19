#!/bin/bash
# Clean up old Ollama models to free disk space

set -e

echo "=========================================="
echo "Ollama Model Cleanup"
echo "=========================================="
echo ""

# Check if any Ollama containers are running
OLLAMA_CONTAINERS=$(docker ps -a --filter "name=ollama" --format "{{.Names}}" | head -1)

if [ -z "$OLLAMA_CONTAINERS" ]; then
    echo "❌ No Ollama containers found"
    echo "Please start Ollama first"
    exit 1
fi

echo "Using container: $OLLAMA_CONTAINERS"
echo ""

# List all models
echo "=========================================="
echo "Current Models"
echo "=========================================="
docker exec $OLLAMA_CONTAINERS ollama list

echo ""
echo "=========================================="
echo "Model Storage Details"
echo "=========================================="
echo ""

# Get size of Ollama volume
VOLUME_NAME="ollama-models"
if docker volume ls | grep -q $VOLUME_NAME; then
    echo "Volume: $VOLUME_NAME"
    echo ""

    # Show volume size (approximate)
    VOLUME_PATH=$(docker volume inspect $VOLUME_NAME --format '{{.Mountpoint}}' 2>/dev/null || echo "")
    if [ -n "$VOLUME_PATH" ] && [ -d "$VOLUME_PATH" ]; then
        echo "Volume size:"
        sudo du -sh "$VOLUME_PATH" 2>/dev/null || echo "  (Unable to check - need sudo)"
        echo ""
        echo "Detailed breakdown:"
        sudo du -sh "$VOLUME_PATH"/* 2>/dev/null | sort -h || echo "  (Unable to check - need sudo)"
    fi
else
    echo "Volume $VOLUME_NAME not found"
fi

echo ""
echo "=========================================="
echo "Cleanup Options"
echo "=========================================="
echo ""

# Parse model list
MODELS=$(docker exec $OLLAMA_CONTAINERS ollama list | tail -n +2 | awk '{print $1}')

if [ -z "$MODELS" ]; then
    echo "No models found to clean up"
    exit 0
fi

echo "Available models to remove:"
echo ""

INDEX=1
declare -a MODEL_ARRAY
for MODEL in $MODELS; do
    # Get model size
    SIZE=$(docker exec $OLLAMA_CONTAINERS ollama list | grep "^$MODEL" | awk '{print $2}')
    echo "  [$INDEX] $MODEL ($SIZE)"
    MODEL_ARRAY[$INDEX]=$MODEL
    INDEX=$((INDEX + 1))
done

echo ""
echo "  [0] Remove ALL models (fresh start)"
echo "  [Q] Quit (don't remove anything)"
echo ""

read -p "Select models to remove (comma-separated, e.g., 1,3 or 0 for all): " SELECTION

# Handle quit
if [[ "$SELECTION" =~ ^[Qq]$ ]]; then
    echo "Cleanup cancelled"
    exit 0
fi

# Handle remove all
if [ "$SELECTION" = "0" ]; then
    echo ""
    echo "⚠️  WARNING: This will remove ALL models!"
    read -p "Are you sure? (yes/no): " CONFIRM

    if [ "$CONFIRM" = "yes" ]; then
        echo ""
        echo "Removing all models..."
        for MODEL in $MODELS; do
            echo "  Removing $MODEL..."
            docker exec $OLLAMA_CONTAINERS ollama rm $MODEL
        done
        echo ""
        echo "✅ All models removed"
    else
        echo "Cancelled"
        exit 0
    fi
else
    # Remove selected models
    IFS=',' read -ra SELECTED <<< "$SELECTION"

    echo ""
    echo "Removing selected models..."
    for NUM in "${SELECTED[@]}"; do
        # Trim whitespace
        NUM=$(echo $NUM | xargs)

        if [ -n "${MODEL_ARRAY[$NUM]}" ]; then
            MODEL="${MODEL_ARRAY[$NUM]}"
            echo "  Removing $MODEL..."
            docker exec $OLLAMA_CONTAINERS ollama rm $MODEL
        else
            echo "  Invalid selection: $NUM (skipped)"
        fi
    done
    echo ""
    echo "✅ Selected models removed"
fi

echo ""
echo "=========================================="
echo "Remaining Models"
echo "=========================================="
docker exec $OLLAMA_CONTAINERS ollama list

echo ""
echo "=========================================="
echo "Additional Cleanup (Optional)"
echo "=========================================="
echo ""
echo "To free even more space, you can:"
echo ""
echo "1. Remove unused Docker images:"
echo "   docker image prune -a"
echo ""
echo "2. Remove unused Docker volumes:"
echo "   docker volume prune"
echo ""
echo "3. Remove build cache:"
echo "   docker builder prune -a"
echo ""
echo "⚠️  Only run these if you're sure you don't need them!"
echo ""
