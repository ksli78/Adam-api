#!/bin/bash
#
# GPU Memory Diagnostic for vLLM
# Checks what's actually using GPU memory
#

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "GPU Memory Diagnostic"
echo "========================================${NC}"
echo ""

echo -e "${BLUE}1. GPU Configuration:${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo ""

echo -e "${BLUE}2. Processes using GPU memory:${NC}"
nvidia-smi
echo ""

echo -e "${BLUE}3. Docker container GPU access:${NC}"
if docker ps | grep -q vllm-server; then
    echo "vLLM container is running"
    echo ""
    echo "Checking CUDA_VISIBLE_DEVICES inside container:"
    docker exec vllm-server nvidia-smi -L 2>/dev/null || echo "Cannot access nvidia-smi in container"
    echo ""
    echo "Environment variables:"
    docker exec vllm-server env | grep -i cuda
else
    echo "vLLM container is not running"
fi
echo ""

echo -e "${BLUE}4. Docker GPU configuration:${NC}"
docker inspect vllm-server 2>/dev/null | grep -A 20 "DeviceRequests" || echo "Container not found"
echo ""

echo -e "${BLUE}5. Checking if Ollama is using GPU:${NC}"
if docker ps | grep -q ollama; then
    echo -e "${YELLOW}⚠ Ollama container is running!${NC}"
    echo "This might be using GPU memory"
    docker ps | grep ollama
    echo ""
    echo "Stop Ollama with:"
    echo "  docker stop \$(docker ps -q --filter name=ollama)"
elif pgrep -x ollama > /dev/null; then
    echo -e "${YELLOW}⚠ Ollama process is running (not in Docker)!${NC}"
    echo "This might be using GPU memory"
    ps aux | grep ollama | grep -v grep
    echo ""
    echo "Stop with: sudo systemctl stop ollama"
else
    echo -e "${GREEN}✓ Ollama is not running${NC}"
fi
echo ""

echo -e "${BLUE}6. Available memory calculation:${NC}"
GPU0_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
GPU1_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 2>/dev/null || echo "0")

echo "GPU 0 free memory: ${GPU0_FREE} MiB"
echo "GPU 1 free memory: ${GPU1_FREE} MiB"
echo "Total free: $((GPU0_FREE + GPU1_FREE)) MiB"
echo ""

if [ "$GPU0_FREE" -lt 8000 ] || [ "$GPU1_FREE" -lt 8000 ]; then
    echo -e "${RED}⚠ Less than 8GB free on one or more GPUs${NC}"
    echo "An 8B model needs ~8-10GB free"
    echo ""
    echo "Something is using your GPU memory!"
fi
echo ""

echo -e "${BLUE}7. Recommendations:${NC}"
echo ""

# Check if we need to stop something
if docker ps | grep -q ollama; then
    echo -e "${YELLOW}ACTION REQUIRED:${NC} Stop Ollama container"
    echo "  docker stop \$(docker ps -q --filter name=ollama)"
    echo ""
fi

# Check tensor parallelism
if docker ps | grep -q vllm-server; then
    TENSOR_SIZE=$(docker logs vllm-server 2>&1 | grep -i "tensor.parallel" | head -1)
    if [ -n "$TENSOR_SIZE" ]; then
        echo "Tensor parallel config: $TENSOR_SIZE"
    fi
fi

echo -e "${GREEN}========================================${NC}"
