#!/bin/bash
#
# vLLM Error Diagnostic Script
# Analyzes vLLM container logs to identify common issues
#

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "vLLM Error Diagnostic"
echo "========================================${NC}"
echo ""

# Check if container exists
if ! docker ps -a | grep -q vllm-server; then
    echo -e "${RED}✗ vLLM container not found${NC}"
    echo "Run: sudo docker compose -f docker-compose.vllm.yml up -d vllm"
    exit 1
fi

# Check container status
STATUS=$(docker inspect -f '{{.State.Status}}' vllm-server)
echo "Container status: $STATUS"
echo ""

# Get last 100 lines of logs
echo -e "${BLUE}Analyzing logs...${NC}"
echo ""

LOGS=$(docker logs vllm-server 2>&1 | tail -100)

# Check for common errors

# 1. CUDA/GPU errors
if echo "$LOGS" | grep -qi "CUDA\|cuda\|GPU\|gpu"; then
    echo -e "${YELLOW}[GPU-RELATED MESSAGES]${NC}"
    echo "$LOGS" | grep -i "CUDA\|cuda\|GPU\|gpu" | tail -10
    echo ""
fi

# 2. Out of memory errors
if echo "$LOGS" | grep -qi "out of memory\|OOM\|CUDA out of memory"; then
    echo -e "${RED}[ERROR: OUT OF MEMORY]${NC}"
    echo "vLLM is running out of GPU memory"
    echo ""
    echo "Solutions:"
    echo "  1. Reduce GPU memory utilization:"
    echo "     Edit docker-compose.vllm.yml:"
    echo "     Change: --gpu-memory-utilization 0.90"
    echo "     To:     --gpu-memory-utilization 0.85"
    echo ""
    echo "  2. Use a smaller model:"
    echo "     Change: --model mistralai/Mistral-Small-Instruct-2409"
    echo "     To:     --model mistralai/Mistral-7B-Instruct-v0.3"
    echo ""
    echo "  3. Check GPU memory:"
    nvidia-smi
    exit 1
fi

# 3. Model download errors
if echo "$LOGS" | grep -qi "HuggingFace\|huggingface\|download\|Connection\|HTTP Error"; then
    echo -e "${YELLOW}[MODEL DOWNLOAD MESSAGES]${NC}"
    echo "$LOGS" | grep -i "download\|HuggingFace\|HTTP" | tail -10
    echo ""

    if echo "$LOGS" | grep -qi "Error\|Failed\|Connection"; then
        echo -e "${RED}[ERROR: Model download failed]${NC}"
        echo ""
        echo "Possible causes:"
        echo "  1. No internet connection"
        echo "  2. HuggingFace Hub is down"
        echo "  3. Model name is incorrect"
        echo ""
        echo "Test internet:"
        echo "  curl -I https://huggingface.co"
        exit 1
    fi
fi

# 4. Tensor parallelism errors
if echo "$LOGS" | grep -qi "tensor.parallel\|distributed\|nccl"; then
    echo -e "${YELLOW}[TENSOR PARALLELISM MESSAGES]${NC}"
    echo "$LOGS" | grep -i "tensor\|parallel\|distributed" | tail -10
    echo ""

    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$GPU_COUNT" -lt 2 ]; then
        echo -e "${RED}[ERROR: Not enough GPUs]${NC}"
        echo "vLLM is configured for 2 GPUs (--tensor-parallel-size 2)"
        echo "But only $GPU_COUNT GPU detected"
        echo ""
        echo "Solution: Edit docker-compose.vllm.yml"
        echo "  Change: --tensor-parallel-size 2"
        echo "  To:     --tensor-parallel-size 1"
        exit 1
    fi
fi

# 5. Port binding errors
if echo "$LOGS" | grep -qi "port.*already.*use\|Address already in use"; then
    echo -e "${RED}[ERROR: Port already in use]${NC}"
    echo "Port 8001 is already in use"
    echo ""
    echo "Check what's using it:"
    echo "  sudo lsof -i :8001"
    echo "  sudo netstat -tulpn | grep 8001"
    exit 1
fi

# 6. Permission errors
if echo "$LOGS" | grep -qi "permission denied\|cannot access"; then
    echo -e "${RED}[ERROR: Permission denied]${NC}"
    echo "$LOGS" | grep -i "permission\|cannot access"
    echo ""
    echo "Check Docker permissions and volume mounts"
    exit 1
fi

# 7. Check for successful startup
if echo "$LOGS" | grep -qi "Application startup complete\|Uvicorn running"; then
    echo -e "${GREEN}✓ vLLM appears to have started successfully!${NC}"
    echo ""
    echo "Recent logs:"
    echo "$LOGS" | tail -20
    echo ""
    echo "Test with:"
    echo "  curl http://localhost:8001/health"
    exit 0
fi

# 8. Still downloading model?
if echo "$LOGS" | grep -qi "Downloading\|Fetching"; then
    echo -e "${YELLOW}[INFO: Model download in progress]${NC}"
    echo ""
    echo "$LOGS" | grep -i "download\|fetch" | tail -5
    echo ""
    echo "Model download can take 10-30 minutes (~24GB)"
    echo ""
    echo "Monitor progress with:"
    echo "  docker logs -f vllm-server"
    exit 0
fi

# 9. General error check
if echo "$LOGS" | grep -qi "error\|failed\|exception"; then
    echo -e "${RED}[ERRORS FOUND]${NC}"
    echo ""
    echo "$LOGS" | grep -i "error\|failed\|exception" | tail -20
    echo ""
fi

# Show full recent logs if no specific issue found
echo -e "${YELLOW}[RECENT LOGS - Last 30 lines]${NC}"
echo ""
echo "$LOGS" | tail -30
echo ""

echo -e "${BLUE}========================================"
echo "Diagnostic Complete"
echo "========================================${NC}"
echo ""
echo "For full logs, run:"
echo "  docker logs vllm-server"
echo ""
echo "For live monitoring:"
echo "  docker logs -f vllm-server"
echo ""
