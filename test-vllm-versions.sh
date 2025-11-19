#!/bin/bash
# Test different vLLM versions to find one that works with Quadro RTX 5000

set -e

echo "=========================================="
echo "vLLM Version Compatibility Test"
echo "=========================================="
echo ""
echo "Testing different vLLM versions with SmolLM2-1.7B on Quadro RTX 5000"
echo ""

# Stop any running containers
echo "Stopping existing vLLM containers..."
docker stop vllm-gpu0 vllm-gpu1 2>/dev/null || true
docker rm vllm-gpu0 vllm-gpu1 2>/dev/null || true

# Wait for GPU to clear
sleep 5

# Check GPU memory
echo ""
echo "Initial GPU memory:"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
echo ""

# Array of vLLM versions to test
VERSIONS=(
    "v0.6.3.post1"    # Stable release
    "v0.6.2"          # Earlier stable
    "v0.5.5"          # Even earlier
    "latest"          # Current latest
)

MODEL="HuggingFaceTB/SmolLM2-1.7B-Instruct"
TEST_PORT=8099

for VERSION in "${VERSIONS[@]}"; do
    echo "=========================================="
    echo "Testing vLLM version: $VERSION"
    echo "=========================================="

    # Run vLLM with minimal settings
    docker run -d \
        --name vllm-test \
        --gpus '"device=0"' \
        -p ${TEST_PORT}:8000 \
        -v vllm_models:/root/.cache/huggingface \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        vllm/vllm-openai:${VERSION} \
        --model ${MODEL} \
        --dtype float16 \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.60 \
        --enforce-eager \
        --port 8000 \
        2>&1 | tee vllm-test-${VERSION}.log &

    DOCKER_PID=$!

    echo "Container started (PID: $DOCKER_PID)"
    echo "Waiting for startup (max 3 minutes)..."

    # Wait up to 3 minutes for startup
    TIMEOUT=180
    ELAPSED=0
    SUCCESS=false

    while [ $ELAPSED -lt $TIMEOUT ]; do
        # Check if container is still running
        if ! docker ps | grep -q vllm-test; then
            echo "❌ Container stopped unexpectedly!"
            echo ""
            echo "Last 30 lines of logs:"
            docker logs vllm-test 2>&1 | tail -30
            break
        fi

        # Check if health endpoint responds
        if curl -s http://localhost:${TEST_PORT}/health > /dev/null 2>&1; then
            echo "✅ SUCCESS! vLLM $VERSION started successfully"

            # Test generation
            echo "Testing generation..."
            RESPONSE=$(curl -s http://localhost:${TEST_PORT}/v1/completions \
                -H "Content-Type: application/json" \
                -d "{
                    \"model\": \"${MODEL}\",
                    \"prompt\": \"2+2=\",
                    \"max_tokens\": 10,
                    \"temperature\": 0
                }" 2>&1)

            if echo "$RESPONSE" | grep -q "choices"; then
                echo "✅ Generation test PASSED"
                echo ""
                echo "GPU Memory after successful load:"
                nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
                echo ""
                echo "===================="
                echo "WORKING VERSION FOUND: $VERSION"
                echo "===================="
                SUCCESS=true
            else
                echo "⚠️  Health check passed but generation failed"
                echo "Response: $RESPONSE"
            fi
            break
        fi

        # Check for OOM in logs
        if docker logs vllm-test 2>&1 | grep -i "out of memory" | tail -1; then
            echo "❌ Out of Memory error detected"
            echo ""
            echo "GPU memory at failure:"
            nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
            break
        fi

        # Progress indicator
        if [ $((ELAPSED % 10)) -eq 0 ]; then
            echo "  ... $ELAPSED seconds elapsed ..."
        fi

        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done

    if [ "$SUCCESS" = true ]; then
        echo ""
        echo "SUCCESS! You should use vLLM version: $VERSION"
        echo ""
        echo "Update your docker-compose files to use:"
        echo "    image: vllm/vllm-openai:${VERSION}"
        echo ""

        # Stop test container
        docker stop vllm-test 2>/dev/null || true
        docker rm vllm-test 2>/dev/null || true

        exit 0
    fi

    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "❌ Timeout after $TIMEOUT seconds"
    fi

    echo ""
    echo "Cleaning up..."
    docker stop vllm-test 2>/dev/null || true
    docker rm vllm-test 2>/dev/null || true

    # Wait for GPU to clear
    sleep 5
    echo ""
done

echo ""
echo "=========================================="
echo "All versions failed!"
echo "=========================================="
echo ""
echo "This suggests a deeper incompatibility between vLLM and Quadro RTX 5000."
echo ""
echo "Alternatives to try:"
echo "1. Text Generation Inference (TGI) by HuggingFace"
echo "2. LocalAI with CUDA backend"
echo "3. Optimized Ollama with vLLM-like performance"
echo "4. Run vLLM directly on host (not Docker)"
echo ""

exit 1
