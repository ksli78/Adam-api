# vLLM Troubleshooting Guide - Quadro RTX 5000

Quick fixes for common vLLM startup issues on RHEL9 with Quadro RTX 5000 GPUs.

---

## Quick Diagnostic

Run this first to identify the issue:
```bash
bash diagnose-vllm-errors.sh
```

Or manually check logs:
```bash
docker logs vllm-server 2>&1 | tail -50
```

---

## Common Issues & Solutions

### **Issue 1: CUDA Out of Memory**

**Symptoms:**
```
RuntimeError: CUDA out of memory
OutOfMemoryError: CUDA OOM
```

**Cause:** Mistral-Small 22B is too large for your GPU configuration

**Solution A: Reduce GPU Memory Utilization**
```bash
# Stop vLLM
sudo docker compose -f docker-compose.vllm.yml down vllm

# Edit docker-compose.vllm.yml
nano docker-compose.vllm.yml

# Change line ~20:
# FROM: --gpu-memory-utilization 0.90
# TO:   --gpu-memory-utilization 0.85

# Restart
sudo docker compose -f docker-compose.vllm.yml up -d vllm
```

**Solution B: Use Smaller Model (Recommended for RTX 5000)**
```bash
# Edit docker-compose.vllm.yml
nano docker-compose.vllm.yml

# Change line ~16:
# FROM: --model mistralai/Mistral-Small-Instruct-2409
# TO:   --model mistralai/Mistral-7B-Instruct-v0.3

# Change line ~17 (use 1 GPU instead of 2):
# FROM: --tensor-parallel-size 2
# TO:   --tensor-parallel-size 1

# Restart
sudo docker compose -f docker-compose.vllm.yml down
sudo docker compose -f docker-compose.vllm.yml up -d vllm
```

**Mistral-7B benefits:**
- ✅ Fits in 1 GPU (16GB)
- ✅ ~2x faster than Mistral-Small
- ✅ Still excellent quality for RAG
- ✅ Better for 3-5 concurrent users

---

### **Issue 2: Tensor Parallelism Error**

**Symptoms:**
```
AssertionError: tensor_parallel_size must be <= number of GPUs
NCCL error
Distributed initialization failed
```

**Cause:** Only 1 GPU detected but config says 2

**Solution:**
```bash
# Check GPUs
nvidia-smi

# If only 1 GPU is visible, edit docker-compose.vllm.yml:
nano docker-compose.vllm.yml

# Change line ~17:
# FROM: --tensor-parallel-size 2
# TO:   --tensor-parallel-size 1

# Change line ~30:
# FROM: count: 2
# TO:   count: 1

# Restart
sudo docker compose -f docker-compose.vllm.yml up -d vllm
```

---

### **Issue 3: Model Download Stuck/Failed**

**Symptoms:**
```
Downloading model...
Connection timeout
HTTP Error 503
```

**Cause:** Slow internet or HuggingFace Hub issues

**Solution A: Wait and Monitor**
```bash
# Model is ~24GB, can take 30+ minutes
docker logs -f vllm-server

# Look for download progress
```

**Solution B: Test Internet Connection**
```bash
curl -I https://huggingface.co
```

**Solution C: Pre-download Model (if download keeps failing)**
```bash
# On a machine with better internet
pip install huggingface-hub
python3 << EOF
from huggingface_hub import snapshot_download
snapshot_download("mistralai/Mistral-Small-Instruct-2409")
EOF

# Copy ~/.cache/huggingface to RHEL9 server
# Mount it in docker-compose.vllm.yml volumes section
```

---

### **Issue 4: Port Already in Use**

**Symptoms:**
```
Error: Port 8001 already in use
Address already in use: 8001
```

**Solution:**
```bash
# Find what's using port 8001
sudo lsof -i :8001
sudo netstat -tulpn | grep 8001

# Kill the process or change vLLM port
# Edit docker-compose.vllm.yml:
# FROM: - "8001:8000"
# TO:   - "8002:8000"
```

---

### **Issue 5: Container Crashes Immediately**

**Symptoms:**
```
docker ps -a
# Shows: Exited (1) 2 seconds ago
```

**Solution:**
```bash
# Get full error
docker logs vllm-server

# Common causes:
# 1. GPU driver issue
nvidia-smi  # Should show GPUs

# 2. nvidia-container-toolkit not installed
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 3. Wrong GPU architecture
# Quadro RTX 5000 = Turing (compute capability 7.5) = supported
```

---

### **Issue 6: "Application startup complete" but health check fails**

**Symptoms:**
```
# Logs show startup complete
# But: curl http://localhost:8001/health fails
```

**Solution:**
```bash
# Check if port is accessible
curl http://localhost:8001/health

# Check container network
docker inspect vllm-server | grep IPAddress

# Test from inside container
docker exec -it vllm-server curl http://localhost:8000/health

# If works inside but not outside = port mapping issue
# Check docker-compose.vllm.yml ports section
```

---

## Recommended Configuration for Quadro RTX 5000

Based on your 2x RTX 5000 (16GB each), here's the optimal config:

### **Option A: Mistral-7B (Single GPU - Recommended)**
```yaml
command: >
  --model mistralai/Mistral-7B-Instruct-v0.3
  --tensor-parallel-size 1
  --dtype float16
  --max-model-len 16384
  --gpu-memory-utilization 0.85
  --port 8000
```

**Pros:**
- ✅ Fits comfortably in 1 GPU
- ✅ Fast inference (~60-100 tokens/sec)
- ✅ Leaves 1 GPU free for other workloads
- ✅ Excellent quality for RAG

---

### **Option B: Mistral-Small (Dual GPU - Maximum Quality)**
```yaml
command: >
  --model mistralai/Mistral-Small-Instruct-2409
  --tensor-parallel-size 2
  --dtype float16
  --max-model-len 12288
  --gpu-memory-utilization 0.80
  --port 8000
```

**Pros:**
- ✅ Higher quality responses
- ⚠️ Slower than Mistral-7B
- ⚠️ Uses both GPUs
- ⚠️ Tight memory fit

---

## Verification Checklist

After fixing issues, verify vLLM is working:

```bash
# 1. Container is running
docker ps | grep vllm-server
# Should show: Up X minutes (healthy)

# 2. Health check passes
curl http://localhost:8001/health
# Should return: {"status":"ok"} or similar

# 3. Text generation works
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "What is 2+2?",
    "max_tokens": 50
  }'
# Should return JSON with generated text

# 4. GPUs are being used
nvidia-smi
# Should show vLLM process using GPU memory
```

---

## Performance Benchmarks (Expected)

| Configuration | Tokens/sec | Latency (RAG) | Concurrent Users |
|---------------|------------|---------------|------------------|
| Mistral-7B (1 GPU) | 60-100 | 10-15s | 3-5 excellent |
| Mistral-Small (2 GPU) | 40-70 | 15-20s | 3-5 good |
| Ollama Mistral-Small | 20-30 | 45-60s | 3-5 poor |

---

## Getting Help

If issues persist:

1. **Capture full logs:**
   ```bash
   docker logs vllm-server > vllm-logs.txt
   ```

2. **Check GPU status:**
   ```bash
   nvidia-smi > gpu-status.txt
   ```

3. **Check Docker config:**
   ```bash
   cat docker-compose.vllm.yml > config.txt
   ```

4. Share these files for debugging

---

## Quick Commands Reference

```bash
# Start vLLM
sudo docker compose -f docker-compose.vllm.yml up -d vllm

# Stop vLLM
sudo docker compose -f docker-compose.vllm.yml down vllm

# View logs (live)
docker logs -f vllm-server

# View logs (last 100 lines)
docker logs vllm-server 2>&1 | tail -100

# Restart vLLM
sudo docker compose -f docker-compose.vllm.yml restart vllm

# Check status
docker ps | grep vllm

# Check GPU usage
nvidia-smi

# Test health
curl http://localhost:8001/health

# Test generation
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistralai/Mistral-7B-Instruct-v0.3","prompt":"test","max_tokens":10}'
```

---

## Still Having Issues?

Run the diagnostic:
```bash
bash diagnose-vllm-errors.sh
```

This will identify the specific problem and suggest fixes.
