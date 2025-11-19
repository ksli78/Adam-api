# Dual-GPU vLLM Deployment Guide

Complete guide for deploying vLLM across both Quadro RTX 5000 GPUs for maximum performance.

---

## Overview

**Architecture:**
```
┌─────────────────────────────────────────┐
│  Adam API (Port 8000)                   │
│  ├─ Load Balancer                       │
│  ├─> vLLM GPU 0 (Port 8001)            │
│  └─> vLLM GPU 1 (Port 8002)            │
└─────────────────────────────────────────┘
         │                    │
         ▼                    ▼
   ┌──────────┐        ┌──────────┐
   │ GPU 0    │        │ GPU 1    │
   │ RTX 5000 │        │ RTX 5000 │
   │ 16GB     │        │ 16GB     │
   └──────────┘        └──────────┘
```

**Benefits:**
- ✅ **2x throughput** - Handle 2 requests simultaneously
- ✅ **Perfect for 3-5 users** - No queuing for first 2 users
- ✅ **Redundancy** - If one GPU fails, other continues
- ✅ **Load balancing** - Automatic distribution across GPUs
- ✅ **Better resource utilization** - Both GPUs fully utilized

---

## Quick Start

### **1. Deploy Dual vLLM Instances**

```bash
# On RHEL9 server
cd /path/to/Adam-api

# Stop any existing vLLM
docker compose -f docker-compose.vllm.yml down

# Start dual vLLM setup
docker compose -f docker-compose.dual-vllm.yml up -d vllm-gpu0 vllm-gpu1

# Monitor both instances
docker logs -f vllm-gpu0 &
docker logs -f vllm-gpu1 &
```

**Expected output:**
```
vllm-gpu0 | Downloading ibm-granite/granite-3.1-8b-instruct...
vllm-gpu1 | Loading model from cache...  (uses same downloaded model)
vllm-gpu0 | Application startup complete
vllm-gpu1 | Application startup complete
```

**First startup:** 5-10 minutes (model download)
**Subsequent startups:** 1-2 minutes

---

### **2. Verify Both Instances**

```bash
# Check GPU 0 health
curl http://localhost:8001/health

# Check GPU 1 health
curl http://localhost:8002/health

# Both should return: {"status":"ok"} or similar
```

**Test generation on each GPU:**

```bash
# Test GPU 0
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3.1-8b-instruct",
    "prompt": "What is 2+2?",
    "max_tokens": 50
  }'

# Test GPU 1
curl http://localhost:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-3.1-8b-instruct",
    "prompt": "What is 3+3?",
    "max_tokens": 50
  }'
```

Both should return valid JSON with generated text.

---

### **3. Verify GPU Usage**

```bash
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1234    C   ...python3.10                     10500MiB |
|    1   N/A  N/A      5678    C   ...python3.10                     10500MiB |
+-----------------------------------------------------------------------------+
```

✅ **Both GPUs should show ~10-11GB usage**

---

## Performance Comparison

### **Before: Single GPU or Ollama**

| Concurrent Users | Response Time |
|------------------|---------------|
| 1 user | 45s (Ollama) / 15s (vLLM single) |
| 2 users | 90s / 30s (queued) |
| 3 users | 135s / 45s (queued) |
| 4 users | 180s / 60s (queued) |
| 5 users | 225s / 75s (queued) |

### **After: Dual GPU vLLM**

| Concurrent Users | Response Time |
|------------------|---------------|
| 1 user | **10-15s** ✅ |
| 2 users | **10-15s** ✅ (parallel!) |
| 3 users | **20-25s** ✅ (1 queued) |
| 4 users | **20-25s** ✅ (parallel!) |
| 5 users | **30-35s** ✅ (1 queued) |

**Key improvement:** First 2 users get instant parallel processing!

---

## Code Integration

### **Option 1: Update Existing Code (Recommended)**

Replace the vLLM client import in your Python files:

**Before:**
```python
from vllm_client import VLLMClient

client = VLLMClient(host="http://vllm:8000")
```

**After:**
```python
from vllm_client_lb import VLLMClient
import os

# Load balance across both GPUs
hosts = os.getenv("VLLM_HOSTS", "http://vllm-gpu0:8000,http://vllm-gpu1:8000").split(",")
client = VLLMClient(hosts=hosts, strategy="round-robin")
```

**Files to update:**
1. `airgapped_rag_advanced.py`
2. `sql_query_handler.py`
3. `query_classifier.py`
4. `metadata_extractor.py`

---

### **Option 2: Environment Variable (Easiest)**

Set `VLLM_HOSTS` in your `.env` file:

```bash
VLLM_HOSTS=http://vllm-gpu0:8000,http://vllm-gpu1:8000
```

Then update one line in your code:

```python
from vllm_client_lb import VLLMClient
import os

# Auto-detects multiple hosts
hosts = os.getenv("VLLM_HOSTS", "http://localhost:8000").split(",")
client = VLLMClient(hosts=hosts)
```

---

## Load Balancing Strategies

### **Round-Robin (Default)**

Distributes requests evenly:
```
Request 1 → GPU 0
Request 2 → GPU 1
Request 3 → GPU 0
Request 4 → GPU 1
...
```

**Best for:** Consistent load distribution

**Usage:**
```python
client = VLLMClient(hosts=hosts, strategy="round-robin")
```

---

### **Random**

Randomly selects GPU for each request:

**Best for:** Simple load balancing without state

**Usage:**
```python
client = VLLMClient(hosts=hosts, strategy="random")
```

---

## Monitoring

### **Real-time GPU Monitoring**

```bash
# Watch GPU usage (updates every 1 second)
watch -n 1 nvidia-smi

# Look for:
# - Both GPUs showing ~10-11GB usage
# - Both GPUs at 80-95% utilization during inference
```

---

### **Container Logs**

```bash
# GPU 0 logs
docker logs -f vllm-gpu0

# GPU 1 logs
docker logs -f vllm-gpu1

# Both simultaneously
docker logs -f vllm-gpu0 & docker logs -f vllm-gpu1

# Filter for errors
docker logs vllm-gpu0 2>&1 | grep -i error
docker logs vllm-gpu1 2>&1 | grep -i error
```

---

### **Health Monitoring**

Create a health check script:

```bash
#!/bin/bash
# health-check.sh

echo "Checking vLLM GPU 0..."
curl -s http://localhost:8001/health || echo "GPU 0 FAILED"

echo "Checking vLLM GPU 1..."
curl -s http://localhost:8002/health || echo "GPU 1 FAILED"

echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
```

Run periodically:
```bash
watch -n 30 bash health-check.sh
```

---

## Troubleshooting

### **Issue 1: Only One GPU Running**

**Symptoms:**
```bash
docker ps
# Only shows vllm-gpu0 or vllm-gpu1
```

**Solution:**
```bash
# Check which failed
docker compose -f docker-compose.dual-vllm.yml ps

# Check logs
docker logs vllm-gpu0
docker logs vllm-gpu1

# Restart failed instance
docker compose -f docker-compose.dual-vllm.yml up -d vllm-gpu0
docker compose -f docker-compose.dual-vllm.yml up -d vllm-gpu1
```

---

### **Issue 2: One GPU Out of Memory**

**Symptoms:**
```
CUDA out of memory on GPU 0
GPU 1 is fine
```

**Cause:** Imbalanced load or one GPU has other process

**Solution:**
```bash
# Check what's using GPU 0
nvidia-smi

# Kill other processes if needed
# Or reduce memory allocation for that instance
nano docker-compose.dual-vllm.yml
# Change --gpu-memory-utilization 0.85 to 0.75 for that GPU
```

---

### **Issue 3: Load Not Balancing**

**Symptoms:** All requests go to one GPU

**Check:**
```python
# In Python, check client status
client.get_host_status()
# Should show both hosts as healthy
```

**Fix:**
```python
# Verify multiple hosts configured
print(client.hosts)
# Should show: ['http://vllm-gpu0:8000', 'http://vllm-gpu1:8000']

# Test health check
client.health_check()
```

---

### **Issue 4: Model Downloaded Twice**

**Symptoms:** Each GPU downloading model separately (2x storage)

**Cause:** Volume not shared correctly

**Fix:** Both instances should use same volume:
```yaml
volumes:
  - vllm-models:/root/.cache/huggingface  # Same volume!
```

Check:
```bash
docker volume ls | grep vllm
# Should show only ONE volume: vllm_models
```

---

## Advanced Configuration

### **Different Models per GPU**

Run different models on each GPU:

```yaml
# GPU 0: Large model for complex queries
vllm-gpu0:
  command: >
    --model ibm-granite/granite-3.1-8b-instruct
    --max-model-len 16384

# GPU 1: Smaller model for fast queries
vllm-gpu1:
  command: >
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct
    --max-model-len 8192
```

Then route queries based on complexity.

---

### **Resource Limits**

Limit CPU/RAM per instance:

```yaml
vllm-gpu0:
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 16G
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0']
            capabilities: [gpu]
```

---

## Production Deployment

### **1. Test First**

```bash
# Test dual setup locally
docker compose -f docker-compose.dual-vllm.yml up -d

# Run load test
for i in {1..10}; do
  curl http://localhost:8001/v1/completions -d '...' &
  curl http://localhost:8002/v1/completions -d '...' &
done
wait

# Check both handled requests
docker logs vllm-gpu0 | grep -i "generated"
docker logs vllm-gpu1 | grep -i "generated"
```

---

### **2. Update Application**

Update your Python code to use load-balanced client (see Code Integration above).

---

### **3. Deploy**

```bash
# Build updated API with load-balancing client
docker compose -f docker-compose.dual-vllm.yml build adam-api

# Start full stack
docker compose -f docker-compose.dual-vllm.yml up -d

# Monitor startup
docker logs -f adam-api
```

---

### **4. Verify End-to-End**

```bash
# Test RAG query (should use load balancer)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the PTO policy?",
    "conversation_id": "test-001"
  }'

# Check which GPU handled it
docker logs vllm-gpu0 | tail -20
docker logs vllm-gpu1 | tail -20
```

---

## Rollback Plan

If dual-GPU setup has issues:

```bash
# Stop dual setup
docker compose -f docker-compose.dual-vllm.yml down

# Revert to single GPU
docker compose -f docker-compose.vllm.yml up -d vllm

# Or revert to Ollama
docker compose -f docker-compose.ollama.yml up -d ollama
```

---

## Performance Tuning

### **For Maximum Throughput:**

```yaml
command: >
  --model ibm-granite/granite-3.1-8b-instruct
  --dtype float16
  --max-model-len 4096           # Smaller context = more throughput
  --gpu-memory-utilization 0.95  # Use more GPU memory
  --swap-space 2                 # Less CPU fallback
```

### **For Maximum Quality:**

```yaml
command: >
  --model ibm-granite/granite-3.1-8b-instruct
  --dtype float16
  --max-model-len 16384          # Larger context
  --gpu-memory-utilization 0.80  # Conservative
  --swap-space 8                 # More CPU fallback
```

### **For Balanced (Recommended):**

```yaml
command: >
  --model ibm-granite/granite-3.1-8b-instruct
  --dtype float16
  --max-model-len 8192
  --gpu-memory-utilization 0.85
  --swap-space 4
```

---

## Summary

**Your Setup:**
- 2x Quadro RTX 5000 (16GB each)
- 2x vLLM instances (independent)
- 1x Load balancer (in API)

**Expected Performance:**
- Single user: **10-15s** (vs 45s with Ollama)
- 2 concurrent users: **10-15s each** (parallel)
- 3-5 concurrent users: **20-30s** (minimal queuing)

**Next Steps:**
1. Deploy dual vLLM: `docker compose -f docker-compose.dual-vllm.yml up -d`
2. Verify both GPUs working: `curl http://localhost:8001/health && curl http://localhost:8002/health`
3. Update code to use `vllm_client_lb.py`
4. Test and enjoy the performance! 🚀

---

For issues, run diagnostics:
```bash
bash diagnose-gpu-memory.sh
bash diagnose-vllm-errors.sh
```
