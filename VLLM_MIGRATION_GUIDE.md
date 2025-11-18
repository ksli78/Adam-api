# vLLM Migration Guide

Complete guide for migrating from Ollama to vLLM on Quadro RTX 5000 GPUs.

**Expected Performance Improvement:**
- Single RAG query: **45s → 13-18s** (2.5-3.5x faster)
- Concurrent queries (3-5 users): **Much better batching, minimal slowdown**
- SQL queries: Similar or better performance

---

## Prerequisites

- ✅ Docker with GPU support (nvidia-docker)
- ✅ 2x NVIDIA Quadro RTX 5000 GPUs (32GB total VRAM)
- ✅ ~30GB free disk space (for model download)
- ✅ Internet connection (first time model download)

---

## Migration Steps

### **Phase 1: Setup vLLM (Parallel to Ollama)**

Run vLLM alongside Ollama to test before fully switching.

#### **1.1 Start vLLM Server**

```bash
# Start vLLM container
docker-compose -f docker-compose.vllm.yml up -d vllm

# Monitor first startup (model download - takes 10-30 minutes)
docker logs -f vllm-server
```

**What to expect:**
```
INFO:     Downloading mistralai/Mistral-Small-Instruct-2409...
INFO:     Loading model weights...
INFO:     Initializing engine with tensor parallelism=2
INFO:     GPU 0: NVIDIA Quadro RTX 5000
INFO:     GPU 1: NVIDIA Quadro RTX 5000
INFO:     Available memory: GPU 0: 14.4 GiB, GPU 1: 14.4 GiB
INFO:     Loaded model successfully!
INFO:     Started server process
```

Wait for: **`Application startup complete`**

#### **1.2 Test vLLM Server**

```bash
# Test health endpoint
curl http://localhost:8001/health

# Test text generation
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-Small-Instruct-2409",
    "prompt": "What is 2+2?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Expected response:**
```json
{
  "choices": [
    {
      "text": " 2+2 equals 4.",
      "finish_reason": "stop"
    }
  ]
}
```

✅ **If this works, vLLM is ready!**

---

### **Phase 2: Update Code to Use vLLM**

#### **2.1 Update Python Files**

You need to update **3 files** to replace Ollama client with vLLM client:

##### **File 1: `airgapped_rag_advanced.py`**

**Find (around line 66-67):**
```python
import ollama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://adam.amentumspacemissions.com:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small:22b")
```

**Replace with:**
```python
from vllm_client import VLLMClient

VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:8001")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-Small-Instruct-2409")
```

**Find (around line 127):**
```python
self.ollama_client = ollama.Client(host=OLLAMA_HOST)
```

**Replace with:**
```python
self.ollama_client = VLLMClient(host=VLLM_HOST, timeout=300)
```

##### **File 2: `sql_query_handler.py`**

**Find (around line 87-88):**
```python
import ollama

def __init__(
    self,
    database_name: str,
    config_path: str = "config/databases.yaml",
    ollama_host: str = "http://localhost:11434",
    model_name: str = "llama3:8b"
):
```

**Replace with:**
```python
from vllm_client import VLLMClient

def __init__(
    self,
    database_name: str,
    config_path: str = "config/databases.yaml",
    ollama_host: str = "http://localhost:8001",  # Now vLLM endpoint
    model_name: str = "mistralai/Mistral-Small-Instruct-2409"
):
```

**Find (around line 101):**
```python
self.ollama_client = ollama.Client(host=ollama_host)
```

**Replace with:**
```python
self.ollama_client = VLLMClient(host=ollama_host, timeout=300)
```

##### **File 3: `query_classifier.py`**

**Find (around line 70-71):**
```python
import ollama

def __init__(
    self,
    ollama_host: str = "http://localhost:11434",
    model_name: str = "llama3:8b"
):
```

**Replace with:**
```python
from vllm_client import VLLMClient

def __init__(
    self,
    ollama_host: str = "http://localhost:8001",  # Now vLLM endpoint
    model_name: str = "mistralai/Mistral-Small-Instruct-2409"
):
```

**Find (around line 80):**
```python
self.ollama_client = ollama.Client(host=ollama_host)
```

**Replace with:**
```python
self.ollama_client = VLLMClient(host=ollama_host, timeout=300)
```

##### **File 4: `metadata_extractor.py`** (if using Advanced RAG)

**Find:**
```python
import ollama
```

**Replace with:**
```python
from vllm_client import VLLMClient
```

**Find:**
```python
self.client = ollama.Client(host=ollama_host)
```

**Replace with:**
```python
self.client = VLLMClient(host=ollama_host, timeout=300)
```

#### **2.2 Update sql_routes.py**

**Find (around line 24):**
```python
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
```

**Replace with:**
```python
VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:8001")
```

**Find (around line 77 and 410) - Update parameter:**
```python
ollama_host=OLLAMA_HOST,
```

**Replace with:**
```python
ollama_host=VLLM_HOST,
```

---

### **Phase 3: Update Configuration**

#### **3.1 Update Environment Variables**

Create or update your `.env` file:

```bash
# Copy example configuration
cp .env.vllm.example .env

# Edit as needed
nano .env
```

**Required variables:**
```bash
LLM_BACKEND=vllm
VLLM_HOST=http://vllm:8000  # Inside Docker network
LLM_MODEL=mistralai/Mistral-Small-Instruct-2409
LLM_CONTEXT_WINDOW=16384
```

#### **3.2 Update docker-compose.yml (Optional)**

If you want to use vLLM as default, update `docker-compose.yml`:

**Replace:**
```yaml
environment:
  - OLLAMA_HOST=http://ollama:11434
```

**With:**
```yaml
environment:
  - VLLM_HOST=http://vllm:8000
```

---

### **Phase 4: Testing**

#### **4.1 Rebuild API Container**

```bash
# Rebuild with vLLM client
docker-compose -f docker-compose.vllm.yml build adam-api

# Start full stack
docker-compose -f docker-compose.vllm.yml up -d

# Check logs
docker logs -f adam-api
```

**Look for:**
```
INFO - VLLMClient initialized: host=http://vllm:8000, timeout=300s
INFO - Advanced RAG Pipeline initialized successfully!
```

#### **4.2 Test RAG Query**

```bash
# Test document query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the PTO policy?",
    "conversation_id": "test-001"
  }'
```

**Measure response time:**
```bash
time curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the PTO policy?", "conversation_id": "test-001"}'
```

**Expected timing:**
- **Ollama:** ~45+ seconds
- **vLLM:** ~13-18 seconds ✅

#### **4.3 Test Streaming**

Open `streaming_demo.html` in browser and test:
1. Document query: "What is the PTO policy?"
2. SQL query: "How many employees work here?"
3. Concurrent queries (open 3-5 tabs, submit simultaneously)

**Expected behavior:**
- Tokens stream smoothly
- Concurrent queries don't slow down significantly
- No timeouts

#### **4.4 Load Test (3-5 Concurrent Users)**

```bash
# Install Apache Bench (if not installed)
sudo apt-get install apache2-utils

# Simulate 5 concurrent users, 10 requests
ab -n 10 -c 5 -p query.json -T "application/json" \
  http://localhost:8000/query
```

**Create `query.json`:**
```json
{"question": "What is the PTO policy?", "conversation_id": "load-test"}
```

**Expected results:**
- vLLM: Most requests complete in 15-20 seconds
- Ollama: Requests queue up, 60-120+ seconds

---

### **Phase 5: Production Deployment**

#### **5.1 Update Production Environment**

```bash
# SSH to production server
ssh user@adam.amentumspacemissions.com

# Pull latest code
cd /path/to/Adam-api
git pull

# Start vLLM (keeps Ollama running as backup)
docker-compose -f docker-compose.vllm.yml up -d vllm

# Wait for model download and initialization (monitor logs)
docker logs -f vllm-server

# Update API container
docker-compose -f docker-compose.vllm.yml up -d adam-api

# Monitor for errors
docker logs -f adam-api
```

#### **5.2 Gradual Rollout (Recommended)**

**Option 1: Blue-Green Deployment**
1. Keep Ollama running on port 11434
2. Run vLLM on port 8001
3. Run new API version on port 8002 (using vLLM)
4. Test thoroughly
5. Switch load balancer when confident
6. Stop Ollama after 1 week

**Option 2: Feature Flag**
Add environment variable to switch backends:
```python
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # or "vllm"

if LLM_BACKEND == "vllm":
    from vllm_client import VLLMClient as LLMClient
    LLM_HOST = os.getenv("VLLM_HOST")
else:
    import ollama
    LLMClient = ollama.Client
    LLM_HOST = os.getenv("OLLAMA_HOST")
```

---

## Troubleshooting

### **Issue 1: vLLM container won't start**

**Symptom:**
```
RuntimeError: No GPU available
```

**Solution:**
```bash
# Verify GPUs are visible
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check GPU driver
nvidia-smi

# Reinstall nvidia-docker if needed
```

---

### **Issue 2: Model download fails**

**Symptom:**
```
OSError: HuggingFace Hub is unreachable
```

**Solution:**
```bash
# Pre-download model on host
pip install huggingface-hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mistral-Small-Instruct-2409')"

# Mount to vLLM container
docker-compose -f docker-compose.vllm.yml down
# Edit docker-compose.vllm.yml to mount ~/.cache/huggingface
docker-compose -f docker-compose.vllm.yml up -d
```

---

### **Issue 3: Out of memory (OOM)**

**Symptom:**
```
CUDA out of memory
```

**Solution:**
```bash
# Reduce GPU memory utilization
# Edit docker-compose.vllm.yml:
--gpu-memory-utilization 0.85  # Reduce from 0.90

# Or switch to smaller model:
--model mistralai/Mistral-7B-Instruct-v0.3
```

---

### **Issue 4: Slower than expected**

**Checklist:**
- ✅ Both GPUs are being used? Check `nvidia-smi`
- ✅ `--tensor-parallel-size 2` is set?
- ✅ Model is fully loaded (not swapping to CPU)?
- ✅ Using FP16? (Turing doesn't support FP8)

**Benchmark:**
```bash
# Test first request (model loading)
time curl -X POST http://localhost:8001/v1/completions \
  -d '{"model": "mistralai/Mistral-Small-Instruct-2409", "prompt": "test", "max_tokens": 100}'

# Test second request (should be faster)
time curl -X POST http://localhost:8001/v1/completions \
  -d '{"model": "mistralai/Mistral-Small-Instruct-2409", "prompt": "test", "max_tokens": 100}'
```

---

### **Issue 5: Streaming not working**

**Check:**
1. vLLM returns SSE format (different from Ollama)
2. vllm_client.py correctly parses SSE
3. FastAPI StreamingResponse configured correctly

**Debug:**
```python
# Add logging in vllm_client.py
logger.info(f"Received streaming line: {line_str}")
```

---

## Rollback Plan

If vLLM has issues, rollback to Ollama:

```bash
# Stop vLLM
docker-compose -f docker-compose.vllm.yml down

# Revert code changes
git checkout HEAD -- airgapped_rag_advanced.py sql_query_handler.py query_classifier.py sql_routes.py

# Restart Ollama
docker-compose -f docker-compose.ollama.yml up -d
```

**Keep Ollama running for 1-2 weeks** as backup during vLLM testing.

---

## Performance Monitoring

### **Key Metrics to Track**

1. **Response Time (p50, p95, p99)**
   ```bash
   # Add to API
   import time
   start = time.time()
   # ... LLM call ...
   duration = time.time() - start
   logger.info(f"LLM response time: {duration:.2f}s")
   ```

2. **Concurrent Request Handling**
   - Ollama: Queues requests sequentially
   - vLLM: Batches efficiently

3. **GPU Utilization**
   ```bash
   watch -n 1 nvidia-smi
   ```
   - Both GPUs should show 80-95% utilization
   - Memory should be ~14-15GB per GPU

4. **Tokens per Second**
   - vLLM logs this: `"tokens/s": 82.5`
   - Target: 60-100 tokens/s for Mistral-Small on RTX 5000

---

## Alternative Model Options

If Mistral-Small is still too slow or doesn't fit:

### **Faster Options:**

1. **Mistral-7B-Instruct** (Faster, single GPU)
   ```yaml
   command: >
     --model mistralai/Mistral-7B-Instruct-v0.3
     --tensor-parallel-size 1  # Single GPU
     --gpu-memory-utilization 0.90
   ```
   - Speed: ~2x faster than Mistral-Small
   - Quality: Slightly lower, but excellent for RAG

2. **Llama-3.1-8B** (Similar to Mistral-7B)
   ```yaml
   --model meta-llama/Llama-3.1-8B-Instruct
   ```

### **Higher Quality Options:**

If you need better quality and can wait longer:

1. **Mistral-Nemo-12B** (Good balance)
   ```yaml
   --model mistralai/Mistral-Nemo-Instruct-2407
   --tensor-parallel-size 2
   ```

---

## Support & Resources

- **vLLM Documentation:** https://docs.vllm.ai/
- **vLLM GitHub:** https://github.com/vllm-project/vllm
- **OpenAI API Spec:** https://platform.openai.com/docs/api-reference
- **Adam RAG Issues:** (Your internal issue tracker)

---

## Summary

**Total Migration Time:** ~5-7 hours
- Setup vLLM: 1-2 hours (including model download)
- Code changes: 2-3 hours
- Testing: 2 hours

**Expected Results:**
- ✅ 2.5-3.5x faster RAG queries (45s → 13-18s)
- ✅ Better concurrent user handling (3-5 users)
- ✅ More consistent latency
- ✅ Production-ready inference

**Next Steps:**
1. Start with Phase 1 (test vLLM alongside Ollama)
2. Validate performance meets expectations
3. Proceed with full migration
4. Monitor for 1 week before decommissioning Ollama

Good luck with the migration! 🚀
