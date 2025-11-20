# Dual-GPU Ollama Diagnostics

## Current Situation

### Hardware Setup
- **Server**: 2x GPUs with 16GB VRAM each
- **Model**: mistral-small:22b
- **Memory usage**: ~14GB per GPU (2GB free)
- **Ollama instances**:
  - Instance 1: http://adam.amentumspacemissions.com:11434
  - Instance 2: http://adam.amentumspacemissions.com:11435

### Performance Issues
```
TTFT (1st query):  12.598s
TTFT (2nd query):  12.001s  ← Should be <1s if model stayed loaded!
Token generation:  6.8 tokens/second
Expected (Q4):     15-25 tokens/second
Expected (Q5):     10-18 tokens/second
```

## Diagnostic Commands

Run these on your server to diagnose the issues:

### 1. Check Which GPU Each Ollama Instance Is Using

```bash
# Watch GPU usage in real-time
nvtop

# Or use nvidia-smi
watch -n 1 nvidia-smi
```

**What to look for:**
- When you send a query, does only ONE GPU spike to 100%?
- Or do BOTH GPUs show activity?
- Are both Ollama instances using different GPUs?

### 2. Check Running Models on Each Instance

```bash
# Check instance 1
curl http://adam.amentumspacemissions.com:11434/api/ps

# Check instance 2
curl http://adam.amentumspacemissions.com:11435/api/ps
```

**What to look for:**
```json
{
  "models": [
    {
      "name": "mistral-small:22b",
      "size": 13421772800,  // Size in bytes
      "digest": "...",
      "expires_at": "2025-11-20T10:51:42.911Z"  // Should be ~10 min from last use
    }
  ]
}
```

**Critical questions:**
1. Is the model loaded on BOTH instances or just one?
2. Does the `expires_at` update after warmup?
3. Does it show 10 minutes in the future?

### 3. Check Model Quantization Level

```bash
# Get detailed model info from instance 1
curl http://adam.amentumspacemissions.com:11434/api/show -d '{
  "name": "mistral-small:22b"
}'

# Check instance 2
curl http://adam.amentumspacemissions.com:11435/api/show -d '{
  "name": "mistral-small:22b"
}'
```

**What to look for:**
```json
{
  "modelfile": "...",
  "parameters": "...",
  "template": "...",
  "details": {
    "format": "gguf",
    "family": "mistral",
    "parameter_size": "22B",
    "quantization_level": "Q4_K_M"  ← Look for this!
  }
}
```

**Quantization levels:**
- Q3_K_M: ~8-9GB, 6-10 tok/s (your speed matches this!)
- Q4_K_M: ~11-13GB, 15-25 tok/s ✅
- Q5_K_M: ~14-16GB, 10-18 tok/s ✅
- Q6_K: ~17-19GB, 8-15 tok/s

### 4. Check Docker GPU Assignment

If Ollama instances are in Docker:

```bash
# List running containers
docker ps | grep ollama

# Check GPU assignment for each container
docker inspect ollama-gpu0 | grep -A 10 "DeviceRequests"
docker inspect ollama-gpu1 | grep -A 10 "DeviceRequests"
```

**What to look for:**
```json
"DeviceRequests": [
  {
    "Driver": "nvidia",
    "DeviceIDs": ["0"],  ← GPU 0
    "Capabilities": [["gpu"]]
  }
]
```

Instance 1 should have `"DeviceIDs": ["0"]`
Instance 2 should have `"DeviceIDs": ["1"]`

### 5. Test keep_alive Directly

```bash
# Send test request to instance 1 with keep_alive
curl http://adam.amentumspacemissions.com:11434/api/generate -d '{
  "model": "mistral-small:22b",
  "prompt": "Hello",
  "stream": false,
  "keep_alive": "10m"
}'

# Immediately check if model is still loaded
curl http://adam.amentumspacemissions.com:11434/api/ps

# Wait 2 minutes and check again
sleep 120
curl http://adam.amentumspacemissions.com:11434/api/ps
```

**What to look for:**
- Does `api/ps` show the model loaded?
- Does `expires_at` show ~10 minutes in future?
- Does it stay loaded after 2 minutes?

### 6. Check Ollama Version

```bash
curl http://adam.amentumspacemissions.com:11434/api/version
curl http://adam.amentumspacemissions.com:11435/api/version
```

**Keep_alive parameter support:**
- Ollama < 0.1.15: No keep_alive support
- Ollama >= 0.1.15: Supports keep_alive ✅

## Expected Findings & Solutions

### Finding 1: Both Ollama Instances on Same GPU
**If nvtop shows only GPU 0 spiking during queries:**

**Solution:**
```bash
# Fix Docker GPU assignment
# In docker-compose.yml or docker run commands:

# Container 1 (port 11434)
docker run -d --gpus device=0 ...

# Container 2 (port 11435)
docker run -d --gpus device=1 ...
```

### Finding 2: Model Only Loaded on One Instance
**If `api/ps` shows model on instance 1 but not instance 2:**

**Solution:**
The warmup code I added should fix this! When you restart:
```
Warming instance 1/2: http://adam.amentumspacemissions.com:11434
  ✅ Instance 1 ready
Warming instance 2/2: http://adam.amentumspacemissions.com:11435
  ✅ Instance 2 ready  ← Both should show ready
```

### Finding 3: Model is Q3 Quantization
**If model info shows Q3_K_M:**

**Solution:**
```bash
# Pull better quantization
ollama pull mistral-small:22b-instruct-q4_k_m
# or
ollama pull mistral-small:22b-instruct-q5_k_m

# Then update your config
LLM_MODEL=mistral-small:22b-instruct-q4_k_m
```

**Expected improvement:**
- Token generation: 15-25 tok/s (vs current 6.8)
- Total time: ~35-40s (vs current 93s)
- **2.5x faster!**

### Finding 4: keep_alive Not Working
**If `expires_at` doesn't update or model unloads quickly:**

**Solutions:**

A. **Update Ollama** (if version < 0.1.15):
```bash
docker pull ollama/ollama:latest
# Restart containers
```

B. **Set global keep_alive**:
```bash
# In docker-compose.yml or docker run:
environment:
  - OLLAMA_KEEP_ALIVE=10m
```

C. **Use Modelfile** (if parameter not working):
```bash
# Create Modelfile
FROM mistral-small:22b
PARAMETER keep_alive 10m

# Create custom model
ollama create mistral-small-keep-alive -f Modelfile

# Update config
LLM_MODEL=mistral-small-keep-alive
```

### Finding 5: Large Context Window Impact
**Your config: LLM_CONTEXT_WINDOW=16384**

With mistral-small:22b and 16K context:
- Prompt processing: 5-10s (explains some of the 12s!)
- Memory usage: Higher
- Token generation: Slower

**Solution - Test with smaller context:**
```python
LLM_CONTEXT_WINDOW=8192  # Test with 8K
```

**Expected improvement:**
- Prompt processing: 2-5s
- TTFT: 8-10s (vs 12s)
- More VRAM for model weights

## Quick Test Sequence

Run these commands in order:

```bash
# 1. Check current state
echo "=== Checking running models ==="
curl http://adam.amentumspacemissions.com:11434/api/ps
curl http://adam.amentumspacemissions.com:11435/api/ps

# 2. Check quantization
echo "=== Checking model quantization ==="
curl http://adam.amentumspacemissions.com:11434/api/show -d '{"name":"mistral-small:22b"}' | grep -i quant

# 3. Send test query to instance 1 with keep_alive
echo "=== Testing keep_alive on instance 1 ==="
time curl http://adam.amentumspacemissions.com:11434/api/generate -d '{
  "model": "mistral-small:22b",
  "prompt": "Say hello",
  "stream": false,
  "keep_alive": "10m"
}'

# 4. Send test query to instance 2 with keep_alive
echo "=== Testing keep_alive on instance 2 ==="
time curl http://adam.amentumspacemissions.com:11435/api/generate -d '{
  "model": "mistral-small:22b",
  "prompt": "Say hello",
  "stream": false,
  "keep_alive": "10m"
}'

# 5. Check if both models stayed loaded
sleep 5
echo "=== Checking if models stayed loaded ==="
curl http://adam.amentumspacemissions.com:11434/api/ps
curl http://adam.amentumspacemissions.com:11435/api/ps

# 6. Watch GPU usage during a real query
echo "=== Watch GPU usage - run a query from your app now! ==="
watch -n 1 nvidia-smi
```

## Share Your Results

Please run the commands above and share:

1. **Model quantization level**: Q3? Q4? Q5?
2. **GPU assignment**: Are instances on different GPUs?
3. **keep_alive status**: Do models stay loaded?
4. **Ollama version**: Supporting keep_alive parameter?
5. **nvtop during query**: Which GPU(s) spike to 100%?

This will tell us exactly what to fix!
