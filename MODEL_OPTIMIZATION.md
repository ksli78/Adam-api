# Model Optimization - Fix Slow Token Generation

## Current Performance Analysis

From your logs:

### ✅ What's Working Well
- **Model warmup**: Loads at startup (19s)
- **RAG retrieval**: 0.166-0.421s (excellent!)
- **Embedding GPU**: NVIDIA RTX 3500 Ada Laptop (8GB VRAM) - fast! ✅
- **LLM GPUs**: 2x 16GB VRAM GPUs - model fits! ✅

### 🚨 Performance Issues

```
Model: mistral-small:22b
Hardware: 2x 16GB VRAM GPUs (14GB used, 2GB free each)
Model fits comfortably in VRAM ✅

But still slow:
First query TTFT:  12.598s
Second query TTFT: 12.001s  (should be <1s if model stayed loaded!)
Token generation:  6.8 tokens/second

Expected for Q4 on 16GB GPU: 15-25 tok/s
Expected for Q5 on 16GB GPU: 10-18 tok/s
```

## Root Causes: Configuration Issues

Since the model fits in VRAM (14GB used on 16GB GPU), the issues are:

### 1. Model Not Staying Loaded (12s TTFT on both queries)
The second query took 12s TTFT even within 10 minutes → `keep_alive` not working

**Possible causes:**
- Load balancer switching to instance without model loaded
- `keep_alive` parameter not supported by Ollama version
- Both Ollama instances running on same GPU
- Model being unloaded for other reasons

### 2. Slow Token Generation (6.8 tok/s)
For mistral-small:22b on 16GB GPU:
- Q3 quantization: 6-10 tok/s ← **Your speed matches this!**
- Q4 quantization: 15-25 tok/s
- Q5 quantization: 10-18 tok/s

**Likely cause:** Model is Q3 quantized (over-aggressive for 16GB GPU)

## Solution 1: Fix Dual-GPU Configuration (RECOMMENDED) ⚡

**Your hardware can handle mistral-small:22b - let's fix the config!**

### Run Diagnostics First:

See **DUAL_GPU_DIAGNOSTICS.md** for detailed diagnostic commands.

Quick check:
```bash
# Check model quantization
curl http://adam.amentumspacemissions.com:11434/api/show -d '{"name":"mistral-small:22b"}' | grep quant

# Check if models stay loaded
curl http://adam.amentumspacemissions.com:11434/api/ps
curl http://adam.amentumspacemissions.com:11435/api/ps
```

### A. Upgrade to Q4 or Q5 Quantization

If model is Q3, upgrade to Q4 or Q5:

```bash
# Pull Q4 (fastest, 15-25 tok/s)
ollama pull mistral-small:22b-instruct-q4_k_m

# Or pull Q5 (better quality, 10-18 tok/s)
ollama pull mistral-small:22b-instruct-q5_k_m
```

Update your config:
```bash
LLM_MODEL=mistral-small:22b-instruct-q4_k_m
```

**Expected improvement:**
- Token generation: 15-25 tok/s (vs 6.8)
- Total time: ~35s (vs 93s)
- **2.5x faster!**

### B. Verify Both Instances Warm Up

The code update should now warm both instances. Watch for:
```
✅ All 2 Ollama instances warmed up and ready!
```

If not, check Docker GPU assignment.

### C. Reduce Context Window

Your current setting of 16K is expensive for prompt processing:

```python
LLM_CONTEXT_WINDOW=8192  # Test with 8K instead of 16K
```

**Expected improvement:**
- Prompt processing: 2-5s (vs 5-10s)
- TTFT: 7-10s (vs 12s)

## Solution 2: Switch to Mistral 7B (Alternative)

**If configuration fixes don't work or you want maximum speed:**

### How to Switch:

**Set environment variable or update config:**
```bash
LLM_MODEL=mistral:7b-instruct-v0.3
```

### Expected Performance with Mistral 7B:

```
Memory usage:      4-5GB VRAM per GPU
First query TTFT:  0.5-1s (with warmup)
Warm query TTFT:   0.3-0.5s ✅
Token generation:  40-60 tokens/second ✅
Total time:        ~12 seconds for 544 tokens (vs current 93s)
Quality:           Excellent for RAG (more concise answers)
```

**Improvement:**
- **8x faster total time** (12s vs 93s)
- **Instant TTFT** on warm queries (<1s)
- **Room for more users** (lower VRAM usage)

## Solution 3: Keep mistral-small:22b But Optimize

**Only if the fixes in Solution 1 don't work:**

These are fallback options covered in Solution 1 above.

## Solution 4: Fix Load Balancer Issues

The 12s TTFT on the second query suggests the model is being unloaded or the load balancer is routing to an instance without the model loaded.

### Check Both Ollama Instances:

```bash
# Check first instance
curl http://adam.amentumspacemissions.com:11434/api/ps

# Check second instance
curl http://adam.amentumspacemissions.com:11435/api/ps
```

**Look for:**
- Is the model loaded on both instances?
- Is memory usage high?

### Warm Up BOTH Instances:

Update the warmup code in `airgapped_rag_advanced.py`:

```python
# Warm up ALL Ollama instances (not just one via load balancer)
logger.info(f"Warming up Ollama model '{LLM_MODEL}' on ALL instances...")
for i, host in enumerate(OLLAMA_HOSTS):
    try:
        single_client = OllamaClient(host=host)
        warmup_response = single_client.generate(
            model=LLM_MODEL,
            prompt="Hello",
            options={"temperature": 0.1, "num_predict": 5},
            stream=False,
            keep_alive="10m"
        )
        logger.info(f"  ✅ Instance {i+1} ({host}): Model loaded and ready")
    except Exception as e:
        logger.warning(f"  ⚠️  Instance {i+1} ({host}): Failed to warm up: {e}")
```

### Set Global Keep-Alive:

On your Ollama servers, set:
```bash
OLLAMA_KEEP_ALIVE=10m
```

## Solution 4: Compare with Employee Directory

Check if Employee Directory uses a different model or configuration:

```bash
# Search for Employee Directory model config
grep -r "employee" . --include="*.py" | grep -i model
```

If it uses a smaller/different model, that might explain why it's faster.

## My Recommendations

### Priority 1: Fix Current Setup (Keep mistral-small:22b)

Your hardware can handle it! Just need to fix configuration:

1. **Run diagnostics** (see DUAL_GPU_DIAGNOSTICS.md)
2. **Upgrade to Q4 quantization** if currently Q3
3. **Reduce context window** to 8K
4. **Verify both GPU instances** warm up correctly

**Expected improvement:**
- Token generation: 15-25 tok/s (vs 6.8) = **2.5x faster**
- Total time: ~35s (vs 93s) = **2.5x faster**
- TTFT (warm): 2-5s (vs 12s) if keep_alive works

### Priority 2: Consider Mistral 7B for Maximum Speed

If you want the absolute best user experience:

**Advantages:**
- **8x faster** (12s vs 93s total time)
- **Instant warm queries** (<1s TTFT)
- **More concurrent users** (lower VRAM per query)
- **Better for RAG** (more concise, less rambling)

**When to choose:**
- User experience is top priority
- You value speed over marginal quality differences
- You want to support more concurrent users

## Quick Diagnostic Check

Run this now to identify the issue:

```bash
# Check quantization level
curl http://adam.amentumspacemissions.com:11434/api/show -d '{"name":"mistral-small:22b"}' 2>/dev/null | grep -i quant

# Check if models are loaded
curl http://adam.amentumspacemissions.com:11434/api/ps 2>/dev/null
curl http://adam.amentumspacemissions.com:11435/api/ps 2>/dev/null
```

This will tell you immediately if the model is Q3 (needs upgrade) or if only one instance has the model loaded.

## Summary

| Solution | Token Rate | Total Time | TTFT (warm) | Effort |
|----------|-----------|------------|-------------|--------|
| **Current** | 6.8 tok/s | 93s | 12s | - |
| **Fix config + Q4** | 15-25 tok/s | 35s | 2-5s | Medium |
| **Mistral 7B** | 40-60 tok/s | 12s | <1s | Easy |

**Next step:** Run the diagnostic commands above to see what needs fixing!
