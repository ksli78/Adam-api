# Model Optimization - Fix Slow Token Generation

## Current Performance Analysis

From your logs:

### ✅ What's Working Well
- **Model warmup**: Loads at startup (19s)
- **RAG retrieval**: 0.166-0.421s (excellent!)
- **GPU detected**: NVIDIA RTX 3500 Ada Laptop (8GB VRAM)

### 🚨 Performance Issues

```
Model: mistral-small:22b
GPU: NVIDIA RTX 3500 Ada (8GB VRAM)

First query TTFT:  12.598s
Second query TTFT: 12.001s  (should be <1s if model stayed loaded!)
Token generation:  6.8 tokens/second (should be 30-50+)
```

## Root Cause: Model Too Large for GPU

**mistral-small:22b** = 22 BILLION parameters

### Memory Requirements:
- **FP16 (full precision)**: ~44GB VRAM
- **Q4 quantization**: ~11GB VRAM
- **Q3 quantization**: ~8GB VRAM (barely fits)
- **Your GPU**: 8GB VRAM

**The model is too large!** It's likely:
1. Heavily quantized (Q3) to squeeze into 8GB → slow inference
2. Spilling to system RAM → very slow swapping
3. Partially running on CPU → extremely slow

This explains:
- **Slow token generation**: 6.8 tok/s (Q3 quantization + memory pressure)
- **12s "TTFT" on both queries**: Model unloading and reloading due to memory pressure

## Solution 1: Switch to Mistral 7B (RECOMMENDED) ⚡

**Best option for your 8GB GPU!**

### How to Switch:

**Option A: Set environment variable** (if using batch file or startup script)
```batch
set LLM_MODEL=mistral
```

**Option B: Update Python code** (if not using env var)

Edit the file that starts your app or sets the environment variable to:
```python
LLM_MODEL=mistral
# or more specific:
LLM_MODEL=mistral:7b-instruct-v0.3
```

### Expected Performance with Mistral 7B:

```
Memory usage:      4-5GB VRAM (comfortably fits!)
First query TTFT:  1-2s (with warmup)
Warm query TTFT:   0.5-1s ✅
Token generation:  35-50 tokens/second ✅
Total time:        ~15 seconds for 544 tokens (vs current 93s)
Quality:           Excellent for RAG (actually better for concise answers)
```

**Your users would see:**
- Retrieval: 0.4s
- First token: 1s
- Full answer: 15s total vs current 93s
- **6x faster!** 🚀

## Solution 2: Optimize mistral-small:22b (If You Must Keep It)

If you really need the 22B model:

### A. Check Current Quantization

```bash
# On your Ollama server
curl http://adam.amentumspacemissions.com:11434/api/show -d '{"name":"mistral-small:22b"}'
```

Look for the quantization level (Q3, Q4, Q5, etc.)

### B. Try Q4 or Q5 (Better Quality)

```bash
# Pull better quantization
ollama pull mistral-small:22b-instruct-q4_k_m
# or
ollama pull mistral-small:22b-instruct-q5_k_m
```

Then update:
```
LLM_MODEL=mistral-small:22b-instruct-q4_k_m
```

**Expected improvement:**
- Token generation: 10-15 tok/s (better than 6.8, but still slow)
- TTFT: Still 10-12s due to model size
- Total time: ~40-50s for 544 tokens

### C. Reduce Context Window

Smaller context = less memory pressure:

```python
LLM_CONTEXT_WINDOW=8192  # Reduce from 16384
```

## Solution 3: Fix Load Balancer Keep-Alive Issue

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

## My Strong Recommendation

**Switch to Mistral 7B immediately:**

1. **6x faster responses** (15s vs 93s)
2. **Better user experience** (<1s TTFT warm queries)
3. **More reliable** (no memory pressure, no swapping)
4. **Better quality** for RAG (22B often over-explains, 7B is more concise)

For RAG tasks, model size matters less than you think. The quality comes from:
1. **Good retrieval** (you have this! ✅)
2. **Good prompting** (you have this! ✅)
3. **Fast iteration** (currently blocked by slow model)

A fast, responsive 7B model provides better UX than a slow 22B model for this use case.

## Quick Win Test

Try this right now to see the difference:

```bash
# In your startup script or environment
set LLM_MODEL=mistral:7b-instruct-v0.3

# Restart your API
# Run a test query
```

You should immediately see:
- TTFT: ~1s (vs 12s)
- Token generation: 35-50 tok/s (vs 6.8)
- Total time: ~15s (vs 93s)

## Summary

| Metric | Current (22B) | With Mistral 7B | Improvement |
|--------|---------------|-----------------|-------------|
| TTFT (warm) | 12s | 1s | **12x faster** ✅ |
| Token rate | 6.8 tok/s | 40 tok/s | **6x faster** ✅ |
| Total time | 93s | 15s | **6x faster** ✅ |
| Memory | 8GB (maxed) | 5GB | **More headroom** ✅ |
| Quality | Excellent | Excellent | **Same** ✅ |

**Just change `LLM_MODEL=mistral` and restart. That's it!**
