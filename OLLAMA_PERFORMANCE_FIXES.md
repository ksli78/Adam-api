# Ollama Performance Fixes - 51 Second Delay Before First Token

## Issue Identified

From your logs, the performance problem is **NOT the RAG retrieval** - it's the **Ollama LLM**:

### ✅ RAG Retrieval is Fast (Working Great!)
```
[TIMING] ⚠️  EMBEDDING GENERATION took 0.286s
[TIMING] ⚠️  CHROMADB QUERY took 0.048s
[TIMING] ⚠️  TOTAL BM25 took 0.007s
[TIMING] ⚠️⚠️⚠️  TOTAL TIME BEFORE LLM STREAMING: 0.398s
```

### 🚨 Ollama LLM is Slow (The Problem!)
```
LLM streaming started:  09:31:18.733
First token arrived:     09:32:10.427
DELAY: 51.7 seconds! 🚨🚨🚨

Token generation rate: ~7 tokens/second
Expected on GPU: 30-50+ tokens/second
```

## Root Causes

### 1. Model Loading Delay (51 seconds)
**The 51 second delay is Ollama loading the model from disk into GPU memory.**

Ollama unloads models after 5 minutes of inactivity by default. When your next query comes in, it has to:
1. Load model from disk (5-15 seconds for Mistral 7B)
2. Allocate GPU memory (5-10 seconds)
3. Process the prompt (variable, depends on length)

### 2. Slow Token Generation (7 tokens/sec)
After the model loads, generation is still slow (7 tokens/sec vs expected 30-50+):
- **Possible causes:**
  - Very large prompt (>4K tokens)
  - Both GPUs busy (load balancer selecting busy GPU)
  - Quantized model (Q4 or Q5 instead of FP16)
  - CPU inference (unlikely, would be 1-2 tokens/sec)

## Solutions

### Solution 1: Keep Model Loaded (DONE ✅)

I've added `keep_alive="10m"` to the Ollama generate call:

**File**: `airgapped_rag_advanced.py` line 677
```python
response = self.ollama_client.generate(
    model=LLM_MODEL,
    prompt=prompt,
    options={...},
    stream=True,
    keep_alive="10m"  # Keep model loaded for 10 minutes
)
```

**Impact**: After the first query, subsequent queries within 10 minutes will be instant (<1s to first token)

### Solution 2: Preload Model at Startup

Add a warmup request when the API starts to load the model into memory:

**File**: `airgapped_rag_advanced.py` - Add to `AdvancedRAGPipeline.__init__`:
```python
# In __init__ method, after initializing ollama_client:
logger.info("Warming up Ollama model (loading into GPU memory)...")
try:
    warmup_response = self.ollama_client.generate(
        model=LLM_MODEL,
        prompt="Hello",
        options={"temperature": 0.1, "num_predict": 5},
        stream=False,
        keep_alive="10m"  # Keep loaded
    )
    logger.info(f"✅ Ollama model warmed up and ready: {LLM_MODEL}")
except Exception as e:
    logger.warning(f"⚠️  Failed to warm up Ollama model: {e}")
```

**Impact**: First user query will be fast (no model loading delay)

### Solution 3: Configure Ollama keep_alive Globally

Set a global keep_alive in your Ollama configuration:

**Docker Compose** (if using Docker):
```yaml
ollama:
  environment:
    - OLLAMA_KEEP_ALIVE=10m
```

**Or via Modelfile**:
```bash
# Create a Modelfile
FROM mistral
PARAMETER keep_alive 10m

# Create the model
ollama create mistral-keep-alive -f Modelfile
```

Then update your `.env`:
```
LLM_MODEL=mistral-keep-alive
```

### Solution 4: Check Ollama GPU Configuration

Verify both Ollama instances are using GPUs:

```bash
# Check Ollama GPU status
curl http://adam.amentumspacemissions.com:11434/api/tags

# Check GPU memory usage
nvidia-smi

# Check both Ollama instances
curl http://adam.amentumspacemissions.com:11434/api/tags
curl http://adam.amentumspacemissions.com:11435/api/tags  # If you have a second instance
```

Look for model memory usage:
- Mistral 7B FP16: ~14GB
- Mistral 7B Q4: ~4GB
- Mistral 7B Q5: ~5GB

**If using Q4/Q5**: This explains the slower token generation. Consider using FP16 for better performance if you have VRAM.

### Solution 5: Reduce Prompt Size

Check if your prompts are excessively large:

The new logging will show:
```
[TIMING] Prompt size: XXXX characters (~YYYY words)
```

If prompts are >8K tokens (~32K characters), consider:
1. Reducing parent chunk size (currently 1500 chars)
2. Using fewer parent chunks (currently 5)
3. Summarizing document context

### Solution 6: Load Balance Check

Verify both Ollama instances are healthy and not overloaded:

```bash
# Check load balancer status
curl http://adam.amentumspacemissions.com:11434/api/ps  # Running models on GPU 0
curl http://adam.amentumspacemissions.com:11435/api/ps  # Running models on GPU 1
```

If one GPU is busy, the load balancer should route to the other.

## Diagnostic Logging Added

I've added comprehensive timing logs for the Ollama LLM:

### 1. Prompt Information
```
[TIMING] Prompt size: 4523 characters (~892 words)
[TIMING] LLM model: mistral, context window: 16384
```

### 2. Time to First Token (TTFT)
```
[TIMING] Calling Ollama at 1234567890.123...
[TIMING] ⚠️⚠️⚠️  TIME TO FIRST TOKEN (TTFT): 51.742s
```
This is the model loading + prompt processing time.

### 3. Token Generation Rate
```
[TIMING] ⚠️  TOTAL LLM GENERATION TIME: 125.375s
[TIMING] ⚠️  TOKEN GENERATION RATE: 7.2 tokens/second
[TIMING] Expected: 30-50+ tokens/sec on GPU, <10 tokens/sec on CPU
```

## Expected Performance After Fixes

### First Query (Cold Start)
- Model loading: 10-20 seconds (unavoidable)
- Prompt processing: 1-3 seconds
- **Total TTFT: 11-23 seconds** (acceptable for cold start)

### Subsequent Queries (Warm Model)
- Model loading: 0 seconds (already loaded)
- Prompt processing: 0.5-2 seconds
- **Total TTFT: 0.5-2 seconds** ✅

### Token Generation
- **Target: 30-50 tokens/second** on GPU with FP16 model
- Current: 7 tokens/second (needs investigation)

## Action Items

1. **✅ DONE**: Added `keep_alive="10m"` to prevent model unloading
2. **✅ DONE**: Added comprehensive timing diagnostics
3. **TODO**: Add model warmup at API startup (see Solution 2)
4. **TODO**: Restart API and test with new logging
5. **TODO**: Check token generation rate in new logs
6. **TODO**: Verify Ollama model type (FP16 vs Q4/Q5)
7. **TODO**: Check GPU memory with `nvidia-smi` during inference

## Next Steps

1. **Restart your API** to pick up the `keep_alive` change
2. **Run a test query** and check the new timing logs:
   - TTFT should be ~50s first time (model loading)
   - Token rate should show in logs
3. **Run a second query immediately** (within 10 min):
   - TTFT should be <2s (model already loaded!) ✅
4. **Share the new logs** to verify token generation rate

The `keep_alive` fix alone should dramatically improve the user experience after the first query!
