# Performance Diagnostics - RAG Query Streaming

## Issue Summary

The RAG Policy and Procedures stream is experiencing very long delays (>60 seconds) before tokens start appearing on screen, which is much slower than expected and slower than the Employee Directory stream.

## Expected vs Actual Performance

### Expected Behavior
- **Employee Directory**: 3-8 seconds to first token
- **RAG Policy**: 600ms-2.5 seconds to first token ✅ (SHOULD BE FASTER!)

### Actual Behavior (Reported)
- **Employee Directory**: "Much shorter" delay (a few seconds)
- **RAG Policy**: >60 seconds to first token ⚠️ (MUCH SLOWER!)

This is the **opposite** of what should happen!

## Likely Root Causes

### 1. Embedding Model Running on CPU (MOST LIKELY)
The e5-large-v2 embedding model is a large model (1.3GB) that requires GPU acceleration:
- **With GPU**: 100-500ms per query ✅
- **Without GPU (CPU)**: 10-60+ seconds per query ⚠️⚠️⚠️

**Location**: `parent_child_store.py` lines 421-426
```python
query_embedding = self.embedding_model.encode(
    query_text,
    convert_to_numpy=True
).tolist()
```

### 2. Large ChromaDB Collection
If the document collection is very large, ChromaDB queries can be slow:
- Small collections (<1000 chunks): <100ms
- Large collections (>10,000 chunks): 1-5 seconds

**Location**: `parent_child_store.py` lines 433-440

### 3. BM25 Index Building
Hybrid search builds a BM25 index on-the-fly for each query:
- Small result set (30 docs): <100ms
- Large result set (100+ docs): 500ms-2s

**Location**: `parent_child_store.py` lines 522-534

## Diagnostic Changes Made

I've added comprehensive timing logs to pinpoint the exact bottleneck:

### 1. Embedding Generation Timing
- **File**: `parent_child_store.py`
- **Lines**: 419-426
- **Logs**:
  - `[TIMING] Starting embedding generation...`
  - `[TIMING] ⚠️  EMBEDDING GENERATION took X.XXXs`

### 2. ChromaDB Query Timing
- **File**: `parent_child_store.py`
- **Lines**: 431-440
- **Logs**:
  - `[TIMING] Starting ChromaDB query...`
  - `[TIMING] ⚠️  CHROMADB QUERY took X.XXXs`

### 3. BM25 Timing
- **File**: `parent_child_store.py`
- **Lines**: 521-534
- **Logs**:
  - `[TIMING] Building BM25 index...`
  - `[TIMING] ⚠️  TOTAL BM25 took X.XXXs`

### 4. Overall Retrieval Timing
- **File**: `airgapped_rag_advanced.py`
- **Lines**: 527-538
- **Logs**:
  - `[TIMING] ⚠️  DOCUMENT RETRIEVAL took X.XXXs`

### 5. Total Time Before Streaming
- **File**: `airgapped_rag_advanced.py`
- **Lines**: 629-632
- **Logs**:
  - `[TIMING] ⚠️⚠️⚠️  TOTAL TIME BEFORE LLM STREAMING: X.XXXs`
  - This is the delay users experience!

### 6. GPU Detection at Startup
- **File**: `parent_child_store.py`
- **Lines**: 202-217
- **Logs**:
  - `⚠️  Using device for embeddings: cuda/cpu`
  - `✅ GPU detected: <name>` OR
  - `⚠️⚠️⚠️  NO GPU DETECTED! Embeddings will run on CPU...`

## How to Diagnose

### Step 1: Check GPU Status
When you start the API, look for these logs:
```
⚠️  Using device for embeddings: cuda
✅ GPU detected: NVIDIA GeForce RTX 3090
Embedding model device: cuda:0
```

If you see:
```
⚠️  Using device for embeddings: cpu
⚠️⚠️⚠️  NO GPU DETECTED! Embeddings will run on CPU which is VERY SLOW!
```

**This is your problem!** The embedding model needs GPU access.

### Step 2: Check Query Timing
When you run a RAG query, look for:
```
[TIMING] Starting embedding generation...
[TIMING] ⚠️  EMBEDDING GENERATION took 0.234s    <-- Should be <1s with GPU
[TIMING] Starting ChromaDB query...
[TIMING] ⚠️  CHROMADB QUERY took 0.156s
[TIMING] Building BM25 index...
[TIMING] ⚠️  TOTAL BM25 took 0.089s
[TIMING] ⚠️⚠️⚠️  TOTAL TIME BEFORE LLM STREAMING: 0.842s    <-- Total delay
```

If you see embedding generation >10 seconds, the GPU is not being used!

## Solutions

### If GPU Not Detected

1. **Check Docker GPU Access** (if using Docker):
   ```bash
   docker run --gpus all ...
   ```

2. **Check CUDA Installation**:
   ```bash
   python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

3. **Check GPU Permissions**:
   ```bash
   nvidia-smi
   ```

### If ChromaDB is Slow

1. **Check collection size**:
   ```python
   # Add this to your diagnostic output
   logger.info(f"Child collection size: {self.child_collection.count()}")
   ```

2. **Consider reducing top_k** in hybrid search (currently 30→150 for BM25)

### If BM25 is Slow

1. **Disable hybrid search** temporarily to test:
   ```python
   use_hybrid=False  # In query request
   ```

2. **Reduce semantic result count** before BM25:
   ```python
   top_k=min(top_k * 5, 100)  # Line 476, reduce multiplier from 5 to 3
   ```

## Next Steps

1. **Restart your API** with the diagnostic logging
2. **Test a RAG query** and capture the logs
3. **Share the timing logs** to identify the exact bottleneck
4. **Fix the root cause** based on the timing data

The logs will clearly show which operation is taking >60 seconds!

## Files Modified

- `parent_child_store.py`: Added timing logs for embedding, ChromaDB, and BM25
- `airgapped_rag_advanced.py`: Added timing logs for overall retrieval and total delay
