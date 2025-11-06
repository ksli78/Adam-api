# Upgrading to llama3.1:8b Setup Guide

This guide explains how to upgrade from llama3:8b to llama3.1:8b for the LLM-based document selection feature.

## Why Upgrade?

**llama3.1:8b** offers significant improvements over llama3:8b:
- ✅ **128K context window** (vs 8K) - can handle more documents
- ✅ **Better reasoning** - improved document selection accuracy
- ✅ **Same memory footprint** - ~8GB VRAM with default quantization
- ✅ **Fits your hardware** - works on both dev (12GB) and production (32GB)

## Memory Requirements

### Dev Machine (12 GB VRAM)
- Model (Q4_K_M): ~5 GB
- Context (32K): ~2 GB
- Overhead: ~1 GB
- **Total: ~8 GB** ✅ Fits comfortably

### Production (32 GB VRAM, 2 GPUs)
- Model (Q4_K_M): ~5 GB
- Context (32K): ~2 GB
- Overhead: ~1 GB
- **Total: ~8 GB per GPU** ✅ Plenty of room

Can optionally use:
- Higher quantization (Q8_0) for better quality: ~12 GB
- Larger context (64K): ~12 GB total
- Both: ~16 GB (still fits easily)

## Installation Steps

### 1. Pull llama3.1:8b Model

On your **Docker container where Ollama is running**:

```bash
# Pull the model (will download ~4.7 GB)
ollama pull llama3.1:8b

# Verify it's available
ollama list

# Expected output should include:
# llama3.1:8b    ...    4.7 GB    ...
```

### 2. Update Environment Variables (Optional)

The code now defaults to llama3.1:8b, but you can override via environment:

```bash
# In your .env file or docker-compose.yml
LLM_MODEL=llama3.1:8b
LLM_CONTEXT_WINDOW=32768  # 32K tokens (default)
```

### 3. Restart the Application

```bash
# If using Docker Compose
docker-compose restart

# If running directly
# Stop the current process and restart
python airgapped_rag_advanced.py
```

### 4. Verify the Upgrade

Check the logs on startup - you should see:

```
INFO - MetadataExtractor initialized: model=llama3.1:8b, host=http://localhost:11434, max_chars=6000, context_window=32768
```

## Configuration Options

### Context Window Size

Default is 32K tokens, which is sufficient for:
- ~50 documents with full metadata
- ~30 candidate documents in two-stage selection

To change:

```bash
# Increase for larger document sets (max 128K for llama3.1)
LLM_CONTEXT_WINDOW=65536  # 64K tokens

# Note: Larger context = more VRAM usage
# 32K = ~2 GB
# 64K = ~4 GB
# 128K = ~8 GB
```

### Model Selection

To use different models:

```bash
# Use llama3.1:70b (requires more VRAM, better quality)
LLM_MODEL=llama3.1:70b

# Or stay with llama3:8b (not recommended - small context)
LLM_MODEL=llama3:8b
```

## Testing the Upgrade

Test with the work hours query that was previously failing:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "How many hours can I work in a single week?",
  "use_llm_selection": true,
  "max_documents": 10
}'
```

**Expected behavior:**
```
Stage 1: Using hybrid search to find top 30 candidate documents
Stage 1 complete: Narrowed to 30 candidate documents
Top 10 candidates: EN-PO-0301.pdf (policy), ...
Document catalog size: 45000 characters, 30 documents
Stage 2: Using LLM to select best documents from candidates
Stage 2 complete: LLM selected 2 final documents:
  - EN-PO-0301.pdf (type: policy)
```

## Rollback (If Needed)

If you need to revert to llama3:8b:

```bash
# Set environment variable
LLM_MODEL=llama3:8b
LLM_CONTEXT_WINDOW=8192  # Smaller context for llama3

# Restart application
docker-compose restart
```

## Troubleshooting

### "Model not found" error

```bash
# Pull the model
ollama pull llama3.1:8b

# Check it's available
ollama list
```

### Out of memory errors

```bash
# Reduce context window
LLM_CONTEXT_WINDOW=16384  # 16K tokens

# Or use smaller quantization (edit Modelfile)
# Q4_0 instead of Q4_K_M saves ~1GB
```

### Slow performance

```bash
# Ensure GPU is being used
ollama ps

# Check GPU memory usage
nvidia-smi

# If not using GPU, check Ollama GPU support
```

## Performance Comparison

| Feature | llama3:8b | llama3.1:8b |
|---------|-----------|-------------|
| Context Window | 8K tokens | 128K tokens |
| Memory (default) | ~5 GB | ~5 GB |
| Memory (with 32K ctx) | N/A (too big) | ~8 GB |
| Document Capacity | ~10 docs | ~50 docs |
| Reasoning Quality | Good | Better |
| Speed | Fast | Similar |

## Next Steps

After upgrading:
1. ✅ Test document selection with various queries
2. ✅ Monitor VRAM usage with `nvidia-smi`
3. ✅ Adjust context window if needed
4. ✅ Consider re-ingesting documents to regenerate questions with better model

## Support

If you encounter issues:
1. Check Ollama logs: `docker logs <ollama-container>`
2. Check application logs for model initialization
3. Verify VRAM availability: `nvidia-smi`
4. Ensure Docker has GPU access (if using Docker)
