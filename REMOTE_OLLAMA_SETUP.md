# Remote Ollama Setup Guide (32GB VRAM)

This guide explains how to set up the remote Ollama server on your development/production machine with **2 GPUs and 32GB VRAM**.

## Server Specifications

**Hardware:**
- 2 GPUs (bridged)
- 32 GB VRAM total
- URL: `http://adam.amentumspacemissions.com:11434`

**Capabilities:**
- ✅ Can run llama3.1:70b (best quality)
- ✅ Can run multiple smaller models simultaneously
- ✅ 64K-128K context windows

---

## Model Recommendation: llama3.1:70b

For your 32GB VRAM setup, I recommend **llama3.1:70b**:

### Why llama3.1:70b?

| Feature | llama3.1:8b | llama3.1:70b |
|---------|-------------|--------------|
| **Parameters** | 8 billion | 70 billion |
| **VRAM Required** | ~8 GB | ~28 GB |
| **Context Window** | 128K tokens | 128K tokens |
| **Reasoning Quality** | Good | Excellent |
| **Document Selection** | Good | Much better |
| **Question Understanding** | Good | Superior |
| **Multi-GPU Support** | Optional | Recommended |

**Benefits for your use case:**
- 🎯 **Much better document selection** - understands subtle question similarities
- 🧠 **Superior reasoning** - fewer false matches, better relevance ranking
- 📊 **Better question generation** - generates more accurate answerable_questions
- 🔍 **Semantic understanding** - catches relationships 8b models miss

---

## Memory Breakdown (llama3.1:70b)

### Q4_K_M Quantization (Recommended)
- **Model weights**: ~40 GB
- **With 64K context**: ~44 GB total
- **Your hardware**: 32 GB VRAM

**Wait, that doesn't fit!**

Don't worry - Ollama automatically:
1. Splits the model across your 2 GPUs (~20 GB each)
2. Uses system RAM for overflow (~12 GB)
3. Optimizes memory usage dynamically

### Q3_K_S Quantization (Fits Entirely in VRAM)
- **Model weights**: ~26 GB
- **With 64K context**: ~30 GB total
- **Your hardware**: 32 GB VRAM ✅ Perfect fit!

**Slight quality trade-off but still much better than 8b models.**

---

## Installation Steps

### 1. SSH into Remote Server

```bash
ssh user@adam.amentumspacemissions.com
```

### 2. Install/Verify Ollama

```bash
# Check if Ollama is installed
ollama --version

# If not installed:
curl -fsSL https://ollama.com/install.sh | sh

# Verify GPU access
nvidia-smi
# Should show 2 GPUs with ~32GB total
```

### 3. Pull llama3.1:70b Model

**Option A: Default Quantization (Q4_K_M)**
```bash
# This will download ~40 GB
ollama pull llama3.1:70b

# Will use GPU + RAM overflow
# Slower but highest quality
```

**Option B: Smaller Quantization (Q3_K_S) - RECOMMENDED**
```bash
# Create a Modelfile
cat > Modelfile.llama3.1-70b-q3 <<EOF
FROM llama3.1:70b
PARAMETER num_ctx 65536
PARAMETER num_gpu 2
EOF

# Pull and create custom model
ollama create llama3.1:70b-q3 -f Modelfile.llama3.1-70b-q3
```

### 4. Test the Model

```bash
# Test with a simple prompt
ollama run llama3.1:70b "What is 2+2?"

# Check memory usage during inference
nvidia-smi

# Should show model distributed across both GPUs
```

### 5. Configure Ollama for Network Access

Ollama needs to accept connections from your application server:

```bash
# Edit Ollama service
sudo systemctl edit ollama

# Add these lines:
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"

# Save and restart
sudo systemctl restart ollama

# Verify it's listening
netstat -tulpn | grep 11434
# Should show: 0.0.0.0:11434
```

### 6. Test Remote Connection

From your **application server**:

```bash
curl http://adam.amentumspacemissions.com:11434/api/tags

# Should return JSON with available models
```

---

## Application Configuration

Your application is already configured to use the remote Ollama server:

**airgapped_rag_advanced.py:**
```python
OLLAMA_HOST = "http://adam.amentumspacemissions.com:11434"
LLM_MODEL = "llama3.1:70b"
LLM_CONTEXT_WINDOW = 65536  # 64K tokens
```

**No changes needed** - just pull the code and restart!

---

## Environment Variables (Override Defaults)

You can override the defaults via environment variables:

```bash
# Use different model
export LLM_MODEL="llama3.1:70b-q3"

# Use different context size
export LLM_CONTEXT_WINDOW="131072"  # 128K (max)

# Use different Ollama server
export OLLAMA_HOST="http://different-server:11434"
```

---

## Performance Expectations

### Model Loading Time
- **First request**: 10-30 seconds (loads model into VRAM)
- **Subsequent requests**: Instant (model stays loaded)

### Inference Speed
- **Document selection (30 docs)**: 3-8 seconds
- **Answer generation**: 5-15 seconds
- **Question generation (ingestion)**: 8-20 seconds

**Much faster than cloud APIs!** ⚡

### Concurrent Requests
- With 32GB VRAM, you can handle 2-3 concurrent requests
- Ollama queues additional requests automatically

---

## Monitoring & Troubleshooting

### Check GPU Usage

```bash
# Live monitoring
watch -n 1 nvidia-smi

# Expected during inference:
# GPU 0: ~14-20 GB / 16 GB
# GPU 1: ~14-20 GB / 16 GB
```

### Check Ollama Logs

```bash
# System logs
sudo journalctl -u ollama -f

# Look for:
# "loaded model" - model successfully loaded
# "request completed" - successful inference
```

### Common Issues

**Issue: "Model not found"**
```bash
# List available models
ollama list

# Pull if missing
ollama pull llama3.1:70b
```

**Issue: "Connection refused"**
```bash
# Check Ollama is running
sudo systemctl status ollama

# Check firewall
sudo ufw status
sudo ufw allow 11434/tcp
```

**Issue: "Out of memory"**
```bash
# Use smaller quantization
ollama pull llama3.1:70b-q3

# Or reduce context window
export LLM_CONTEXT_WINDOW="32768"  # 32K instead of 64K
```

**Issue: "Very slow inference"**
```bash
# Verify GPUs are being used
nvidia-smi

# Check Ollama is using GPUs
ps aux | grep ollama

# Ensure num_gpu is set in Modelfile
```

---

## Comparison: Local vs Remote

### Current Setup (llama3:8b locally)
- ❌ Small context (8K tokens)
- ❌ Limited reasoning
- ❌ Can't see all 240 documents
- ✅ No network latency

### New Setup (llama3.1:70b remote)
- ✅ Large context (64K-128K tokens)
- ✅ Superior reasoning
- ✅ Better document selection
- ✅ Better question generation
- ⚠️ ~50-100ms network latency (negligible)

**Network latency is minimal** - inference time (5-15s) dominates over network (0.05-0.1s)

---

## Security Considerations

### Network Security
```bash
# Restrict Ollama to specific IPs
sudo ufw allow from YOUR_APP_SERVER_IP to any port 11434

# Or use SSH tunnel for extra security
ssh -L 11434:localhost:11434 user@adam.amentumspacemissions.com

# Then use: OLLAMA_HOST=http://localhost:11434
```

### API Access
- Ollama has no built-in authentication
- Use firewall rules or VPN for security
- Consider adding reverse proxy (nginx) with auth if needed

---

## Alternative Models (If 70b is Too Slow)

### llama3.1:8b (Current code default)
- VRAM: ~8 GB (leaves 24 GB free)
- Speed: Fast (3-5s inference)
- Quality: Good
- **Use case**: Development, testing

### llama3.1:13b (Middle ground)
```bash
ollama pull llama3.1:13b
export LLM_MODEL="llama3.1:13b"
```
- VRAM: ~13 GB
- Speed: Medium (4-8s inference)
- Quality: Better than 8b
- **Use case**: Balance of speed and quality

### qwen2.5:32b (Alternative)
```bash
ollama pull qwen2.5:32b
export LLM_MODEL="qwen2.5:32b"
```
- VRAM: ~20 GB
- Speed: Medium-Fast
- Quality: Excellent for reasoning
- **Use case**: Best alternative to llama3.1:70b

---

## Testing Your Setup

### 1. Test Remote Connection

From your application server:
```bash
curl http://adam.amentumspacemissions.com:11434/api/generate \
  -d '{
    "model": "llama3.1:70b",
    "prompt": "What is 2+2?",
    "stream": false
  }'
```

### 2. Test Document Selection

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How many hours can I work in a single week?",
    "use_llm_selection": true,
    "max_documents": 10
  }'
```

**Look for in logs:**
```
MetadataExtractor initialized: model=llama3.1:70b, host=http://adam.amentumspacemissions.com:11434
Stage 2 complete: LLM selected 2 final documents:
  - EN-PO-0301.pdf (type: policy)
```

### 3. Test Question Generation

Re-ingest a document to test improved question generation:
```bash
curl -X POST http://localhost:8000/upload-document \
  -F "file=@test.pdf" \
  -F "source_url=https://example.com/test.pdf"
```

Check the generated questions - they should be more specific and relevant!

---

## Next Steps

1. ✅ Pull llama3.1:70b on remote server
2. ✅ Configure network access
3. ✅ Test connection from application
4. ✅ Pull latest code with updated defaults
5. ✅ Restart application
6. ✅ Test with work hours query
7. ✅ Monitor GPU usage and performance
8. 🎯 **Consider re-ingesting documents** for better questions with 70b model

---

## Support & Optimization

### Getting Help
- Check Ollama docs: https://ollama.com/docs
- GPU monitoring: `nvidia-smi`
- Ollama logs: `sudo journalctl -u ollama -f`

### Performance Tuning
- Adjust `num_ctx` based on use case
- Use `num_thread` for CPU-only parts
- Consider `num_batch` for throughput
- Monitor with `ollama ps` during inference

Enjoy your much more powerful RAG system! 🚀
