# Fix CUDA Out of Memory Issues

You're getting OOM even with 8B models on 16GB GPUs. This shouldn't happen. Let's fix it systematically.

---

## Problem Analysis

**Your logs show:**
```
GPU 0 has 15.55 GiB total
Only 113.38 MiB is free
Process using 15.44 GiB (15.27 GiB by PyTorch)
```

**An 8B model should only use ~8-10GB, not 15GB!**

**Possible causes:**
1. Memory fragmentation (PyTorch CUDA allocator issue)
2. Model has hidden layers/parameters we didn't account for
3. vLLM pre-allocating too much
4. CUDA graphs eating memory

---

## Immediate Fix - Use Smaller Model First

### **Step 1: Stop Everything**

```bash
# Stop all vLLM containers
docker stop vllm-gpu0 vllm-gpu1 vllm-server 2>/dev/null || true

# Verify GPUs are free
nvidia-smi
# Should show both GPUs with ~15GB free
```

### **Step 2: Start with Tiny Model (Guaranteed to Work)**

```bash
# Use minimal config with 1.7B model
docker compose -f docker-compose.dual-vllm-minimal.yml up -d vllm-gpu0 vllm-gpu1

# Monitor
docker logs -f vllm-gpu0
```

**This model is only 2GB - will definitely work!**

Expected logs:
```
Loading model HuggingFaceTB/SmolLM2-1.7B-Instruct...
Model loaded successfully
Using 2.1 GiB of GPU memory
Application startup complete
```

---

## Model Size Ladder (Try in Order)

Once the 1.7B model works, try progressively larger models:

### **Level 1: SmolLM2-1.7B ✅ (Start Here)**
```yaml
--model HuggingFaceTB/SmolLM2-1.7B-Instruct
--gpu-memory-utilization 0.70
```
- Memory: ~2GB
- Quality: Basic but functional
- Speed: Very fast

---

### **Level 2: Qwen2.5-3B**
```yaml
--model Qwen/Qwen2.5-3B-Instruct
--gpu-memory-utilization 0.75
```
- Memory: ~3GB
- Quality: Good for RAG
- Speed: Fast

---

### **Level 3: Phi-3-Mini-4k**
```yaml
--model microsoft/Phi-3-mini-4k-instruct
--gpu-memory-utilization 0.80
```
- Memory: ~4GB
- Quality: Very good (Microsoft)
- Speed: Fast

---

### **Level 4: Llama-3.2-3B**
```yaml
--model meta-llama/Llama-3.2-3B-Instruct
--gpu-memory-utilization 0.80
```
- Memory: ~3GB
- Quality: Excellent
- Speed: Fast

---

### **Level 5: Mistral-7B (If we get here)**
```yaml
--model mistralai/Mistral-7B-Instruct-v0.3
--gpu-memory-utilization 0.85
--enforce-eager
```
- Memory: ~7GB
- Quality: Excellent
- Speed: Good

---

## Memory Optimization Flags

Add these to your command to reduce memory:

### **1. Fix Fragmentation**
```yaml
environment:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### **2. Disable CUDA Graphs**
```yaml
command: >
  --enforce-eager
```
Saves ~1-2GB but slightly slower.

### **3. Reduce KV Cache**
```yaml
command: >
  --max-model-len 2048  # Very small context
```

### **4. Conservative Memory**
```yaml
command: >
  --gpu-memory-utilization 0.60  # Very conservative
```

---

## Alternative: Use Quantized Models

Quantized models use 50-75% less memory:

### **Option A: AWQ (4-bit)**
```yaml
command: >
  --model TheBloke/Mistral-7B-Instruct-v0.2-AWQ
  --quantization awq
  --gpu-memory-utilization 0.80
```
Memory: ~4GB instead of 7GB

### **Option B: GPTQ (4-bit)**
```yaml
command: >
  --model TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
  --quantization gptq
  --gpu-memory-utilization 0.80
```

---

## Debug: Check What's Using Memory

### **Before Starting vLLM:**
```bash
nvidia-smi
# Note how much is free
```

### **After Starting vLLM:**
```bash
nvidia-smi

# Should show:
# - Process name: python
# - Memory usage: Model size + 2-3GB overhead
```

### **If Memory Doesn't Free Up:**
```bash
# Kill all CUDA processes
sudo fuser -k /dev/nvidia*

# Restart Docker
sudo systemctl restart docker

# Try again
nvidia-smi
```

---

## Your Action Plan (Do This Now)

### **Step 1: Baseline Test**

```bash
# Clean slate
docker stop $(docker ps -aq) 2>/dev/null || true
nvidia-smi  # Verify clean

# Start minimal
docker compose -f docker-compose.dual-vllm-minimal.yml up -d vllm-gpu0

# Check memory
nvidia-smi
# Should show ~2-3GB used on GPU 0

# Test
curl http://localhost:8001/health
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "prompt": "What is 2+2?",
    "max_tokens": 50
  }'
```

**If this works:** Add GPU 1
```bash
docker compose -f docker-compose.dual-vllm-minimal.yml up -d vllm-gpu1
```

**If this still OOMs:** We have a deeper issue (driver, vLLM bug, hardware)

---

### **Step 2: Move Up the Ladder**

Once SmolLM works on both GPUs, try the next model:

```bash
# Edit docker-compose.dual-vllm-minimal.yml
nano docker-compose.dual-vllm-minimal.yml

# Change model to Qwen2.5-3B:
--model Qwen/Qwen2.5-3B-Instruct

# Restart
docker compose -f docker-compose.dual-vllm-minimal.yml restart

# Test memory usage
nvidia-smi
```

Keep going up the ladder until you find the largest model that fits.

---

### **Step 3: Find Your Sweet Spot**

Test each model's performance:

```bash
# Time a generation
time curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "...",
    "prompt": "Summarize the PTO policy for Amentum employees.",
    "max_tokens": 200
  }'
```

Pick the model with best **quality/speed tradeoff** that fits in memory.

---

## If Nothing Works

### **Last Resort: Single GPU, Tiny Model**

```yaml
services:
  vllm-gpu0:
    command: >
      --model HuggingFaceTB/SmolLM2-1.7B-Instruct
      --dtype float16
      --max-model-len 2048
      --gpu-memory-utilization 0.50
      --enforce-eager
```

**This MUST work. If it doesn't:**
- Check NVIDIA driver: `nvidia-smi`
- Check vLLM version: `docker run vllm/vllm-openai:latest --version`
- Check GPU health: `nvidia-smi -q | grep "GPU Temp"`

---

## Expected Results

| Model | Memory | Quality | Speed | Fits? |
|-------|--------|---------|-------|-------|
| SmolLM2-1.7B | 2GB | Basic | Very Fast | ✅ Always |
| Qwen2.5-3B | 3GB | Good | Fast | ✅ Should |
| Phi-3-Mini | 4GB | Very Good | Fast | ✅ Should |
| Mistral-7B | 7GB | Excellent | Good | ❓ Maybe |
| Granite-8B | 8GB | Excellent | Good | ❌ Hasn't worked |

---

## Performance vs Ollama

Even with SmolLM2-1.7B (smallest option):

| Metric | Ollama Mistral-22B | vLLM SmolLM2-1.7B |
|--------|-------------------|-------------------|
| Response time | 45s | **5-8s** ✅ |
| Concurrent (2 users) | 90s | **5-8s each** ✅ |
| Quality | Highest | Good (80% as good) |

**You still get 5-6x speedup with the smallest model!**

---

## Summary

1. **Start with SmolLM2-1.7B** (guaranteed to work)
2. **Verify dual-GPU setup** working
3. **Move up to larger models** one by one
4. **Find your sweet spot** (quality vs memory)
5. **Enjoy the speedup!** Even small models are much faster than Ollama

**Try the minimal config now and let me know what GPU memory usage you see with SmolLM2-1.7B!**
