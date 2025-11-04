# Installation Troubleshooting - rapidfuzz Hanging on Windows

## Problem
`pip install -r requirements.txt` hangs on downloading rapidfuzz-3.9.2-cp311-win_amd64.whl.metadata

## Solutions (Try in Order)

### Solution 1: Install rapidfuzz Separately First
```bash
# Cancel current install (Ctrl+C)

# Install rapidfuzz alone with timeout
pip install rapidfuzz==3.9.2 --timeout=60 --no-cache-dir

# If successful, install rest of requirements
pip install -r requirements.txt --no-cache-dir
```

### Solution 2: Use Older Version of rapidfuzz
```bash
# Try a known stable version
pip install rapidfuzz==3.6.0 --timeout=60 --no-cache-dir

# Then install rest
pip install -r requirements.txt --no-cache-dir
```

### Solution 3: Install Without rapidfuzz (Temporary)
```bash
# Install everything except rapidfuzz
pip install -r requirements_no_rapidfuzz.txt --no-cache-dir

# Try rapidfuzz later or skip if not critical
```

### Solution 4: Use Different PyPI Mirror
```bash
# Try with a different index
pip install rapidfuzz==3.9.2 --index-url https://pypi.org/simple --timeout=60
```

### Solution 5: Check Network/Firewall
```bash
# Test PyPI connectivity
ping pypi.org

# Check if behind corporate firewall/proxy
# May need to configure proxy settings
```

### Solution 6: Install from Pre-built Wheel
```bash
# Download wheel manually from:
# https://pypi.org/project/rapidfuzz/#files

# Then install locally:
pip install path\to\rapidfuzz-3.9.2-cp311-cp311-win_amd64.whl
```

## What is rapidfuzz Used For?

rapidfuzz is used by ChromaDB for fuzzy string matching. It's not critical for the core RAG functionality but helps with:
- Approximate string matching in metadata
- Query spelling corrections (we have pyspellchecker as backup)

**You can run the system without it if needed.**

## Recommended Approach for Your Setup

Since you're on Windows with GPU, try this sequence:

```bash
# 1. Cancel current install (Ctrl+C)

# 2. Install PyTorch first (it's large and important)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --timeout=120

# 3. Install core dependencies
pip install fastapi uvicorn pydantic sentence-transformers transformers chromadb ollama --no-cache-dir --timeout=120

# 4. Try rapidfuzz with timeout
pip install rapidfuzz --timeout=60 --no-cache-dir

# 5. Install remaining dependencies
pip install -r requirements.txt --no-cache-dir
```

## After Installation - Verify GPU Setup

```bash
python verify_gpu_setup.py
```

This will confirm:
- PyTorch installed correctly
- CUDA/GPU detected
- e5-large-v2 model can load
- Ready for document ingestion
