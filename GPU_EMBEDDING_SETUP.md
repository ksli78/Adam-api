# GPU & e5-large-v2 Embedding Model Setup

## ✅ Current Configuration Status

Everything is **already configured** in your branch! Here's what's in place:

### 1. PyTorch with CUDA 11.8 ✅
**File**: `requirements.txt` (lines 6-10)
```python
# PyTorch with CUDA 11.8 support for GPU acceleration
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.1.0+cu118
torchvision==0.16.0+cu118
torchaudio==2.1.0+cu118
```

### 2. e5-large-v2 Embedding Model ✅
**File**: `parent_child_store.py` (line 182)
```python
embedding_model: str = "intfloat/e5-large-v2"  # 1024 dimensions
```

### 3. GPU Auto-Detection ✅
**File**: `parent_child_store.py` (lines 202-210)
```python
# Detect GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# Initialize with GPU
self.embedding_model = SentenceTransformer(embedding_model, device=device)
```

### 4. e5 Query Prefix for Optimal Retrieval ✅
**File**: `parent_child_store.py` (lines 405-411)
```python
# For e5 models, prefix queries with "query: " for best performance
if "e5" in self.embedding_model_name.lower():
    query_text = f"query: {expanded_query}"
```

### 5. Acronym Expansion ✅
**File**: `config/acronyms.json`
- Automatically expands acronyms before embedding (PTO → Paid Time Off)
- Helps e5 model understand domain-specific terms

---

## 🚀 How to Re-Ingest Documents

### Step 1: Verify GPU Setup
```bash
python verify_gpu_setup.py
```

This will check:
- ✅ PyTorch and CUDA availability
- ✅ GPU name and memory
- ✅ e5-large-v2 model loading
- ✅ Embedding generation on GPU

### Step 2: Clear Existing Embeddings (Optional)
If you want to start fresh:
```bash
# Delete existing ChromaDB
rm -rf /data/airgapped_rag/chromadb_advanced
```

### Step 3: Re-Ingest Documents
Start the API and upload documents:
```bash
python airgapped_rag_advanced.py
```

Then use the upload endpoint:
```bash
curl -X POST "http://localhost:8000/upload-document" \
  -F "file=@your_document.pdf"
```

---

## 📊 Performance Improvements

### e5-large-v2 vs Previous Models:
| Model | Dimensions | Retrieval Quality | GPU Required |
|-------|-----------|------------------|--------------|
| all-MiniLM-L6-v2 | 384 | Good | No |
| all-mpnet-base-v2 | 768 | Better | No |
| **e5-large-v2** | **1024** | **Best** | **Recommended** |

### GPU Acceleration Benefits:
- **10-50x faster** embedding generation
- Can process large documents in seconds vs minutes
- Essential for e5-large-v2 (larger model)

---

## 🔧 Troubleshooting

### GPU Not Detected?
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Reinstall PyTorch with CUDA
pip install torch==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### Model Download Issues?
The first time you run, e5-large-v2 (~1.3GB) will download from HuggingFace.

### Out of GPU Memory?
The model uses ~2-3GB GPU memory. If needed, you can batch documents or use CPU fallback.

---

## 📈 What's Different from Yesterday?

All the GPU and e5-large-v2 code was already merged from:
- Commit `43d3806`: Set PyTorch CUDA 11.8 as default
- Commit `e9d9a76`: Add PyTorch to requirements.txt
- Commit `1ebf159`: Add GPU acceleration support
- Commit `aa0e891`: Add e5 model query prefix support
- Commit `a2d917f`: Switch to e5-large-v2 embedding model

**Everything is ready to go!** 🎉
