# syntax=docker/dockerfile:1

#######################################################################
# Stage: API (GPU-enabled) — FastAPI app + cache PUBLIC HF models
#######################################################################
FROM python:3.11-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# App requirements
COPY requirements.txt /tmp/requirements.txt

# 1) toolchain
# 2) pin numpy<2 to satisfy faiss ABI
# 3) install your app deps
# 4) install CUDA-enabled PyTorch stack
# 5) install faiss-cpu built against numpy 1.x
# 6) modern HF stack
RUN pip install --upgrade pip setuptools wheel && \
    pip install "numpy==1.26.4" && \
    pip install -r /tmp/requirements.txt && \
    # SAFETY: re-pin numpy/faiss ABI-compatible
    pip install --no-deps --upgrade numpy==1.26.4 faiss-cpu==1.8.0 && \
    # ---- GPU Torch stack (CUDA 12.1) ----
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        --force-reinstall torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 && \
    pip install faiss-cpu==1.8.0.post1 && \
    pip install --upgrade --upgrade-strategy eager \
        transformers==4.46.1 \
        sentence-transformers==3.2.0 \
        tokenizers==0.20.1 \
        huggingface_hub==0.25.2 \
        safetensors>=0.4.4 \
        pillow>=10.4.0 && \
    # ---- ADD THIS: Docling with VLM extras ----
    pip install --no-cache-dir "docling[vlm]"


# Avoid torchvision import during build-time caching
ENV TRANSFORMERS_NO_TORCHVISION=1

# Pre-download PUBLIC IBM repos only (no token required)
# - Granite embedding (ModernBERT)
# - Granite reranker  (ModernBERT)
RUN python - <<'PY'
from huggingface_hub import snapshot_download

repos = [
    "ibm-granite/granite-embedding-english-r2",
    "ibm-granite/granite-embedding-reranker-english-r2",
]
for repo in repos:
    snapshot_download(
        repo_id=repo,
        local_dir=None,              # use HF_HOME cache (/opt/hf)
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8
    )
print("Public Granite repos cached successfully.")
PY

# Pre-download Docling's default VLM (SmolDocling, Transformers backend)
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ds4sd/SmolDocling-256M-preview",
    local_dir=None,              # goes to HF cache (HF_HOME=/opt/hf)
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=8
)
print("SmolDocling cached successfully.")
PY

ENV PYTHONPATH=/app
# Copy app
WORKDIR /app
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


#######################################################################
# Stage: OLLAMA — GPU-enabled Ollama + pre-pulled Granite 4
#######################################################################
FROM ollama/ollama:latest AS ollama

# Use granite4 tag (no ibm/ prefix)
ARG OLLAMA_MODEL=granite4:latest

# Pre-pull at build time and exit cleanly
RUN set -eux; \
  (ollama serve >/tmp/ollama.log 2>&1 &) ; \
  for i in $(seq 1 60); do \
    curl -fsS http://127.0.0.1:11434/api/tags && break; \
    sleep 1; \
  done; \
  OLLAMA_HOST=127.0.0.1:11434 ollama pull "${OLLAMA_MODEL}"; \
  OLLAMA_HOST=127.0.0.1:11434 ollama show "${OLLAMA_MODEL}" >/dev/null; \
  pkill -f "ollama serve" || true; \
  sleep 1; \
  exit 0

EXPOSE 11434
CMD ["serve"]
