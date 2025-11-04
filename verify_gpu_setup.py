#!/usr/bin/env python3
"""
Verification script for GPU and e5-large-v2 embedding model setup.
Run this to verify your configuration before re-ingesting documents.
"""

import torch
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("GPU & e5-large-v2 CONFIGURATION VERIFICATION")
print("=" * 60)

# Check PyTorch and CUDA
print("\n1. PyTorch Configuration:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("   ⚠️  WARNING: CUDA not available - will use CPU (slower)")

# Test e5-large-v2 model loading
print("\n2. Loading e5-large-v2 model...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    model = SentenceTransformer("intfloat/e5-large-v2", device=device)
    print(f"   ✅ Model loaded successfully!")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Test encoding
    print("\n3. Testing embedding generation...")
    test_text = "query: What is the PTO policy?"
    embedding = model.encode(test_text, convert_to_tensor=True)
    print(f"   ✅ Generated embedding shape: {embedding.shape}")
    print(f"   Embedding device: {embedding.device}")

    print("\n" + "=" * 60)
    print("✅ ALL CHECKS PASSED - Ready for document ingestion!")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Verify NVIDIA drivers: nvidia-smi")
    print("3. Check CUDA toolkit installation")
