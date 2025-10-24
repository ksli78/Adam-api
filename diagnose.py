"""
Diagnostic script to identify why "No module named 'api'" error occurs.

Run this to diagnose the issue:
    python diagnose.py
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("Air-Gapped RAG - Diagnostic Tool")
print("=" * 70)
print()

# 1. Check current directory
print("1. Current Directory:")
print(f"   {os.getcwd()}")
print()

# 2. Check if old modules exist
print("2. Checking for old modules:")
old_modules = ['api', 'ingestion', 'qa', 'retrieval', 'config.py']
for module in old_modules:
    exists = Path(module).exists()
    status = "❌ FOUND (PROBLEM!)" if exists else "✅ Not found (good)"
    print(f"   {module}: {status}")
print()

# 3. Check if new file exists
print("3. Checking for new file:")
new_file = Path("airgapped_rag.py")
exists = new_file.exists()
status = "✅ Found (good)" if exists else "❌ NOT FOUND (PROBLEM!)"
print(f"   airgapped_rag.py: {status}")
print()

# 4. Check Python path
print("4. Python sys.path (checking for 'api' references):")
suspicious = False
for p in sys.path:
    if 'api' in p.lower() or 'ingestion' in p.lower():
        print(f"   ⚠️  {p}")
        suspicious = True
if not suspicious:
    print("   ✅ No suspicious paths found")
print()

# 5. Check environment variables
print("5. Environment Variables:")
env_vars = ['PYTHONPATH', 'OLLAMA_BASE_URL', 'DATA_DIR', 'HOST', 'PORT']
for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"   {var}: {value}")
    else:
        print(f"   {var}: (not set)")
print()

# 6. Check if we can import the new module
print("6. Testing import of airgapped_rag:")
try:
    # Try to import without actually loading FastAPI
    spec = __import__('importlib.util').util.find_spec('airgapped_rag')
    if spec is None:
        print("   ❌ Cannot find airgapped_rag module")
        print("   → Make sure you're in the correct directory")
    else:
        print(f"   ✅ Found at: {spec.origin}")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# 7. Check installed packages
print("7. Checking installed packages:")
try:
    import fastapi
    print(f"   ✅ fastapi: {fastapi.__version__}")
except ImportError:
    print("   ❌ fastapi: NOT INSTALLED")

try:
    import chromadb
    print(f"   ✅ chromadb: {chromadb.__version__}")
except ImportError:
    print("   ❌ chromadb: NOT INSTALLED")

try:
    import ollama
    print(f"   ✅ ollama: (installed)")
except ImportError:
    print("   ❌ ollama: NOT INSTALLED")

try:
    import uvicorn
    print(f"   ✅ uvicorn: {uvicorn.__version__}")
except ImportError:
    print("   ❌ uvicorn: NOT INSTALLED")
print()

# 8. Check for __pycache__
print("8. Checking for cached bytecode:")
cache_found = False
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        print(f"   ⚠️  Found: {root}/__pycache__/")
        cache_found = True
if not cache_found:
    print("   ✅ No __pycache__ directories found")
print()

# 9. Recommendations
print("=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)

if Path('api').exists():
    print("❌ Old 'api' directory still exists!")
    print("   Fix: Run 'git pull' to get latest code")
    print()

if not new_file.exists():
    print("❌ airgapped_rag.py not found!")
    print("   Fix: Make sure you're in the project root directory")
    print()

if suspicious:
    print("⚠️  Suspicious paths in sys.path")
    print("   Fix: Close all terminals and restart")
    print()

if cache_found:
    print("⚠️  Cached bytecode found")
    print("   Fix: Run this in PowerShell:")
    print("   Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Force -Recurse")
    print()

# Final recommendation
print("To run the API, use ONE of these commands:")
print("   python airgapped_rag.py")
print("   python run_dev.py")
print()
print("DO NOT use: uvicorn api.main:app (old command)")
print("=" * 70)
