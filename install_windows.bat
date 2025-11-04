@echo off
REM Windows Installation Script for ADAM RAG System
REM Handles dependency conflicts and avoids pip resolver hangs

echo ============================================
echo ADAM RAG System - Windows Installation
echo ============================================
echo.

REM Step 1: Upgrade pip
echo Step 1: Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 goto :error
echo.

REM Step 2: Install NumPy first (many packages depend on it)
echo Step 2: Installing NumPy...
pip install numpy==1.26.4 --no-cache-dir
if %errorlevel% neq 0 goto :error
echo.

REM Step 3: Install PyTorch with CUDA (large, install early)
echo Step 3: Installing PyTorch with CUDA 11.8...
echo This may take several minutes...
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --timeout=300
if %errorlevel% neq 0 goto :error
echo.

REM Step 4: Install core ML packages
echo Step 4: Installing AI/ML packages...
pip install transformers==4.48.1 tokenizers --no-cache-dir
pip install sentence-transformers==3.2.0 --no-cache-dir
pip install accelerate==0.34.2 --no-cache-dir
if %errorlevel% neq 0 goto :error
echo.

REM Step 5: Install web framework (pydantic, fastapi, uvicorn)
echo Step 5: Installing web framework...
pip install pydantic --no-cache-dir
pip install fastapi==0.110.1 --no-cache-dir
pip install uvicorn[standard]==0.27.1 --no-cache-dir
if %errorlevel% neq 0 goto :error
echo.

REM Step 6: Install vector database and RAG tools
echo Step 6: Installing RAG tools...
pip install chromadb --no-cache-dir --timeout=180
pip install ollama --no-cache-dir
pip install pypdf PyMuPDF --no-cache-dir
if %errorlevel% neq 0 goto :error
echo.

REM Step 7: Install retrieval helpers (skip rapidfuzz if it hangs)
echo Step 7: Installing retrieval helpers...
pip install rank_bm25==0.2.2 --no-cache-dir
pip install pyspellchecker --no-cache-dir
echo Trying rapidfuzz with timeout...
pip install rapidfuzz==3.6.0 --timeout=60 --no-cache-dir
if %errorlevel% neq 0 (
    echo WARNING: rapidfuzz failed to install, continuing without it...
    echo The system will work fine without rapidfuzz.
)
echo.

REM Step 8: Install document processing
echo Step 8: Installing document processing tools...
pip install docling --no-cache-dir --timeout=180
pip install nltk regex pyyaml --no-cache-dir
if %errorlevel% neq 0 goto :error
echo.

REM Step 9: Install database connector
echo Step 9: Installing SQL database connector...
pip install pyodbc --no-cache-dir
if %errorlevel% neq 0 goto :error
echo.

REM Step 10: Install remaining packages
echo Step 10: Installing remaining packages...
pip install scikit-learn==1.5.0 --no-cache-dir
pip install python-multipart --no-cache-dir
pip install huggingface_hub==0.25.2 safetensors==0.4.5 --no-cache-dir
pip install faiss-cpu==1.8.0 --no-cache-dir
if %errorlevel% neq 0 goto :error
echo.

echo ============================================
echo Installation completed successfully!
echo ============================================
echo.
echo Next steps:
echo 1. Verify GPU setup: python verify_gpu_setup.py
echo 2. Start the API: python airgapped_rag_advanced.py
echo.
goto :end

:error
echo.
echo ============================================
echo ERROR: Installation failed!
echo ============================================
echo Check the error message above.
echo You may need to:
echo - Check your internet connection
echo - Disable antivirus temporarily
echo - Run as administrator
echo.
exit /b 1

:end
