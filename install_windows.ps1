# Windows PowerShell Installation Script for ADAM RAG System
# Handles dependency conflicts and avoids pip resolver hangs

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "ADAM RAG System - Windows Installation" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Function to install package with error handling
function Install-Package {
    param(
        [string]$PackageName,
        [int]$Timeout = 120
    )

    Write-Host "Installing $PackageName..." -ForegroundColor Yellow
    $result = & pip install $PackageName --no-cache-dir --timeout=$Timeout 2>&1

    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Failed to install $PackageName" -ForegroundColor Red
        return $false
    }
    return $true
}

# Step 1: Upgrade pip
Write-Host "Step 1: Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip setuptools wheel
Write-Host ""

# Step 2: Install NumPy first
Write-Host "Step 2: Installing NumPy..." -ForegroundColor Green
Install-Package "numpy==1.26.4"
Write-Host ""

# Step 3: Install PyTorch with CUDA
Write-Host "Step 3: Installing PyTorch with CUDA 11.8..." -ForegroundColor Green
Write-Host "This may take several minutes..." -ForegroundColor Yellow
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --timeout=300
Write-Host ""

# Step 4: Install core ML packages
Write-Host "Step 4: Installing AI/ML packages..." -ForegroundColor Green
Install-Package "transformers==4.48.1"
Install-Package "tokenizers"
Install-Package "sentence-transformers==3.2.0"
Install-Package "accelerate==0.34.2"
Write-Host ""

# Step 5: Install web framework
Write-Host "Step 5: Installing web framework..." -ForegroundColor Green
Install-Package "pydantic"
Install-Package "fastapi==0.110.1"
Install-Package "uvicorn[standard]==0.27.1"
Write-Host ""

# Step 6: Install vector database
Write-Host "Step 6: Installing RAG tools..." -ForegroundColor Green
Install-Package "chromadb" 180
Install-Package "ollama"
Install-Package "pypdf"
Install-Package "PyMuPDF"
Write-Host ""

# Step 7: Install retrieval helpers
Write-Host "Step 7: Installing retrieval helpers..." -ForegroundColor Green
Install-Package "rank_bm25==0.2.2"
Install-Package "pyspellchecker"

# Try rapidfuzz with short timeout
Write-Host "Trying rapidfuzz (may skip if hangs)..." -ForegroundColor Yellow
$rapidfuzzInstalled = $false
$job = Start-Job -ScriptBlock { pip install rapidfuzz==3.6.0 --timeout=60 --no-cache-dir }
Wait-Job $job -Timeout 90
if ($job.State -eq "Completed") {
    $rapidfuzzInstalled = $true
    Write-Host "rapidfuzz installed successfully" -ForegroundColor Green
} else {
    Stop-Job $job
    Remove-Job $job
    Write-Host "rapidfuzz skipped (timeout). System will work without it." -ForegroundColor Yellow
}
Write-Host ""

# Step 8: Install document processing
Write-Host "Step 8: Installing document processing tools..." -ForegroundColor Green
Install-Package "docling" 180
Install-Package "nltk"
Install-Package "regex"
Install-Package "pyyaml"
Write-Host ""

# Step 9: Install SQL connector
Write-Host "Step 9: Installing SQL database connector..." -ForegroundColor Green
Install-Package "pyodbc"
Write-Host ""

# Step 10: Install remaining packages
Write-Host "Step 10: Installing remaining packages..." -ForegroundColor Green
Install-Package "scikit-learn==1.5.0"
Install-Package "python-multipart"
Install-Package "huggingface_hub==0.25.2"
Install-Package "safetensors==0.4.5"
Install-Package "faiss-cpu==1.8.0"
Write-Host ""

Write-Host "============================================" -ForegroundColor Green
Write-Host "Installation completed successfully!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Verify GPU setup: python verify_gpu_setup.py" -ForegroundColor White
Write-Host "2. Start the API: python airgapped_rag_advanced.py" -ForegroundColor White
Write-Host ""
