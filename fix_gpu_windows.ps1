# PowerShell script to fix GPU/CUDA detection issues on Windows
# Run this if PyTorch is not detecting your NVIDIA GPU

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "GPU/CUDA Fix Script for Windows" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Run diagnostics
Write-Host "Step 1: Running diagnostics..." -ForegroundColor Green
python diagnose_gpu.py
Write-Host ""

# Ask user if they want to proceed with fix
Write-Host "============================================" -ForegroundColor Yellow
Write-Host "Common Fix: Reinstall PyTorch with CUDA" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "This will:" -ForegroundColor White
Write-Host "  1. Uninstall current PyTorch installation" -ForegroundColor White
Write-Host "  2. Reinstall PyTorch with CUDA 11.8 support" -ForegroundColor White
Write-Host "  3. Verify GPU detection" -ForegroundColor White
Write-Host ""

$response = Read-Host "Do you want to proceed? (y/n)"

if ($response -ne 'y') {
    Write-Host "Cancelled. Please manually fix the issue." -ForegroundColor Yellow
    exit
}

Write-Host ""
Write-Host "Step 2: Uninstalling current PyTorch..." -ForegroundColor Green
pip uninstall torch torchvision torchaudio -y

Write-Host ""
Write-Host "Step 3: Installing PyTorch with CUDA 11.8..." -ForegroundColor Green
Write-Host "This may take several minutes..." -ForegroundColor Yellow
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Installation failed!" -ForegroundColor Red
    Write-Host "Check your internet connection and try again." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 4: Verifying GPU detection..." -ForegroundColor Green
python diagnose_gpu.py

Write-Host ""
Write-Host "Step 5: Testing with verify_gpu_setup.py..." -ForegroundColor Green
python verify_gpu_setup.py

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Fix completed!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "If GPU is still not detected, you may need to:" -ForegroundColor Yellow
Write-Host "  1. Update NVIDIA drivers" -ForegroundColor White
Write-Host "  2. Install CUDA Toolkit 11.8" -ForegroundColor White
Write-Host "  3. Restart your computer" -ForegroundColor White
Write-Host ""
