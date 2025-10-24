# PowerShell script to export Air-Gapped RAG for RHEL9 deployment
#
# This script:
# 1. Saves Docker images as tar files
# 2. Exports Ollama models from Docker volume
# 3. Packages all deployment files
# 4. Creates a deployment package ready for transfer
#
# Usage:
#   .\export-for-rhel9.ps1
#

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Blue
Write-Host "Export Air-Gapped RAG for RHEL9"
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

# Configuration
$ExportDir = "rhel9-deployment"
$ImagesDir = "$ExportDir/docker-images"
$ModelsDir = "$ExportDir/ollama-models"

#
# Step 1: Create directories
#
Write-Host "Step 1: Creating export directories..." -ForegroundColor Blue

if (Test-Path $ExportDir) {
    Write-Host "Warning: $ExportDir already exists" -ForegroundColor Yellow
    $response = Read-Host "Delete and recreate? (y/n)"
    if ($response -eq 'y') {
        Remove-Item -Recurse -Force $ExportDir
        Write-Host "Deleted existing directory" -ForegroundColor Green
    } else {
        Write-Host "Using existing directory" -ForegroundColor Yellow
    }
}

New-Item -ItemType Directory -Force -Path $ImagesDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green
Write-Host ""

#
# Step 2: Build API image (if not already built)
#
Write-Host "Step 2: Building API Docker image..." -ForegroundColor Blue

try {
    docker images airgapped-rag-api:latest -q | Out-Null
    Write-Host "✓ API image already exists" -ForegroundColor Green
} catch {
    Write-Host "Building API image..." -ForegroundColor Yellow
    docker build -f Dockerfile.airgapped -t airgapped-rag-api:latest .
    Write-Host "✓ API image built" -ForegroundColor Green
}
Write-Host ""

#
# Step 3: Save Docker images
#
Write-Host "Step 3: Saving Docker images..." -ForegroundColor Blue

# Pull latest Ollama image
Write-Host "Pulling latest Ollama image..."
docker pull ollama/ollama:latest
Write-Host "✓ Ollama image pulled" -ForegroundColor Green

# Pull RHEL UBI9 Python base image
Write-Host "Pulling RHEL UBI9 Python base image..."
docker pull registry.access.redhat.com/ubi9/python-311:latest
Write-Host "✓ UBI9 Python image pulled" -ForegroundColor Green

# Save Ollama image (~1.8 GB)
Write-Host "Saving Ollama image (this may take a few minutes)..."
docker save ollama/ollama:latest -o "$ImagesDir/ollama.tar"
$ollamaSize = (Get-Item "$ImagesDir/ollama.tar").Length / 1GB
Write-Host "✓ Ollama image saved ($([math]::Round($ollamaSize, 2)) GB)" -ForegroundColor Green

# Save API image
Write-Host "Saving API image..."
docker save airgapped-rag-api:latest -o "$ImagesDir/airgapped-rag-api.tar"
$apiSize = (Get-Item "$ImagesDir/airgapped-rag-api.tar").Length / 1MB
Write-Host "✓ API image saved ($([math]::Round($apiSize, 2)) MB)" -ForegroundColor Green

# Save UBI9 Python image
Write-Host "Saving UBI9 Python image..."
docker save registry.access.redhat.com/ubi9/python-311:latest -o "$ImagesDir/ubi9-python.tar"
$ubiSize = (Get-Item "$ImagesDir/ubi9-python.tar").Length / 1MB
Write-Host "✓ UBI9 Python image saved ($([math]::Round($ubiSize, 2)) MB)" -ForegroundColor Green
Write-Host ""

#
# Step 4: Export Ollama models
#
Write-Host "Step 4: Exporting Ollama models..." -ForegroundColor Blue

# Check if Ollama container is running and has models
$ollamaContainer = docker ps --format "{{.Names}}" | Select-String -Pattern "ollama"

if ($ollamaContainer) {
    Write-Host "Found Ollama container: $ollamaContainer" -ForegroundColor Green

    # List models
    Write-Host "Checking installed models..."
    docker exec $ollamaContainer ollama list

    # Determine which volume to use
    $volumeName = docker volume ls --format "{{.Name}}" | Select-String -Pattern "ollama" | Select-Object -First 1

    if ($volumeName) {
        Write-Host "Found Ollama volume: $volumeName" -ForegroundColor Green
        Write-Host "Exporting models (this may take several minutes)..."

        # Export models using Alpine container
        docker run --rm `
            -v "${volumeName}:/models" `
            -v "${PWD}/${ModelsDir}:/backup" `
            alpine tar czf /backup/ollama-models.tar.gz -C /models .

        $modelsSize = (Get-Item "$ModelsDir/ollama-models.tar.gz").Length / 1GB
        Write-Host "✓ Models exported ($([math]::Round($modelsSize, 2)) GB)" -ForegroundColor Green
    } else {
        Write-Host "✗ No Ollama volume found" -ForegroundColor Red
        Write-Host "Please run: docker-compose -f docker-compose.ollama.yml up -d" -ForegroundColor Yellow
        Write-Host "Then pull models with: .\setup-ollama-models.sh or manually" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "✗ No Ollama container running" -ForegroundColor Red
    Write-Host "Please start Ollama first:" -ForegroundColor Yellow
    Write-Host "  docker-compose -f docker-compose.ollama.yml up -d" -ForegroundColor Yellow
    Write-Host "  Then pull models and run this script again" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

#
# Step 5: Copy deployment files
#
Write-Host "Step 5: Copying deployment files..." -ForegroundColor Blue

$filesToCopy = @(
    "docker-compose.airgapped.yml",
    "Dockerfile.airgapped",
    "airgapped_rag.py",
    "requirements.txt",
    "example_usage.py",
    "deploy-rhel9.sh",
    "README_AIRGAPPED_RAG.md",
    "INSTALL_AIRGAPPED.md"
)

foreach ($file in $filesToCopy) {
    if (Test-Path $file) {
        Copy-Item $file $ExportDir/
        Write-Host "✓ Copied $file" -ForegroundColor Green
    } else {
        Write-Host "✗ Missing $file" -ForegroundColor Red
    }
}
Write-Host ""

#
# Step 6: Move Docker images to root of export
#
Write-Host "Step 6: Organizing files..." -ForegroundColor Blue

Move-Item "$ImagesDir/*.tar" $ExportDir/ -Force
Move-Item "$ModelsDir/*.tar.gz" $ExportDir/ -Force
Remove-Item $ImagesDir -Recurse -Force
Remove-Item $ModelsDir -Recurse -Force

Write-Host "✓ Files organized" -ForegroundColor Green
Write-Host ""

#
# Step 7: Create deployment README
#
Write-Host "Step 7: Creating deployment README..." -ForegroundColor Blue

$deploymentReadme = @"
# Air-Gapped RAG API - RHEL9 Deployment Package

This package contains everything needed to deploy the Air-Gapped RAG API on RHEL9.

## Contents

- ``ollama.tar`` - Ollama Docker image (~1.8 GB)
- ``airgapped-rag-api.tar`` - RAG API Docker image
- ``ubi9-python.tar`` - RHEL UBI9 Python base image
- ``ollama-models.tar.gz`` - Pre-downloaded Ollama models (~5 GB)
- ``docker-compose.airgapped.yml`` - Docker Compose configuration
- ``deploy-rhel9.sh`` - Automated deployment script
- ``airgapped_rag.py`` - API application code
- ``requirements.txt`` - Python dependencies
- ``example_usage.py`` - CLI tool for testing
- ``README_AIRGAPPED_RAG.md`` - Complete documentation
- ``INSTALL_AIRGAPPED.md`` - Quick start guide

## Deployment Steps

### 1. Transfer to RHEL9

Transfer this entire directory to your RHEL9 system using one of:
- USB drive
- SCP: ``scp -r rhel9-deployment user@rhel9-server:/home/user/``
- Internal file share

### 2. Run Deployment Script

On the RHEL9 system:

````bash
cd rhel9-deployment
sudo bash deploy-rhel9.sh
````

The script will:
- Load Docker images
- Create data directories
- Restore Ollama models
- Start services
- Verify deployment

### 3. Verify

````bash
# Check health
curl http://localhost:8000/health

# View logs
docker-compose -f docker-compose.airgapped.yml logs -f

# Test with example script
python3 example_usage.py --health
````

## Manual Deployment

If you prefer manual steps, see ``INSTALL_AIRGAPPED.md`` or ``README_AIRGAPPED_RAG.md``.

## Package Information

- Created: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
- Created on: $env:COMPUTERNAME
- Created by: $env:USERNAME
- Total size: $('{0:N2}' -f ((Get-ChildItem $ExportDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB)) GB

## Support

For issues or questions, refer to the documentation files included in this package.

---

**Note**: This package is completely self-contained and requires no internet access for deployment.
"@

$deploymentReadme | Out-File -FilePath "$ExportDir/README_DEPLOYMENT.txt" -Encoding UTF8
Write-Host "✓ Deployment README created" -ForegroundColor Green
Write-Host ""

#
# Step 8: Calculate total size and summary
#
Write-Host "Step 8: Calculating package size..." -ForegroundColor Blue

$totalSize = (Get-ChildItem $ExportDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
$fileCount = (Get-ChildItem $ExportDir -File | Measure-Object).Count

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Export Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Package location: $ExportDir" -ForegroundColor Cyan
Write-Host "Total size: $([math]::Round($totalSize, 2)) GB" -ForegroundColor Cyan
Write-Host "Total files: $fileCount" -ForegroundColor Cyan
Write-Host ""

# List main files
Write-Host "Main files:" -ForegroundColor White
Get-ChildItem $ExportDir -File | ForEach-Object {
    $sizeStr = if ($_.Length -gt 1GB) {
        "$([math]::Round($_.Length / 1GB, 2)) GB"
    } elseif ($_.Length -gt 1MB) {
        "$([math]::Round($_.Length / 1MB, 2)) MB"
    } else {
        "$([math]::Round($_.Length / 1KB, 2)) KB"
    }
    Write-Host "  $($_.Name) - $sizeStr" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Blue
Write-Host "  1. Review the contents in: $ExportDir" -ForegroundColor White
Write-Host "  2. Transfer to RHEL9 system (USB/SCP/Network)" -ForegroundColor White
Write-Host "  3. Run: sudo bash deploy-rhel9.sh" -ForegroundColor White
Write-Host ""

# Optional: Create a ZIP file
$createZip = Read-Host "Create a ZIP file for easier transfer? (y/n)"
if ($createZip -eq 'y') {
    Write-Host "Creating ZIP archive..." -ForegroundColor Blue
    $zipName = "airgapped-rag-rhel9-$(Get-Date -Format 'yyyyMMdd').zip"
    Compress-Archive -Path $ExportDir -DestinationPath $zipName -Force
    $zipSize = (Get-Item $zipName).Length / 1GB
    Write-Host "✓ ZIP created: $zipName ($([math]::Round($zipSize, 2)) GB)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Export completed successfully!" -ForegroundColor Green
