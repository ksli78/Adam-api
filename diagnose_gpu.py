"""
Quick GPU Diagnostic Script for Windows
Checks why PyTorch is not detecting CUDA/GPU
"""

import sys
import subprocess

print("=" * 60)
print("GPU DETECTION DIAGNOSTIC")
print("=" * 60)
print()

# Check 1: NVIDIA Driver
print("1. Checking NVIDIA Driver...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ NVIDIA Driver installed")
        print(result.stdout.split('\n')[0:3])  # Show first 3 lines

        # Extract CUDA version from nvidia-smi
        for line in result.stdout.split('\n'):
            if 'CUDA Version' in line:
                print(f"   Driver CUDA Version: {line.split('CUDA Version:')[1].strip().split()[0]}")
    else:
        print("❌ nvidia-smi failed - NVIDIA driver may not be installed")
        print("   Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
except FileNotFoundError:
    print("❌ nvidia-smi not found - NVIDIA driver not installed")
    print("   Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
except Exception as e:
    print(f"❌ Error checking nvidia-smi: {e}")
print()

# Check 2: PyTorch Installation
print("2. Checking PyTorch...")
try:
    import torch
    print(f"✅ PyTorch installed: {torch.__version__}")

    # Check if it's CPU or CUDA version
    if '+cu' in torch.__version__:
        print(f"   ✅ CUDA version detected in PyTorch: {torch.__version__}")
    else:
        print(f"   ❌ CPU-only version installed: {torch.__version__}")
        print("   Need to install CUDA version!")
except ImportError:
    print("❌ PyTorch not installed")
    sys.exit(1)
print()

# Check 3: CUDA Availability in PyTorch
print("3. Checking CUDA availability in PyTorch...")
try:
    import torch
    print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   ✅ CUDA is available!")
        print(f"   CUDA version in PyTorch: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   ❌ CUDA not available in PyTorch")

        # Diagnose why
        print()
        print("   Possible reasons:")
        print("   a) CPU-only PyTorch installed (check version above)")
        print("   b) CUDA driver version incompatible with PyTorch CUDA version")
        print("   c) CUDA toolkit not installed")

        # Check cuDNN
        print()
        print("   Checking cuDNN...")
        if hasattr(torch.backends, 'cudnn'):
            print(f"      cuDNN available: {torch.backends.cudnn.is_available()}")

except Exception as e:
    print(f"❌ Error: {e}")
print()

# Check 4: Environment Variables
print("4. Checking CUDA environment variables...")
import os
cuda_path = os.environ.get('CUDA_PATH')
cuda_home = os.environ.get('CUDA_HOME')
path = os.environ.get('PATH', '')

if cuda_path:
    print(f"   CUDA_PATH: {cuda_path}")
else:
    print("   CUDA_PATH: Not set")

if cuda_home:
    print(f"   CUDA_HOME: {cuda_home}")
else:
    print("   CUDA_HOME: Not set")

if 'cuda' in path.lower():
    print("   ✅ CUDA found in PATH")
else:
    print("   ⚠️  CUDA not found in PATH")
print()

# Summary and Recommendations
print("=" * 60)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 60)

try:
    import torch

    if torch.cuda.is_available():
        print("✅ Everything looks good! GPU should work.")
    else:
        print("❌ GPU not detected. Likely causes:")
        print()

        if '+cu' not in torch.__version__:
            print("PROBLEM: CPU-only PyTorch installed")
            print()
            print("SOLUTION: Reinstall PyTorch with CUDA support:")
            print("  pip uninstall torch torchvision torchaudio")
            print("  pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118")
            print()
        else:
            print("PROBLEM: CUDA version mismatch or driver issue")
            print()
            print("SOLUTIONS:")
            print("1. Update NVIDIA drivers to latest version")
            print("2. Install CUDA Toolkit 11.8 (matches PyTorch)")
            print("   Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive")
            print("3. Reinstall PyTorch:")
            print("   pip install torch==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --force-reinstall")
            print()

except Exception as e:
    print(f"Error during summary: {e}")

print()
print("For more help, see:")
print("  - PyTorch Installation: https://pytorch.org/get-started/locally/")
print("  - NVIDIA Drivers: https://www.nvidia.com/Download/index.aspx")
print("=" * 60)
