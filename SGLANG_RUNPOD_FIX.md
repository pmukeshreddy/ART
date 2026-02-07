# SGLang RunPod H100 Setup Guide

## Problem Summary

When running SGLang on RunPod H100 instances, you may encounter two critical errors:

1. **`libnuma.so.1: cannot open shared object file`** - Missing system library
2. **`cudaGetDriverEntryPointByVersion undefined symbol`** - PyTorch/CUDA version mismatch

## Root Causes

### Error 1: Missing libnuma
- `sgl_kernel` (SGLang's H100 CUDA kernels) requires the NUMA library
- RunPod containers may not have it installed by default
- Cannot be fixed with environment variables alone

### Error 2: PyTorch CUDA Mismatch
- PyTorch was compiled for CUDA 12.8+ but RunPod toolkit is CUDA 12.4
- The symbol `cudaGetDriverEntryPointByVersion` doesn't exist in older CUDA runtimes
- **This is NOT a library path issue** - it's an ABI incompatibility

## Complete Fix (Run on RunPod Server)

```bash
cd ~/ART

# ============================================================
# Step 1: Install missing system library
# ============================================================
apt-get update
apt-get install -y libnuma1 libnuma-dev

# Update library cache
ldconfig

# Verify installation
ldconfig -p | grep libnuma
# Should show: libnuma.so.1 => /usr/lib/x86_64-linux-gnu/libnuma.so.1

# ============================================================
# Step 2: Check your CUDA versions
# ============================================================
# Driver CUDA version
nvidia-smi | grep "CUDA Version"
# Shows: CUDA Version: 12.9 (driver compatibility)

# Toolkit CUDA version
nvcc --version
# Shows: release 12.4, V12.4.131 (actual toolkit)

# Use the TOOLKIT version for PyTorch, not the driver version

# ============================================================
# Step 3: Rebuild SGLang environment with correct CUDA
# ============================================================
# Remove existing broken environment
rm -rf .venv-sglang-server

# Create fresh Python 3.11 environment
python3.11 -m venv .venv-sglang-server
source .venv-sglang-server/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch for CUDA 12.4 (matching nvcc)
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install SGLang (uses the PyTorch we just installed)
pip install "sglang[srt]==0.5.8.post1"

# ============================================================
# Step 4: Verify installation
# ============================================================
# Test PyTorch CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Test sgl_kernel (should not error now)
python -c "from sgl_kernel import common_ops; print('✓ sgl_kernel loaded successfully')"

# Test SGLang
python -c "import sglang; print(f'✓ SGLang {sglang.__version__} ready')"

deactivate

# ============================================================
# Step 5: Run benchmark
# ============================================================
source .venv/bin/activate

python scripts/benchmark_sglang_vs_vllm_megatron.py \
  --backend sglang \
  --scenario C \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --enable-training \
  --num-samples 50 \
  --preserve-cache \
  --sglang-gpu-ids 0 \
  --training-gpu-ids 1,2,3 \
  --num-iterations 5 \
  --sglang-cache-method sleep_wake
```

## What the Code Changes Do

The `sglang_service.py` changes add defensive library path handling, but **they cannot fix the root issues**:

1. **LD_LIBRARY_PATH injection** - Helps subprocess find libraries, but only if they exist
2. **LD_PRELOAD for libnuma** - Only works if libnuma is installed
3. **CUDA_HOME** - Doesn't change PyTorch's compiled CUDA version

## Why Previous Attempts Failed

- **Installing libnuma** alone isn't enough if PyTorch has wrong CUDA version
- **Setting environment variables** doesn't change PyTorch's compiled-in CUDA ABI
- **Using wrong PyTorch** (2.9.1 for CUDA 12.8) when system has CUDA 12.4

## Verification Commands

```bash
# Check if libnuma is installed
ls -la /usr/lib/x86_64-linux-gnu/libnuma.so*

# Check if ldconfig knows about it
ldconfig -p | grep libnuma

# Check PyTorch CUDA version
source .venv-sglang-server/bin/activate
python -c "import torch; print(torch.version.cuda)"
# Should show: 12.4 (matching your nvcc)

# Check system CUDA
nvcc --version
# Should show: release 12.4
```

## Reference

- SGLang requires specific PyTorch builds that match system CUDA
- H100 (SM90) support in `sgl_kernel` requires `libnuma` for multi-GPU NUMA topology
- RunPod instances may have CUDA driver 12.9 but toolkit 12.4 - **use toolkit version**

## Troubleshooting

**Still getting libnuma error?**
```bash
# Force reinstall
apt-get install --reinstall libnuma1
ldconfig
```

**Still getting CUDA symbol error?**
```bash
# Check what CUDA your PyTorch was built for
python -c "import torch; print(torch.version.cuda)"

# If it says 12.8 or 12.9, reinstall with 12.4:
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
pip install "sglang[srt]>=0.5.5" --force-reinstall
```

**Out of disk space during reinstall?**
```bash
# Clean pip cache
pip cache purge

# Remove old packages
apt-get autoremove
apt-get clean
```
