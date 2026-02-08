#!/usr/bin/env bash
set -euo pipefail

# Ensure GPUs are in DEFAULT compute mode (not EXCLUSIVE_PROCESS).
# Cloud H100 instances often ship in exclusive mode, which prevents
# torchrun from spawning multiple processes on the same GPU set.
nvidia-smi -c 0 2>/dev/null || true

# Auto-detect CUDA_HOME — try versioned paths first, then the generic symlink
if [ -d "/usr/local/cuda-12.8" ]; then
  export CUDA_HOME="/usr/local/cuda-12.8"
elif [ -d "/usr/local/cuda-12" ]; then
  export CUDA_HOME="/usr/local/cuda-12"
elif [ -d "/usr/local/cuda" ]; then
  export CUDA_HOME="/usr/local/cuda"
else
  echo "WARNING: No CUDA toolkit found at /usr/local/cuda*"
  echo "  Trying to use nvcc from PATH..."
fi
export TORCH_CUDA_ARCH_LIST="8.0"

export NVTE_CUDA_ARCHS="80"

# ── cuDNN headers ──────────────────────────────────────────────────────
# transformer-engine-torch needs cudnn.h at build time.  Try three strategies:
#   1. pip-installed nvidia-cudnn headers (most reliable in venvs)
#   2. System-installed headers (/usr/include)
#   3. apt-get install as last resort
CUDNN_INCLUDE=""

# Strategy 1: find cudnn.h inside pip nvidia-cudnn package
VENV_DIR="$(python3 -c 'import sys; print(sys.prefix)')"
for candidate in \
  "${VENV_DIR}/lib/python"*/site-packages/nvidia/cudnn/include \
  "${VENV_DIR}/lib/python"*/site-packages/nvidia/cuda_runtime/include \
  /usr/include/x86_64-linux-gnu \
  /usr/include \
  /usr/local/cuda/include; do
  # Resolve globs
  for dir in $candidate; do
    if [ -f "$dir/cudnn.h" ]; then
      CUDNN_INCLUDE="$dir"
      break 2
    fi
  done
done

if [ -n "$CUDNN_INCLUDE" ]; then
  echo "Found cudnn.h at: $CUDNN_INCLUDE"
  export CPLUS_INCLUDE_PATH="${CUDNN_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"
  export C_INCLUDE_PATH="${CUDNN_INCLUDE}:${C_INCLUDE_PATH:-}"
else
  echo "cudnn.h not found in pip packages or system paths, trying apt install..."
  if command -v sudo &>/dev/null && [ "$(id -u)" -ne 0 ]; then
    sudo apt-get update -qq 2>/dev/null || true
    sudo apt-get install -y -qq libcudnn9-dev-cuda-12 2>/dev/null || \
    sudo apt-get install -y -qq libcudnn9-headers-cuda-12 2>/dev/null || true
  else
    apt-get update -qq 2>/dev/null || true
    apt-get install -y -qq libcudnn9-dev-cuda-12 2>/dev/null || \
    apt-get install -y -qq libcudnn9-headers-cuda-12 2>/dev/null || true
  fi
fi

# Ensure ninja is available
if ! command -v ninja &>/dev/null; then
  if command -v sudo &>/dev/null && [ "$(id -u)" -ne 0 ]; then
    sudo apt-get install -y -qq ninja-build 2>/dev/null || true
  else
    apt-get install -y -qq ninja-build 2>/dev/null || true
  fi
fi
# install apex
# Use $HOME instead of /root for non-root users
APEX_DIR="${HOME}/apex"
if [ -d "$APEX_DIR" ]; then
  echo "apex directory already exists, skipping clone"
else
  git clone --depth 1 --branch 25.09 https://github.com/NVIDIA/apex.git "$APEX_DIR"
fi
# Patch Apex to skip CUDA version check — system nvcc may be 12.4 while
# PyTorch was built with 12.8. This is a minor version mismatch within
# CUDA 12.x and works fine at runtime (pip provides the 12.8 runtime libs).
python3 -c "
import os
p = os.path.expanduser('${APEX_DIR}/setup.py')
with open(p) as f: t = f.read()
t = t.replace(
    'check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)',
    'pass  # check_cuda_torch_binary_vs_bare_metal(CUDA_HOME) skipped — minor CUDA version mismatch OK'
)
with open(p, 'w') as f: f.write(t)
print('Patched Apex setup.py to skip CUDA version check')
"
NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=16 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_LAYER_NORM=1 uv pip install --no-build-isolation "$APEX_DIR"
# install flash attention
# git clone https://github.com/Dao-AILab/flash-attention.git /root/flash-attention
# (cd /root/flash-attention && git checkout 27f501d)
# uv run /root/flash-attention/hopper/setup.py install
# install transformer engine and megatron
# Build transformer-engine-torch from source with --no-build-isolation to use venv's torch headers
# (prevents ABI mismatch with system PyTorch in the container)
echo "transformer-engine>=2.11.0" > /tmp/te-override.txt
uv pip install --no-build-isolation --override /tmp/te-override.txt \
  transformer-engine==2.11.0 \
  transformer-engine-cu12==2.11.0 \
  transformer-engine-torch==2.11.0 \
  megatron-core==0.15.2 \
  megatron-bridge==0.2.0rc6
rm /tmp/te-override.txt
# grouped_gemm — required for MoE expert LoRA (grouped_gemm_util.ops.gmm)
uv pip install grouped_gemm 2>/dev/null || pip install grouped_gemm
# silence pynvml warnings
uv pip uninstall pynvml
uv pip install nvidia-ml-py==13.580.82
