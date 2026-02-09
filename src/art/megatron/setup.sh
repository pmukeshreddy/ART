#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="/usr/local/cuda-12.8"
export TORCH_CUDA_ARCH_LIST="9.0"

# ------------------------------------------------------------------
# 1. System packages — skip if already installed (~1-2 min saved)
# ------------------------------------------------------------------
if dpkg -s libcudnn9-headers-cuda-12 ninja-build >/dev/null 2>&1; then
  echo "[setup.sh] libcudnn9-headers-cuda-12 & ninja-build already installed, skipping apt"
else
  apt-get update
  apt-get install -y libcudnn9-headers-cuda-12 ninja-build
fi

# ------------------------------------------------------------------
# 2. Apex — skip if already importable (~5-10 min saved)
# ------------------------------------------------------------------
if python -c "import apex" 2>/dev/null; then
  echo "[setup.sh] apex already installed, skipping build"
else
  if [ -d /root/apex ]; then
    echo "[setup.sh] apex directory already exists, skipping clone"
  else
    git clone --depth 1 --branch 25.09 https://github.com/NVIDIA/apex.git /root/apex
  fi
  NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=16 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_FAST_LAYER_NORM=1 uv pip install --no-build-isolation /root/apex
fi

# ------------------------------------------------------------------
# 3. Transformer Engine + Megatron — skip if already importable (~3-5 min saved)
# ------------------------------------------------------------------
# install flash attention
# git clone https://github.com/Dao-AILab/flash-attention.git /root/flash-attention
# (cd /root/flash-attention && git checkout 27f501d)
# uv run /root/flash-attention/hopper/setup.py install
# Build transformer-engine-torch from source with --no-build-isolation to use venv's torch headers
# (prevents ABI mismatch with system PyTorch in the container)
if python -c "import megatron.bridge; import transformer_engine" 2>/dev/null; then
  echo "[setup.sh] transformer-engine & megatron already installed, skipping"
else
  echo "transformer-engine>=2.11.0" > /tmp/te-override.txt
  uv pip install --no-build-isolation --override /tmp/te-override.txt \
    transformer-engine==2.11.0 \
    transformer-engine-cu12==2.11.0 \
    transformer-engine-torch==2.11.0 \
    megatron-core==0.15.2 \
    megatron-bridge==0.2.0rc6
  rm /tmp/te-override.txt
fi

# ------------------------------------------------------------------
# 4. pynvml fix — skip if already correct (~10s saved)
# ------------------------------------------------------------------
if python -c "import pynvml" 2>/dev/null; then
  uv pip uninstall pynvml
  uv pip install nvidia-ml-py==13.580.82
elif python -c "from nvidia_ml_py import nvmlInit; nvmlInit()" 2>/dev/null; then
  echo "[setup.sh] nvidia-ml-py already installed, skipping"
else
  uv pip install nvidia-ml-py==13.580.82
fi
