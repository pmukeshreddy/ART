#!/usr/bin/env bash
# =============================================================================
# Setup script for SGLang + Megatron vs vLLM + Megatron benchmark
#
# Creates separate Python environments for SGLang (to avoid conflicts with
# vLLM which is already installed in the ART environment).
#
# Prerequisites:
#   - CUDA 12.x installed
#   - Python 3.10+ available
#   - nvidia-smi working
#   - uv package manager installed
#
# Usage:
#   bash benchmarks/sglang_vs_vllm/setup_environments.sh
#
# After setup:
#   python benchmarks/sglang_vs_vllm/run_benchmark.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# =============================================================================
# Helper: uv pip install that works with or without a venv
# =============================================================================
# If VIRTUAL_ENV is set or a .venv exists, uv picks it up automatically.
# Otherwise (bare system python) we need --system so uv doesn't refuse.

uv_pip_install() {
    if [ -n "${VIRTUAL_ENV:-}" ] || [ -d "$PROJECT_ROOT/.venv" ]; then
        uv pip install "$@"
    else
        uv pip install --system "$@"
    fi
}

# =============================================================================
# 1. Validate prerequisites
# =============================================================================

info "Checking prerequisites..."

# Check CUDA
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. CUDA drivers required."
fi
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
info "NVIDIA driver version: $CUDA_VERSION"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
info "GPUs detected: $GPU_COUNT"

# Check Python
if ! command -v python3 &>/dev/null; then
    error "python3 not found"
fi
PYTHON_VERSION=$(python3 --version 2>&1)
info "Python: $PYTHON_VERSION"

# Check uv
if ! command -v uv &>/dev/null; then
    warn "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
info "uv: $(uv --version)"

success "Prerequisites OK"

# =============================================================================
# 2. Verify vLLM environment (existing ART environment)
# =============================================================================

info ""
info "=== Checking vLLM (ART) environment ==="

cd "$PROJECT_ROOT"

# Check if vLLM is installed in the current environment
if python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null; then
    VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)")
    success "vLLM $VLLM_VERSION is already installed in ART environment"
else
    info "vLLM not found. Installing with ART backend extras..."
    uv_pip_install vllm
    success "vLLM installed"
fi

# Check if Megatron deps are available
if python3 -c "import megatron.bridge" 2>/dev/null; then
    success "Megatron bridge is installed"
else
    info "Megatron bridge not found (will be installed on first training run)"
    info "  You can install manually: bash src/art/megatron/setup.sh"
fi

# =============================================================================
# 3. Create SGLang environment
# =============================================================================

info ""
info "=== Setting up SGLang environment ==="

SGLANG_ENV="$HOME/.venvs/sglang-bench"

if [ -d "$SGLANG_ENV" ] && "$SGLANG_ENV/bin/python" -c "import sglang" 2>/dev/null; then
    SGLANG_VERSION=$("$SGLANG_ENV/bin/python" -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
    success "SGLang environment already exists at $SGLANG_ENV (version: $SGLANG_VERSION)"
    info "  To recreate: rm -rf $SGLANG_ENV && bash $0"
else
    info "Creating SGLang virtual environment at $SGLANG_ENV..."
    mkdir -p "$(dirname "$SGLANG_ENV")"

    # Create venv
    uv venv "$SGLANG_ENV" --python python3

    info "Installing SGLang and dependencies..."

    # SGLang gets its own venv — use --python to target it explicitly
    uv pip install --python "$SGLANG_ENV/bin/python" \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124

    # Install SGLang with all extras (server, router, all backends)
    # Ref: https://docs.sglang.ai/start/install.html
    uv pip install --python "$SGLANG_ENV/bin/python" \
        "sglang[all]>=0.4.6.post1"

    # Install additional dependencies for benchmark + bench_serving.py
    uv pip install --python "$SGLANG_ENV/bin/python" \
        aiohttp openai numpy tqdm datasets

    # Verify installation
    if "$SGLANG_ENV/bin/python" -c "import sglang; print(f'SGLang {sglang.__version__}')" 2>/dev/null; then
        SGLANG_VERSION=$("$SGLANG_ENV/bin/python" -c "import sglang; print(sglang.__version__)")
        success "SGLang $SGLANG_VERSION installed successfully"
    else
        error "SGLang installation failed. Check logs above."
    fi
fi

# =============================================================================
# 4. Verify Unsloth environment (for --backends unsloth)
# =============================================================================

info ""
info "=== Checking Unsloth (MoE training) environment ==="

cd "$PROJECT_ROOT"

# Unsloth runs in the same ART/system environment (uses unsloth + unsloth-zoo packages)
# Note: MoE nn.Parameter doesn't support bitsandbytes 4-bit yet → BF16/FP16 LoRA

# Step 1: Install Unsloth packages
# Check if unsloth package files exist (don't try import — it may crash on
# vllm._C ABI mismatch before we've had a chance to mock it)
if python3 -c "import importlib.metadata; print(importlib.metadata.version('unsloth'))" 2>/dev/null; then
    UNSLOTH_VERSION=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('unsloth'))")
    success "Unsloth $UNSLOTH_VERSION package is installed"
else
    info "Unsloth not found. Installing..."
    uv_pip_install --upgrade unsloth unsloth_zoo
fi

# Step 2: Force-upgrade transformers and trl to versions required for MoE
# Unsloth may pull in older versions (e.g. transformers 4.x) but MoE fused
# gate_up_proj layers and torch._grouped_mm require transformers>=5.0.0.
# trl>=0.27.1 is needed for compatibility with transformers v5.
info "Ensuring transformers>=5.0.0 and trl>=0.27.1 for MoE support..."
NEED_TF_UPGRADE=$(python3 -c "
import importlib.metadata as meta
v = meta.version('transformers')
print('yes' if tuple(int(x) for x in v.split('.')[:2]) < (5, 0) else 'no')
" 2>/dev/null || echo "yes")

NEED_TRL_UPGRADE=$(python3 -c "
import importlib.metadata as meta
v = meta.version('trl')
parts = v.split('.')
print('yes' if (int(parts[0]), int(parts[1])) < (0, 27) else 'no')
" 2>/dev/null || echo "yes")

if [ "$NEED_TF_UPGRADE" = "yes" ] || [ "$NEED_TRL_UPGRADE" = "yes" ]; then
    info "Upgrading: transformers>=5.0.0 trl>=0.27.1 ..."
    uv_pip_install --upgrade "transformers>=5.0.0" "trl>=0.27.1"
fi

# Unsloth 2026.2.x blocks datasets>=4.5.0 (causes recursion errors).
# The transformers v5 upgrade may have pulled in a newer datasets version.
NEED_DS_DOWNGRADE=$(python3 -c "
import importlib.metadata as meta
v = meta.version('datasets')
parts = [int(x) for x in v.split('.')[:2]]
print('yes' if parts[0] > 4 or (parts[0] == 4 and parts[1] >= 5) else 'no')
" 2>/dev/null || echo "no")

if [ "$NEED_DS_DOWNGRADE" = "yes" ]; then
    info "Downgrading datasets to 4.3.0 (Unsloth requires <4.5.0)..."
    uv_pip_install "datasets==4.3.0"
fi

# Step 3: Verify Unsloth import (show errors, don't swallow them)
# Unsloth + unsloth_zoo deeply import vLLM internals at load time.
# If vLLM's C extension has an ABI mismatch with PyTorch (common on cloud
# GPU images), we mock vllm._C and unsloth_zoo.vllm_utils before importing.
# This is safe because our Unsloth backend uses SGLang, not vLLM.
info "Verifying Unsloth import..."
if python3 -c "
import sys, types

# Check if vLLM's C extension works
vllm_ok = False
try:
    import vllm._C
    vllm_ok = True
except (ImportError, OSError, AttributeError):
    pass

if not vllm_ok:
    # Mock vllm._C (shallow imports)
    sys.modules['vllm._C'] = types.ModuleType('vllm._C')
    # Mock unsloth_zoo.vllm_utils (prevents deep vllm.model_executor imports)
    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            return lambda *a, **kw: None
    sys.modules['unsloth_zoo.vllm_utils'] = _Stub('unsloth_zoo.vllm_utils')
    print('  (mocked vllm internals — ABI mismatch with PyTorch, using SGLang)')

import unsloth
print(f'Unsloth {unsloth.__version__}')
"; then
    success "Unsloth ready"
else
    warn "Unsloth import failed (see error above). Unsloth backend may not work."
    warn "  Try: pip install --upgrade unsloth unsloth_zoo transformers>=5.0.0 trl>=0.27.1"
fi

# Show final versions
python3 -c "
import importlib.metadata as meta
for pkg in ['unsloth', 'unsloth-zoo', 'transformers', 'trl', 'torch', 'triton']:
    try:
        print(f'  {pkg}: {meta.version(pkg)}')
    except meta.PackageNotFoundError:
        print(f'  {pkg}: not installed')
" 2>/dev/null || true

# Check MoE backend support
python3 -c "
import torch
print(f'  MoE backend auto-detection:')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    if 'H100' in name or 'B200' in name:
        print(f'    GPU: {name} → grouped_mm (optimal)')
    elif 'A100' in name:
        print(f'    GPU: {name} → unsloth_triton (optimal)')
    else:
        print(f'    GPU: {name} → native_torch (fallback)')
else:
    print('    No CUDA GPU available')
" 2>/dev/null || true

# =============================================================================
# 5. Install benchmark dependencies in ART environment
# =============================================================================

info ""
info "=== Installing benchmark dependencies in ART environment ==="

cd "$PROJECT_ROOT"
uv_pip_install aiohttp 2>/dev/null || pip install aiohttp

success "Benchmark dependencies installed"

# =============================================================================
# 6. Verify everything works
# =============================================================================

info ""
info "=== Verification ==="

# vLLM check — find the right python (venv or system)
if [ -n "${VIRTUAL_ENV:-}" ]; then
    VLLM_PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    VLLM_PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    VLLM_PYTHON="python3"
fi
info "vLLM Python: $VLLM_PYTHON"
"$VLLM_PYTHON" -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count: {torch.cuda.device_count()}')
try:
    import vllm
    print(f'  vLLM: {vllm.__version__}')
except ImportError:
    print('  vLLM: not installed (install separately or skip --backends vllm)')
" 2>/dev/null || warn "vLLM verification failed (non-fatal)"

# SGLang check
SGLANG_PYTHON="$SGLANG_ENV/bin/python"
info "SGLang Python: $SGLANG_PYTHON"
"$SGLANG_PYTHON" -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count: {torch.cuda.device_count()}')
import sglang
print(f'  SGLang: {sglang.__version__}')
" 2>/dev/null || warn "SGLang verification failed (non-fatal)"

# =============================================================================
# 7. Print usage instructions
# =============================================================================

info ""
success "=== Setup Complete ==="
echo ""
echo "Environment paths:"
echo "  vLLM Python:   $VLLM_PYTHON"
echo "  SGLang Python: $SGLANG_PYTHON"
echo ""
echo "Run the benchmark:"
echo ""
echo "  # vLLM + SGLang comparison (original)"
echo "  python benchmarks/sglang_vs_vllm/run_benchmark.py \\"
echo "    --sglang-python $SGLANG_PYTHON \\"
echo "    --model Qwen/Qwen2.5-7B-Instruct \\"
echo "    --dataset gsm8k \\"
echo "    --num-steps 3 \\"
echo "    --num-rollouts 16 \\"
echo "    --concurrency 8 \\"
echo "    --tp 2"
echo ""
echo "  # Unsloth MoE + SGLang (new)"
echo "  python benchmarks/sglang_vs_vllm/run_benchmark.py \\"
echo "    --sglang-python $SGLANG_PYTHON \\"
echo "    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\"
echo "    --backends unsloth \\"
echo "    --num-steps 3 \\"
echo "    --num-rollouts 16 \\"
echo "    --unsloth-lora-rank 16 \\"
echo "    --tp 2"
echo ""
echo "  # All three backends"
echo "  python benchmarks/sglang_vs_vllm/run_benchmark.py \\"
echo "    --sglang-python $SGLANG_PYTHON \\"
echo "    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\"
echo "    --backends vllm sglang unsloth \\"
echo "    --num-steps 3 \\"
echo "    --num-rollouts 16 \\"
echo "    --tp 2"
echo ""
echo "  # Unsloth MoE backend options:"
echo "  #   --unsloth-lora-rank 16     Higher rank for MoE (default: 16, Unsloth recommends 16-64)"
echo "  #   --unsloth-moe-backend auto  auto|grouped_mm|unsloth_triton|native_torch"
echo "  #   --unsloth-port 8300        SGLang inference port for Unsloth backend"
echo ""
