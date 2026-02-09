# SGLang + Megatron vs vLLM + Megatron Benchmark

End-to-end benchmarking suite that compares **SGLang + Megatron** against **vLLM + Megatron** for RL training workloads (GRPO with LoRA adapters).

## What This Measures

During RL training, the inference engine handles the rollout phase (generating completions) while Megatron handles the training phase. This benchmark measures:

| Metric | Description |
|--------|-------------|
| **Rollout Throughput** | Tokens/second during inference generation |
| **Time-to-First-Token (TTFT)** | Latency to start generating the first token |
| **Transition Overhead** | Time to sleep/stop inference + wake/restart after training |
| **Training Time** | Megatron forward+backward+optimizer step (control variable) |
| **Total Step Time** | End-to-end wall-clock time per training step |
| **GPU Memory** | Memory usage during inference vs training phases |

## Architecture

Both backends share the **exact same** Megatron training loop (`src/art/megatron/train.py`). The only difference is the inference engine:

```
vLLM + Megatron:                       SGLang + Megatron:
┌──────────────────────┐              ┌──────────────────────┐
│   vLLM (in-process)  │              │ SGLang (subprocess)  │
│   OpenAI API :8100   │              │   OpenAI API :8200   │
└─────────┬────────────┘              └─────────┬────────────┘
          │ sleep/wake                          │ stop/restart
          ▼                                     ▼
┌──────────────────────┐              ┌──────────────────────┐
│  Megatron Training   │              │  Megatron Training   │
│  (torchrun, LoRA)    │              │  (torchrun, LoRA)    │
└──────────────────────┘              └──────────────────────┘
```

**vLLM** uses its in-process sleep/wake mechanism to free GPU memory:
- `do_sleep(level=2)` → offloads weights, discards KV cache
- `do_wake_up()` → reloads weights, reinitializes cache

**SGLang** runs as a separate server process:
- Server is terminated (`SIGTERM`) to free GPU memory
- Server is restarted with new LoRA after training completes

This is a **fair comparison** because both achieve the same goal (free GPU memory for training), and both have real-world overhead that is measured.

## Prerequisites

- NVIDIA GPU(s) with CUDA 12.x
- Python 3.10+
- `uv` package manager
- ART project set up with `uv sync`

## Quick Start

```bash
# 1. Setup environments (creates separate SGLang venv)
bash benchmarks/sglang_vs_vllm/setup_environments.sh

# 2. Run inference benchmark (fastest, no training)
python benchmarks/sglang_vs_vllm/run_benchmark.py \
    --sglang-python ~/.venvs/sglang-bench/bin/python

# 3. View results
cat benchmark_results/benchmark_report.txt
```

## Benchmark Modes

### Inference Mode (default)

Tests inference throughput, TTFT, and transition overhead with simulated training:

```bash
python benchmarks/sglang_vs_vllm/run_benchmark.py \
    --mode inference \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-steps 5 \
    --num-rollouts 32 \
    --concurrency 16 \
    --sglang-python ~/.venvs/sglang-bench/bin/python \
    --output results/inference_bench
```

### Training Mode

Runs actual Megatron training with LoRA updates:

```bash
python benchmarks/sglang_vs_vllm/run_benchmark.py \
    --mode training \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-steps 3 \
    --num-rollouts 16 \
    --sglang-python ~/.venvs/sglang-bench/bin/python \
    --output results/training_bench
```

### Single Backend

```bash
# Only test SGLang
python benchmarks/sglang_vs_vllm/run_benchmark.py --backends sglang

# Only test vLLM
python benchmarks/sglang_vs_vllm/run_benchmark.py --backends vllm
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `inference` | `inference` or `training` |
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID |
| `--backends` | `vllm sglang` | Which backends to test |
| `--num-steps` | `3` | Training steps to simulate |
| `--num-rollouts` | `16` | Rollout requests per step |
| `--concurrency` | `8` | Max concurrent requests |
| `--max-input-tokens` | `1024` | Approx input tokens per prompt |
| `--max-output-tokens` | `1024` | Max output tokens per request |
| `--output` | `benchmark_results` | Results directory |
| `--sglang-python` | auto | Path to SGLang Python |
| `--vllm-port` | `8100` | vLLM server port |
| `--sglang-port` | `8200` | SGLang server port |
| `--tp` | `0` (auto) | Tensor parallel size |
| `--gpu-memory-utilization` | `0.85` | GPU memory fraction |
| `--skip-preflight` | false | Skip environment checks |

## Output

Results are saved to the output directory:

```
benchmark_results/
├── benchmark_report.txt        # Human-readable comparison
├── benchmark_combined.json     # Machine-readable combined metrics
├── vllm_metrics.json          # vLLM detailed metrics
├── sglang_metrics.json        # SGLang detailed metrics
└── sglang_workdir/            # SGLang working directory
    └── logs/
        └── sglang.log         # SGLang server logs
```

### Sample Report

```
================================================================================
  BENCHMARK REPORT: SGLang + Megatron vs vLLM + Megatron
================================================================================

Model: Qwen/Qwen2.5-7B-Instruct
Training Steps: 5

--------------------------------------------------------------------------------
Metric                                         vLLM       SGLang         Diff
--------------------------------------------------------------------------------

=== Rollout (Inference) Performance ===
Avg Rollout Time (s)                          45.23        32.17       +28.9%
Avg Throughput (tok/s)                      2370.5       3120.8       +31.6%
Avg TTFT (s)                                0.3180       0.1360       +57.2%

=== Transition Overhead ===
Avg Transition Overhead (s)                    4.50         8.20       -82.2%

=== End-to-End ===
Avg Step Time (s)                             51.73        42.37       +18.1%

================================================================================
  VERDICT
================================================================================

  SGLang inference throughput is 31.6% HIGHER
  SGLang TTFT is 57.2% LOWER (better)
  SGLang total step time is 18.1% FASTER

  >>> WINNER: SGLang + Megatron <<<
```

## Things That Could Go Wrong

This benchmark handles the following edge cases:

### Environment Issues
- **Conflicting dependencies**: SGLang and vLLM have different PyTorch/CUDA requirements → separate virtual environments
- **Missing Megatron**: Auto-detected; `setup.sh` is called on first use
- **Port conflicts**: Pre-flight check detects occupied ports

### GPU Memory Issues
- **OOM during inference**: Configurable `--gpu-memory-utilization`
- **Memory leaks between runs**: Full GPU cleanup (`pkill`) between backends
- **Fragmentation**: Fresh server restart between SGLang steps

### Server Issues
- **Startup failures**: Timeout with detailed error messages and log file
- **Hung processes**: Process group kill (`SIGKILL` fallback)
- **Stale processes**: Port-based cleanup before server start

### Benchmark Validity
- **Cold vs warm cache**: Warmup requests before timing
- **Statistical noise**: Multiple steps, per-step breakdown
- **Unfair comparison**: Same prompts, same model, same hardware, same training config
- **No cheating**: vLLM uses existing ART code unmodified; SGLang uses standard server launch

## File Structure

```
benchmarks/sglang_vs_vllm/
├── __init__.py                    # Package
├── config.py                      # Benchmark configuration
├── sglang_server.py              # SGLang server lifecycle management
├── sglang_megatron_service.py    # SGLang + Megatron service (mirrors MegatronService)
├── sglang_megatron_backend.py    # SGLang backend (extends LocalBackend)
├── metrics_collector.py          # Metrics collection and comparison reporting
├── run_benchmark.py              # Main benchmark orchestrator
├── setup_environments.sh         # Environment setup script
└── README.md                     # This file
```

## References

- [SGLang](https://github.com/sgl-project/sglang) — High-throughput LLM serving
- [RLinf](https://github.com/RLinf/RLinf) — Megatron + SGLang RL infrastructure
- [veRL](https://github.com/verl-project/verl) — ByteDance RL training framework
- [ART](https://github.com/OpenPipe/ART) — Agent Reinforcement Trainer (this project)
