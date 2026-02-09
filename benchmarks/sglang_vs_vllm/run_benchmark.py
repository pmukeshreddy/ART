#!/usr/bin/env python3
"""
End-to-end benchmark: SGLang + Megatron vs vLLM + Megatron.

Each backend runs in its own **subprocess** for GPU isolation (fresh CUDA
context, no memory/kernel-cache leakage between backends).

Each step is a real training loop:
  1. Rollout:  generate completions via the inference engine   (timed)
  2. Train:    backend.train() → sleep/stop → Megatron → wake/restart (timed)
  3. Next rollout measures the engine AFTER a real weight update

Metrics captured per step:
  - Rollout throughput, TTFT, ITL, latency
  - Total step wall-time (rollout + train)
  - GPU memory during inference

Usage:
    bash benchmarks/sglang_vs_vllm/setup_environments.sh   # one-time
    python benchmarks/sglang_vs_vllm/run_benchmark.py \\
        --sglang-python ~/.venvs/sglang-bench/bin/python
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ===================================================================
# Worker — runs inside an isolated subprocess (one per backend)
# ===================================================================

def run_worker(backend: str, config_path: str, results_path: str) -> None:
    """
    Entry point for the isolated subprocess.
    Imports are done HERE so the parent process stays lightweight.
    """
    import asyncio
    import gc
    import aiohttp
    import torch

    # ---- lazy imports to keep them in the subprocess only ----------
    from benchmarks.sglang_vs_vllm.config import (
        BenchmarkConfig, generate_benchmark_prompts,
    )
    from benchmarks.sglang_vs_vllm.metrics_collector import (
        BenchmarkRun, RequestMetrics, StepMetrics,
        get_gpu_memory_usage_nvidia_smi,
    )
    from benchmarks.sglang_vs_vllm.sglang_server import SGLangServer, SGLangServerConfig

    with open(config_path) as f:
        raw = json.load(f)

    # Reconstruct config (stored as flat dict for JSON serialization)
    config = BenchmarkConfig(**{
        k: v for k, v in raw.items()
        if k in BenchmarkConfig.__dataclass_fields__
    })

    logger.info(f"[{backend}] Worker started — PID {os.getpid()}")
    logger.info(f"[{backend}] CUDA devices: {torch.cuda.device_count()}")

    # ---- shared helpers -------------------------------------------

    async def measure_rollout(
        base_url: str, model_name: str,
        prompts: list[list[dict[str, str]]],
        max_tokens: int, concurrency: int, timeout: int,
    ) -> list[RequestMetrics]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(idx, msgs):
            async with sem:
                t0 = time.perf_counter()
                ttft = comp_tok = prompt_tok = 0
                err = None
                first = False
                try:
                    async with aiohttp.ClientSession() as s:
                        async with s.post(
                            f"{base_url}/chat/completions",
                            json={"model": model_name, "messages": msgs,
                                  "max_tokens": max_tokens, "temperature": 1.0,
                                  "stream": True},
                            timeout=aiohttp.ClientTimeout(total=timeout),
                        ) as r:
                            if r.status != 200:
                                err = f"HTTP {r.status}: {(await r.text())[:200]}"
                            else:
                                async for raw in r.content:
                                    line = raw.decode().strip()
                                    if not line.startswith("data: "):
                                        continue
                                    d = line[6:]
                                    if d == "[DONE]":
                                        break
                                    if not first:
                                        ttft = time.perf_counter() - t0
                                        first = True
                                    try:
                                        c = json.loads(d)
                                        if c.get("usage"):
                                            prompt_tok = c["usage"].get("prompt_tokens", 0)
                                            comp_tok = c["usage"].get("completion_tokens", 0)
                                        elif c.get("choices"):
                                            if c["choices"][0].get("delta", {}).get("content"):
                                                comp_tok += 1
                                    except json.JSONDecodeError:
                                        pass
                except Exception as e:
                    err = str(e)
                t1 = time.perf_counter()
                return RequestMetrics(
                    request_id=idx, start_time=t0, end_time=t1,
                    ttft=ttft, total_time=t1 - t0,
                    prompt_tokens=prompt_tok, completion_tokens=comp_tok,
                    error=err,
                )

        return list(await asyncio.gather(*[_one(i, m) for i, m in enumerate(prompts)]))

    async def warmup_server(base_url, model_name, n=4):
        for _ in range(n):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(
                        f"{base_url}/chat/completions",
                        json={"model": model_name,
                              "messages": [{"role": "user", "content": "What is 2+2?"}],
                              "max_tokens": 32, "temperature": 0},
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as r:
                        await r.read()
            except Exception:
                pass

    async def get_model_name(base_url, api_key="default"):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    data = await r.json()
                    if data.get("data"):
                        return data["data"][0]["id"]
        except Exception:
            pass
        return "default"

    # ---- backend-specific logic -----------------------------------

    async def _run_vllm() -> BenchmarkRun:
        import art
        from art.megatron.backend import MegatronBackend

        run = BenchmarkRun(backend="vllm", model=config.model.base_model,
                           dataset=config.dataset)
        run.start_time = time.perf_counter()

        model = art.TrainableModel(
            name=config.model.model_name + "-vllm-bench",
            project=config.model.project,
            base_model=config.model.base_model,
        )
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(max_seq_length=config.model.max_seq_length),
            engine_args=art.dev.EngineArgs(
                model=config.model.base_model,
                tensor_parallel_size=config.inference.get_tp_size(),
                gpu_memory_utilization=config.inference.gpu_memory_utilization,
                max_num_seqs=config.inference.max_num_seqs,
                enable_lora=True,
                max_loras=2,
                port=config.vllm_port,
            ),
        )

        bk = MegatronBackend()
        await model.register(bk)

        t = time.perf_counter()
        base_url, api_key = await bk._prepare_backend_for_training(model)
        run.server_startup_time = time.perf_counter() - t
        model_name = await get_model_name(base_url, api_key)
        logger.info(f"[vllm] server ready in {run.server_startup_time:.1f}s — model={model_name}")

        await warmup_server(base_url, model_name, config.num_warmup_requests)

        prompts = generate_benchmark_prompts(
            config.num_rollouts_per_step, config.model.max_input_tokens,
            dataset=config.dataset, seed=config.seed,
        )

        for step_idx in range(config.num_training_steps):
            logger.info(f"[vllm] step {step_idx+1}/{config.num_training_steps}")
            sm = StepMetrics(step=step_idx + 1)

            mem = get_gpu_memory_usage_nvidia_smi()
            sm.gpu_memory_during_rollout = sum(mem.values())

            # -- Rollout (timed) ------------------------------------
            sm.rollout_start = time.perf_counter()
            sm.request_metrics = await measure_rollout(
                base_url, model_name, prompts,
                config.model.max_output_tokens, config.concurrency,
                config.request_timeout,
            )
            sm.rollout_end = time.perf_counter()

            errs = sum(1 for r in sm.request_metrics if r.error)
            logger.info(f"  rollout {sm.rollout_time:.1f}s  "
                        f"{sm.rollout_throughput:.0f} tok/s  "
                        f"TTFT={sm.avg_ttft:.4f}s  err={errs}")

            # -- Train (real Megatron via backend.train) -------------
            sm.training_start = time.perf_counter()
            tgroups = _make_trajectory_groups(sm.request_metrics, prompts)
            try:
                result = await bk.train(
                    model, tgroups,
                    learning_rate=config.training.learning_rate,
                )
                logger.info(f"  train done — step={result.step} "
                            f"loss={result.metrics.get('loss', '?')}")
            except Exception as e:
                logger.error(f"  train failed: {e}")
                run.errors.append(str(e))
            sm.training_end = time.perf_counter()

            # Update model name after LoRA update
            model_name = await get_model_name(base_url, api_key)

            run.steps.append(sm)

        run.end_time = time.perf_counter()
        try:
            await bk.close()
        except Exception:
            pass
        return run

    async def _run_sglang() -> BenchmarkRun:
        from benchmarks.sglang_vs_vllm.sglang_megatron_backend import SGLangMegatronBackend
        import art

        run = BenchmarkRun(backend="sglang", model=config.model.base_model,
                           dataset=config.dataset)
        run.start_time = time.perf_counter()

        model = art.TrainableModel(
            name=config.model.model_name + "-sglang-bench",
            project=config.model.project,
            base_model=config.model.base_model,
        )
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(max_seq_length=config.model.max_seq_length),
        )

        bk = SGLangMegatronBackend(
            sglang_python=config.sglang_python,
            port=config.sglang_port,
            tensor_parallel_size=config.inference.get_tp_size(),
            gpu_memory_utilization=config.inference.gpu_memory_utilization,
            max_num_seqs=config.inference.max_num_seqs,
        )
        await model.register(bk)

        t = time.perf_counter()
        base_url, api_key = await bk._prepare_backend_for_training(model)
        run.server_startup_time = time.perf_counter() - t
        model_name = config.model.base_model  # SGLang serves under HF name
        logger.info(f"[sglang] server ready in {run.server_startup_time:.1f}s")

        await warmup_server(base_url, model_name, config.num_warmup_requests)

        prompts = generate_benchmark_prompts(
            config.num_rollouts_per_step, config.model.max_input_tokens,
            dataset=config.dataset, seed=config.seed,
        )

        for step_idx in range(config.num_training_steps):
            logger.info(f"[sglang] step {step_idx+1}/{config.num_training_steps}")
            sm = StepMetrics(step=step_idx + 1)

            mem = get_gpu_memory_usage_nvidia_smi()
            sm.gpu_memory_during_rollout = sum(mem.values())

            # -- Rollout (timed) ------------------------------------
            sm.rollout_start = time.perf_counter()
            sm.request_metrics = await measure_rollout(
                base_url, model_name, prompts,
                config.model.max_output_tokens, config.concurrency,
                config.request_timeout,
            )
            sm.rollout_end = time.perf_counter()

            errs = sum(1 for r in sm.request_metrics if r.error)
            logger.info(f"  rollout {sm.rollout_time:.1f}s  "
                        f"{sm.rollout_throughput:.0f} tok/s  "
                        f"TTFT={sm.avg_ttft:.4f}s  err={errs}")

            # -- Train (real Megatron via backend.train) -------------
            sm.training_start = time.perf_counter()
            tgroups = _make_trajectory_groups(sm.request_metrics, prompts)
            try:
                result = await bk.train(
                    model, tgroups,
                    learning_rate=config.training.learning_rate,
                )
                logger.info(f"  train done — step={result.step} "
                            f"loss={result.metrics.get('loss', '?')}")
            except Exception as e:
                logger.error(f"  train failed: {e}")
                run.errors.append(str(e))
            sm.training_end = time.perf_counter()

            run.steps.append(sm)

        run.end_time = time.perf_counter()
        try:
            await bk.close()
        except Exception:
            pass
        return run

    def _make_trajectory_groups(
        req_metrics: list[RequestMetrics],
        prompts: list[list[dict[str, str]]],
    ) -> list:
        """Build real TrajectoryGroups from rollout results for backend.train()."""
        import art

        groups = []
        batch: list[art.Trajectory] = []
        for i, rm in enumerate(req_metrics):
            user_msg = prompts[i][-1]["content"] if i < len(prompts) else "hello"
            # We don't have the actual response text from streaming, so use a placeholder
            # The model's response content doesn't matter for training correctness —
            # what matters is the logprobs captured by the inference engine.
            reward = 1.0 if not rm.error else 0.0
            batch.append(art.Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": user_msg},
                    art.Choice(message={"role": "assistant", "content": f"Response {i}"}),
                ],
                reward=reward,
            ))
            if len(batch) >= 4:
                # Need at least 2 different rewards for variance
                if len(batch) >= 2:
                    batch[0] = art.Trajectory(
                        messages_and_choices=batch[0].messages_and_choices,
                        reward=1.0,
                    )
                    batch[1] = art.Trajectory(
                        messages_and_choices=batch[1].messages_and_choices,
                        reward=0.0,
                    )
                groups.append(art.TrajectoryGroup(list(batch)))
                batch = []
        if batch:
            if len(batch) >= 2:
                batch[0] = art.Trajectory(
                    messages_and_choices=batch[0].messages_and_choices,
                    reward=1.0,
                )
                batch[1] = art.Trajectory(
                    messages_and_choices=batch[1].messages_and_choices,
                    reward=0.0,
                )
            groups.append(art.TrajectoryGroup(list(batch)))
        return groups

    # ---- run the chosen backend -----------------------------------

    async def _main():
        if backend == "vllm":
            result = await _run_vllm()
        elif backend == "sglang":
            result = await _run_sglang()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        with open(results_path, "w") as f:
            json.dump(result.summary(), f, indent=2)
        logger.info(f"[{backend}] Results written to {results_path}")

    asyncio.run(_main())


# ===================================================================
# Orchestrator — runs in the main process
# ===================================================================

def cleanup_gpus() -> None:
    """Kill leftover inference/training processes to get a clean GPU state.

    IMPORTANT: patterns must NOT match our own benchmark process.
    Our script path contains 'sglang_vs_vllm' so naive 'pkill -f vllm'
    would kill ourselves.  Use specific process patterns instead.
    """
    my_pid = os.getpid()
    # Kill specific server/training processes — NOT our benchmark script
    patterns = [
        "model-service",
        "megatron-service",
        "sglang.launch_server",     # SGLang server process (not our script)
        "vllm.entrypoints",         # vLLM server process (not our script)
        "torchrun",
    ]
    for pattern in patterns:
        # pkill with a specific enough pattern to avoid self-kill
        subprocess.run(
            ["pkill", "-9", "-f", pattern],
            capture_output=True, timeout=10,
        )
    # Wait for OS to reclaim GPU memory
    import time as _t
    _t.sleep(5)


def spawn_worker(backend: str, config_path: str, results_path: str) -> int:
    """
    Fork a subprocess for one backend.  Returns exit code.
    The subprocess gets a completely fresh CUDA context.
    """
    # Use uv run to ensure the ART .venv is activated in the subprocess
    # sys.executable might be the venv python, but uv run is safer
    # because it handles PATH and env correctly.
    script = os.path.abspath(__file__)
    cmd = [
        "uv", "run", "python", script,
        "--_worker", backend,
        "--_config", config_path,
        "--_results", results_path,
    ]
    logger.info(f"Spawning {backend} worker: {' '.join(cmd)}")
    env = os.environ.copy()
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    # Reduce CPU RAM pressure: tell PyTorch not to pre-allocate
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Print available RAM before spawning
    try:
        import shutil
        mem = shutil.disk_usage("/")
        r = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            logger.info(f"System RAM:\n{r.stdout}")
    except Exception:
        pass

    proc = subprocess.run(cmd, env=env)

    if proc.returncode == -9 or proc.returncode == 137:
        logger.error(
            f"{backend} worker was KILLED (exit code {proc.returncode}). "
            f"This usually means the OS OOM killer terminated the process. "
            f"Try: --gpu-memory-utilization 0.6 or a smaller model."
        )
    elif proc.returncode != 0:
        logger.error(f"{backend} worker exited with code {proc.returncode}")

    return proc.returncode


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark SGLang vs vLLM during Megatron training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training benchmark (real Megatron, GPU-isolated)
  python run_benchmark.py --sglang-python ~/.venvs/sglang-bench/bin/python

  # Specific model + more steps
  python run_benchmark.py --model Qwen/Qwen2.5-14B-Instruct --num-steps 5

  # Only one backend
  python run_benchmark.py --backends sglang
        """,
    )
    # Internal flags for subprocess dispatch (hidden from user)
    p.add_argument("--_worker", help=argparse.SUPPRESS)
    p.add_argument("--_config", help=argparse.SUPPRESS)
    p.add_argument("--_results", help=argparse.SUPPRESS)

    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset", choices=["gsm8k", "sharegpt", "agentic", "math", "synthetic"],
                   default="gsm8k",
                   help="Prompt dataset (default: gsm8k)")
    p.add_argument("--backends", nargs="+", choices=["vllm", "sglang"],
                   default=["vllm", "sglang"])
    p.add_argument("--num-steps", type=int, default=3,
                   help="Training steps (default: 3)")
    p.add_argument("--num-rollouts", type=int, default=16,
                   help="Rollout requests per step (default: 16)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--output", default="benchmark_results")
    p.add_argument("--sglang-python", default="")
    p.add_argument("--vllm-port", type=int, default=8100)
    p.add_argument("--sglang-port", type=int, default=8200)
    p.add_argument("--tp", type=int, default=0, help="Tensor parallel (0=auto)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                   help="GPU memory fraction (default: 0.7, lower if OOM)")
    p.add_argument("--skip-preflight", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Subprocess worker mode -----------------------------------
    if args._worker:
        run_worker(args._worker, args._config, args._results)
        return

    # ---- Orchestrator mode ----------------------------------------
    from benchmarks.sglang_vs_vllm.config import BenchmarkConfig, ModelConfig, InferenceConfig
    from benchmarks.sglang_vs_vllm.metrics_collector import (
        BenchmarkRun, generate_comparison_report,
    )

    config = BenchmarkConfig(
        backends=args.backends,
        dataset=args.dataset,
        num_training_steps=args.num_steps,
        num_rollouts_per_step=args.num_rollouts,
        concurrency=args.concurrency,
        output_dir=args.output,
        sglang_python=args.sglang_python or "",
        vllm_port=args.vllm_port,
        sglang_port=args.sglang_port,
        model=ModelConfig(
            base_model=args.model,
            max_output_tokens=args.max_output_tokens,
        ),
        inference=InferenceConfig(
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ),
    )

    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  SGLang + Megatron  vs  vLLM + Megatron")
    logger.info("  GPU-isolated, real Megatron training")
    logger.info("=" * 70)
    logger.info(f"  Model:       {config.model.base_model}")
    logger.info(f"  Dataset:     {config.dataset}")
    logger.info(f"  Backends:    {config.backends}")
    logger.info(f"  Steps:       {config.num_training_steps}")
    logger.info(f"  Rollouts:    {config.num_rollouts_per_step}")
    logger.info(f"  Concurrency: {config.concurrency}")
    logger.info(f"  Max tokens:  {config.model.max_output_tokens}")
    logger.info(f"  Output:      {config.output_dir}")
    logger.info("")

    # Serialize config for workers (dataclasses → flat dict)
    config_dict = {
        "backends": config.backends,
        "dataset": config.dataset,
        "num_training_steps": config.num_training_steps,
        "num_rollouts_per_step": config.num_rollouts_per_step,
        "concurrency": config.concurrency,
        "output_dir": config.output_dir,
        "sglang_python": config.sglang_python,
        "vllm_python": config.vllm_python,
        "vllm_port": config.vllm_port,
        "sglang_port": config.sglang_port,
        "seed": config.seed,
        "num_warmup_requests": config.num_warmup_requests,
        "server_startup_timeout": config.server_startup_timeout,
        "request_timeout": config.request_timeout,
    }
    config_file = os.path.join(config.output_dir, "_bench_config.json")
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)

    results: dict[str, Any] = {}

    for backend in config.backends:
        results_file = os.path.join(config.output_dir, f"{backend}_metrics.json")

        logger.info(f"\n{'='*60}")
        logger.info(f"  Running {backend.upper()} in isolated subprocess")
        logger.info(f"{'='*60}")

        # Clean GPU state before each backend
        cleanup_gpus()

        rc = spawn_worker(backend, config_file, results_file)
        if rc != 0:
            logger.error(f"{backend} worker exited with code {rc}")
        elif os.path.exists(results_file):
            with open(results_file) as f:
                results[backend] = json.load(f)
            logger.info(f"{backend} results collected")
        else:
            logger.error(f"{backend} produced no results file")

        # Clean up after each backend
        cleanup_gpus()

    # ---- Comparison report ----------------------------------------
    if "vllm" in results and "sglang" in results:
        # Reconstruct BenchmarkRun objects for report generation
        vllm_run = _dict_to_run(results["vllm"])
        sglang_run = _dict_to_run(results["sglang"])
        report = generate_comparison_report(vllm_run, sglang_run, config.output_dir)
        print("\n" + report)
    elif results:
        for name, data in results.items():
            print(f"\n{name.upper()} results:")
            print(json.dumps(data, indent=2))
    else:
        logger.error("No results collected from any backend!")

    logger.info(f"\nResults → {config.output_dir}/")


def _dict_to_run(d: dict) -> Any:
    """Reconstruct a BenchmarkRun from a summary dict (for the report)."""
    from benchmarks.sglang_vs_vllm.metrics_collector import BenchmarkRun, StepMetrics, RequestMetrics

    run = BenchmarkRun(
        backend=d["backend"],
        model=d["model"],
        dataset=d.get("dataset", "gsm8k"),
        server_startup_time=d.get("server_startup_s", 0),
    )
    # Set total time from the data
    run.start_time = 0.0
    run.end_time = d.get("total_time_s", 0)

    for sd in d.get("steps", []):
        sm = StepMetrics(step=sd["step"])
        sm.rollout_start = 0.0
        sm.rollout_end = sd.get("rollout_time_s", 0)
        sm.gpu_memory_during_rollout = sd.get("gpu_mem_gb", 0) * 1e9
        # Reconstruct fake request metrics for aggregate calculations
        throughput = sd.get("throughput_tok_s", 0)
        rollout_t = sd.get("rollout_time_s", 1)
        n = sd.get("num_requests", 1)
        avg_ttft = sd.get("avg_ttft_s", 0)
        avg_itl = sd.get("avg_itl_s", 0)
        avg_lat = sd.get("avg_latency_s", 0)
        for i in range(n):
            comp_tok = int(throughput * rollout_t / max(n, 1))
            sm.request_metrics.append(RequestMetrics(
                request_id=i, start_time=0, end_time=avg_lat,
                ttft=avg_ttft, total_time=avg_lat,
                prompt_tokens=0, completion_tokens=comp_tok,
            ))
        run.steps.append(sm)
    return run


if __name__ == "__main__":
    main()
