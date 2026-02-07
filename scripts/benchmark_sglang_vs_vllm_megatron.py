#!/usr/bin/env python3
"""
Benchmark Harness: SGLang vs vLLM with Megatron Training

This script compares SGLang and vLLM inference backends when used with
Megatron distributed training for RL workloads.

Scenarios:
    A: Inference-only (prefix caching effectiveness)
    B: Single training iteration
    C: Full RL training loop (multiple iterations)
    D: Prefix cache utilization test
    E: ART Sleep/Wake benchmark (vLLM only) - time-multiplexed GPU sharing

ART Architecture (Sleep/Wake):
    The ART (Asynchronous Real-time Training) architecture enables time-multiplexed
    GPU sharing between inference and training on the SAME GPU(s):
    
    1. INFERENCE PHASE: vLLM serves requests with model weights + KV cache in GPU
    2. SLEEP: do_sleep(level=2) offloads weights to CPU, discards KV cache
    3. gc_and_empty_cuda_cache() frees GPU memory completely
    4. TRAINING PHASE: Megatron/TRL uses freed GPU for gradient computation
    5. WAKE: do_wake_up() reloads weights from CPU to GPU (~5-15s)
    6. llm.add_lora(checkpoint) loads new LoRA weights from training
    7. llm.resume_generation() resumes inference serving
    
    Per-iteration overhead: ~7-15s (sleep: 2-5s, wake: 5-10s)

Usage:
    # Run all scenarios with SGLang
    python scripts/benchmark_sglang_vs_vllm_megatron.py --backend sglang --scenario all
    
    # Run inference-only with vLLM
    python scripts/benchmark_sglang_vs_vllm_megatron.py --backend vllm --scenario A
    
    # Compare both backends
    python scripts/benchmark_sglang_vs_vllm_megatron.py --backend both --scenario all
    
    # Run ART sleep/wake benchmark (vLLM only)
    python scripts/benchmark_sglang_vs_vllm_megatron.py --backend vllm --scenario E
    
    # Full training with sleep/wake (time-multiplexed GPU)
    python scripts/benchmark_sglang_vs_vllm_megatron.py --backend vllm --scenario C \\
        --vllm-cache-method sleep_wake --enable-training
    
    # SGLang with native sleep/wake (RECOMMENDED — offload weights to CPU, cache preserved)
    python scripts/benchmark_sglang_vs_vllm_megatron.py --backend sglang --scenario C \\
        --sglang-cache-method sleep_wake --enable-training --preserve-cache --sglang-gpu-ids 0
    
    # SGLang with freeze mode (SIGSTOP only, cache preserved, no weight offload)
    python scripts/benchmark_sglang_vs_vllm_megatron.py --backend sglang --scenario C \\
        --sglang-cache-method freeze --enable-training --preserve-cache --sglang-gpu-ids 0
"""

import argparse
import asyncio
import json
import csv
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

# =====================================================================
# FIX: Ensure SGLang subprocess can find system libraries (libnuma, CUDA)
# =====================================================================
# Set environment variables that will be inherited by ALL subprocesses
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
# Force preload libnuma for SGLang sgl_kernel
if os.path.exists('/usr/lib/x86_64-linux-gnu/libnuma.so.1'):
    os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libnuma.so.1:' + os.environ.get('LD_PRELOAD', '')
# =====================================================================

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from openai import AsyncOpenAI


# =============================================================================
# Data Classes for Metrics
# =============================================================================

@dataclass
class InferenceMetrics:
    """Metrics for inference performance."""
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    prefix_cache_hit_rate: float = 0.0
    memory_utilization_gb: float = 0.0
    total_tokens_generated: int = 0
    total_requests: int = 0


@dataclass
class TrainingMetrics:
    """Metrics for training performance."""
    weight_sync_time_s: float = 0.0
    training_step_time_s: float = 0.0
    inference_time_s: float = 0.0
    total_iteration_time_s: float = 0.0
    gpu_memory_training_gb: float = 0.0
    loss: float = 0.0
    grad_norm: float = 0.0
    # ART sleep/wake metrics (vLLM time-multiplexed GPU sharing)
    sleep_time_s: float = 0.0
    wake_time_s: float = 0.0
    sleep_wake_overhead_s: float = 0.0
    memory_freed_during_sleep_gb: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    backend: str
    scenario: str
    model: str
    timestamp: str
    inference_metrics: InferenceMetrics = field(default_factory=InferenceMetrics)
    training_metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    config: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "scenario": self.scenario,
            "model": self.model,
            "timestamp": self.timestamp,
            "inference_metrics": asdict(self.inference_metrics),
            "training_metrics": asdict(self.training_metrics),
            "config": self.config,
        }


# =============================================================================
# GSM8K Dataset Loader
# =============================================================================

class GSM8KDataset:
    """Load and manage GSM8K math problems for benchmarking."""
    
    SYSTEM_PROMPT = """You are a helpful math tutor. Solve the given math problem step by step.
Always end your response with the final answer in the format: #### [number]"""
    
    def __init__(self, split: str = "test", max_samples: int | None = None):
        self.split = split
        self.max_samples = max_samples
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """Load GSM8K from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            # Check if it's actually missing or if it's a different import error
            import importlib.util
            if importlib.util.find_spec("datasets") is None:
                raise ImportError(
                    "The 'datasets' package is required. Install with: pip install datasets"
                ) from e
            else:
                # Package exists but failed to import - show the real error
                raise ImportError(
                    f"The 'datasets' package is installed but failed to import: {e}"
                ) from e
        
        dataset = load_dataset("gsm8k", "main", split=self.split)
        
        for i, item in enumerate(dataset):
            if self.max_samples and i >= self.max_samples:
                break
            self.data.append({
                "question": item["question"],
                "answer": item["answer"],
                "final_answer": self._extract_final_answer(item["answer"]),
            })
        print(f"Loaded {len(self.data)} GSM8K problems")
    
    def _extract_final_answer(self, answer: str) -> str:
        """Extract the final numerical answer from GSM8K format."""
        if "####" in answer:
            return answer.split("####")[-1].strip()
        return ""
    
    def get_prompts(self, n: int, with_system: bool = True) -> list[dict]:
        """Get n prompts for inference."""
        prompts = []
        for i in range(min(n, len(self.data))):
            item = self.data[i % len(self.data)]
            if with_system:
                prompts.append({
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": item["question"]},
                    ],
                    "expected_answer": item["final_answer"],
                })
            else:
                prompts.append({
                    "messages": [
                        {"role": "user", "content": f"Solve: {item['question']}"},
                    ],
                    "expected_answer": item["final_answer"],
                })
        return prompts
    
    def compute_reward(self, response: str, expected: str) -> float:
        """Compute reward based on correctness."""
        # Extract number from response
        response_clean = response.replace(",", "").strip()
        if "####" in response_clean:
            response_answer = response_clean.split("####")[-1].strip()
        else:
            # Try to find last number in response
            import re
            numbers = re.findall(r'-?\d+\.?\d*', response_clean)
            response_answer = numbers[-1] if numbers else ""
        
        expected_clean = expected.replace(",", "").strip()
        return 1.0 if response_answer == expected_clean else 0.0


# =============================================================================
# Backend Manager
# =============================================================================

class BackendManager:
    """Manages SGLang and vLLM backends for benchmarking."""
    
    def __init__(
        self,
        backend_type: Literal["sglang", "vllm"],
        model: str,
        work_dir: str,
        vllm_tp_size: int | None = None,
        preserve_cache: bool = False,
        sglang_gpu_ids: list[int] | None = None,
        sglang_cache_method: str = "freeze",
        vllm_cache_method: str = "http",
        vllm_gpu_ids: list[int] | None = None,
        training_gpu_ids: list[int] | None = None,
        in_process: bool = True,  # Run service in-process to avoid multiprocessing issues
    ):
        self.backend_type = backend_type
        self.model = model
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.vllm_tp_size = vllm_tp_size
        self.preserve_cache = preserve_cache
        self.sglang_gpu_ids = sglang_gpu_ids
        self.sglang_cache_method = sglang_cache_method
        self.vllm_cache_method = vllm_cache_method
        self.vllm_gpu_ids = vllm_gpu_ids
        self.training_gpu_ids = training_gpu_ids
        self.in_process = in_process
        
        self.backend = None
        self.service = None
        self.client: AsyncOpenAI | None = None
        self.model_name: str = ""
        
        # vLLM AsyncLLM instance for sleep/wake operations
        self._vllm_engine = None
        self._is_sleeping = False
    
    async def initialize(self):
        """Initialize the backend."""
        if self.backend_type == "sglang":
            await self._init_sglang()
        else:
            await self._init_vllm()
    
    async def _init_sglang(self):
        """Initialize SGLang backend."""
        import subprocess
        # Kill any stale SGLang/Megatron processes from previous runs to free GPU memory
        subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "megatron-service"], capture_output=True)
        await asyncio.sleep(1.0)
        
        from art.megatron.sglang_backend import SGLangMegatronBackend
        from art import TrainableModel
        
        sglang_config = {
            "server_timeout": 600,
            "mem_fraction_static": 0.90,  # Higher = more cache entries survive
            "log_level": "warning",
            "preserve_cache_during_training": self.preserve_cache,
            "cache_method": self.sglang_cache_method,  # "freeze" (default) | "sleep_wake" | "hot_reload" | "restart"
            # --- SGLang-specific optimizations ---
            "schedule_policy": "lpm",           # Longest Prefix Match: reorders requests for max cache hits
            "chunked_prefill_size": 8192,       # Overlap prefill with decode
            "enable_overlap_schedule": True,    # CPU scheduling parallel with GPU execution
            "enable_torch_compile": True,       # Compiled kernels for faster decode
        }
        
        # Set tensor parallelism for SGLang (for fair comparison with vLLM)
        if self.vllm_tp_size is not None and not self.preserve_cache:
            sglang_config["tensor_parallel_size"] = self.vllm_tp_size
        
        # Configure GPU isolation for cache preservation
        if self.preserve_cache:
            if self.sglang_gpu_ids is not None:
                sglang_config["sglang_gpu_ids"] = self.sglang_gpu_ids
            else:
                # Default: use GPU 0 for SGLang when preserving cache
                # This leaves other GPUs free for Megatron training
                sglang_config["sglang_gpu_ids"] = [0]
            print(f"  SGLang cache preservation mode: GPUs {sglang_config['sglang_gpu_ids']}")
        
        # Pass explicit training GPU IDs if set (ensures fair comparison with vLLM)
        if self.training_gpu_ids is not None:
            sglang_config["training_gpu_ids"] = self.training_gpu_ids
            print(f"  Training GPUs: {self.training_gpu_ids}")
        print(f"  SGLang cache method: {self.sglang_cache_method}")
        
        self.backend = SGLangMegatronBackend(
            path=str(self.work_dir),
            sglang_config=sglang_config,
            in_process=True,  # Must be in-process: mp_actors can't proxy AsyncIterator
                              # from service.train() across process boundary, causing hangs
        )
        
        # Create and register model
        model_obj = TrainableModel(
            name=f"benchmark-{self.backend_type}",
            base_model=self.model,
            project="benchmark",
        )
        await self.backend.register(model_obj)
        
        # Start server
        base_url, api_key = await self.backend._prepare_backend_for_training(model_obj, None)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.service = self.backend._services.get(model_obj.name)
        
        # Get the inference model name
        self.model_name = self.backend._model_inference_name(model_obj)
        self._model_obj = model_obj
    
    async def _init_vllm(self):
        """Initialize vLLM backend."""
        # For vLLM, we use the existing MegatronBackend
        try:
            from art.megatron.backend import MegatronBackend
            from art import TrainableModel
            import torch
            
            # Use specified TP size or default to 1 for fair comparison with SGLang
            # SGLang uses GPU 0 for inference, rest for training
            # For fair comparison, vLLM should also use limited GPUs for inference
            if self.vllm_tp_size is not None:
                tp_size = self.vllm_tp_size
            else:
                # Default to 1 GPU for fair comparison with SGLang
                tp_size = 1
            
            available_gpus = torch.cuda.device_count() or 1
            if tp_size > available_gpus:
                print(f"  Warning: Requested TP={tp_size} but only {available_gpus} GPUs available")
                tp_size = available_gpus
            
            print(f"  Using tensor parallelism: TP={tp_size} (of {available_gpus} available)")
            
            # Build vLLM config for cache preservation
            vllm_config = {
                "preserve_cache_during_training": self.preserve_cache,
                "cache_method": self.vllm_cache_method,
            }
            
            # Configure GPU isolation for cache preservation
            # All methods (http, sleep_wake) use GPU isolation for fair comparison
            if self.preserve_cache:
                if self.vllm_gpu_ids is not None:
                    vllm_config["vllm_gpu_ids"] = self.vllm_gpu_ids
                elif self.sglang_gpu_ids is not None:
                    # Use same GPU IDs as SGLang for fair comparison
                    vllm_config["vllm_gpu_ids"] = self.sglang_gpu_ids
                else:
                    # Default: use GPU 0 for vLLM when preserving cache
                    vllm_config["vllm_gpu_ids"] = [0]
                print(f"  vLLM cache method: {self.vllm_cache_method}")
                print(f"  vLLM GPUs: {vllm_config['vllm_gpu_ids']}")
            
            # Enable sleep/wake support in config
            if self.vllm_cache_method == "sleep_wake":
                vllm_config["enable_sleep_wake"] = True
                print(f"  vLLM sleep/wake: ENABLED")
            
            # Add training GPU IDs for fair comparison
            if self.training_gpu_ids is not None:
                vllm_config["training_gpu_ids"] = self.training_gpu_ids
                print(f"  Training GPUs: {self.training_gpu_ids}")
            
            # Run in-process to avoid multiprocessing pickle errors
            print(f"  In-process mode: {self.in_process}")
            
            self.backend = MegatronBackend(
                path=str(self.work_dir),
                tensor_parallel_size=tp_size,
                vllm_config=vllm_config,
                in_process=self.in_process,
            )
            
            # Create and register model
            model_obj = TrainableModel(
                name=f"benchmark-{self.backend_type}",
                base_model=self.model,
                project="benchmark",
            )
            await self.backend.register(model_obj)
            
            base_url, api_key = await self.backend._prepare_backend_for_training(model_obj, None)
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            self.service = self.backend._services.get(model_obj.name)
            self.model_name = self.backend._model_inference_name(model_obj)
            self._model_obj = model_obj
            
            # Try to capture the vLLM engine for sleep/wake operations
            self._capture_vllm_engine()
            
        except ImportError as e:
            raise RuntimeError(f"vLLM backend requires vLLM installation: {e}")
    
    def _capture_vllm_engine(self):
        """Capture the vLLM AsyncLLM engine for sleep/wake operations."""
        # Try multiple paths to find the engine
        engine = None
        
        # Path 1: From service
        if self.service is not None:
            for attr in ['engine', '_engine', 'llm', '_llm', 'async_llm']:
                if hasattr(self.service, attr):
                    engine = getattr(self.service, attr)
                    if engine is not None:
                        break
        
        # Path 2: From backend
        if engine is None and self.backend is not None:
            for attr in ['engine', '_engine', 'llm', '_llm', 'async_llm']:
                if hasattr(self.backend, attr):
                    engine = getattr(self.backend, attr)
                    if engine is not None:
                        break
        
        # Path 3: Check for vLLM-specific service attributes
        if engine is None and self.service is not None:
            if hasattr(self.service, 'vllm_service'):
                vllm_svc = self.service.vllm_service
                for attr in ['engine', '_engine', 'llm']:
                    if hasattr(vllm_svc, attr):
                        engine = getattr(vllm_svc, attr)
                        if engine is not None:
                            break
        
        if engine is not None:
            self._vllm_engine = engine
            print(f"  ✓ Captured vLLM engine for sleep/wake: {type(engine).__name__}")
    
    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> tuple[str, float, float]:
        """
        Generate completion and return (response, ttft_ms, total_latency_ms).
        """
        start = time.perf_counter()
        first_token_time = None
        response_text = ""
        
        # Use streaming to measure TTFT
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        
        async for chunk in stream:
            if first_token_time is None and chunk.choices[0].delta.content:
                first_token_time = time.perf_counter()
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        
        end = time.perf_counter()
        
        ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
        total_ms = (end - start) * 1000
        
        return response_text, ttft_ms, total_ms
    
    async def generate_for_training(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ):
        """
        Generate completion with logprobs for training.
        Returns (Choice object, response_text, ttft_ms, total_latency_ms, cache_info).
        
        cache_info is a dict with prompt_tokens and cached_tokens extracted from
        the response usage field, enabling real counter-based cache hit measurement
        for both SGLang and vLLM (same method, apples-to-apples comparison).
        
        The Choice object can be used directly in Trajectory for training.
        """
        from openai.types.chat.chat_completion import Choice
        
        start = time.perf_counter()
        
        # Non-streaming request with logprobs for training
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
        )
        
        end = time.perf_counter()
        
        choice = response.choices[0]
        response_text = choice.message.content or ""
        total_ms = (end - start) * 1000
        
        # For non-streaming, TTFT ~= total time (first token comes with response)
        ttft_ms = total_ms
        
        # Extract per-request cache info from response usage
        # Both SGLang and vLLM return this in OpenAI-compatible format
        cache_info = {"prompt_tokens": 0, "cached_tokens": 0}
        if response.usage:
            cache_info["prompt_tokens"] = response.usage.prompt_tokens or 0
            # Try prompt_tokens_details.cached_tokens (OpenAI standard field)
            if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                details = response.usage.prompt_tokens_details
                if hasattr(details, 'cached_tokens'):
                    cache_info["cached_tokens"] = details.cached_tokens or 0
            # Fallback: some SGLang versions put it directly in usage
            if cache_info["cached_tokens"] == 0 and hasattr(response.usage, 'cached_tokens'):
                cache_info["cached_tokens"] = response.usage.cached_tokens or 0
        
        return choice, response_text, ttft_ms, total_ms, cache_info
    
    async def generate_batch(
        self,
        prompts: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[tuple[str, float, float]]:
        """Generate completions for a batch of prompts."""
        tasks = [
            self.generate(p["messages"], max_tokens, temperature)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)
    
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
        return 0.0
    
    async def get_real_cache_stats(self) -> dict:
        """Get actual cache statistics from the backend server.
        
        Uses Prometheus /metrics endpoint for BOTH backends to ensure
        apples-to-apples comparison. Falls back to /get_server_info for SGLang.
        """
        import aiohttp
        
        host = self.service._server_host
        port = self.service._server_port
        
        if self.backend_type == "sglang":
            # Strategy 1: Try /metrics Prometheus endpoint (same method as vLLM)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{host}:{port}/metrics",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            stats = self._parse_sglang_prometheus_metrics(text)
                            if "hit_rate" in stats:
                                stats["source"] = "prometheus"
                                return stats
            except Exception:
                pass
            
            # Strategy 2: Try /get_server_info radix_cache_stats
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{host}:{port}/get_server_info",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            cache_stats = data.get("radix_cache_stats", {})
                            if cache_stats:
                                parsed = self._parse_sglang_server_info_cache(cache_stats)
                                if "hit_rate" in parsed:
                                    parsed["source"] = "server_info"
                                    return parsed
                                # Return raw stats even without hit_rate for debugging
                                cache_stats["source"] = "server_info_raw"
                                return cache_stats
            except Exception:
                pass
                
        elif self.backend_type == "vllm":
            # vLLM provides metrics endpoint (Prometheus format)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{host}:{port}/metrics",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            hit_count = 0
                            miss_count = 0
                            for line in text.splitlines():
                                if line.startswith("vllm_prefix_cache_hit_count"):
                                    hit_count = int(float(line.split()[-1]))
                                elif line.startswith("vllm_prefix_cache_miss_count"):
                                    miss_count = int(float(line.split()[-1]))
                            total = hit_count + miss_count
                            if total > 0:
                                return {
                                    "hit_rate": hit_count / total,
                                    "hit_count": hit_count,
                                    "miss_count": miss_count,
                                    "source": "prometheus",
                                }
            except Exception:
                pass
        return {}
    
    @staticmethod
    def _parse_sglang_prometheus_metrics(text: str) -> dict:
        """Parse SGLang's Prometheus metrics for cache hit information.
        
        SGLang exposes various metrics. We look for cache-related counters
        like sglang_cache_hit_rate, radix_cache_hit, etc.
        """
        stats = {}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            # SGLang cache hit rate (direct gauge)
            if "cache_hit_rate" in line and not line.startswith("#"):
                try:
                    val = float(line.split()[-1])
                    stats["hit_rate"] = val
                except (ValueError, IndexError):
                    pass
            # SGLang prefix match/cache tokens counters
            if "cached_tokens" in line.lower() or "cache_hit" in line.lower():
                try:
                    parts = line.split()
                    metric_name = parts[0]
                    val = float(parts[-1])
                    stats[metric_name] = val
                except (ValueError, IndexError):
                    pass
            if "cache_miss" in line.lower():
                try:
                    parts = line.split()
                    metric_name = parts[0]
                    val = float(parts[-1])
                    stats[metric_name] = val
                except (ValueError, IndexError):
                    pass
        
        # If we found individual hit/miss counters but no hit_rate, compute it
        hit_keys = [k for k in stats if "hit" in k.lower() and k != "hit_rate"]
        miss_keys = [k for k in stats if "miss" in k.lower()]
        if hit_keys and miss_keys and "hit_rate" not in stats:
            total_hits = sum(stats[k] for k in hit_keys)
            total_misses = sum(stats[k] for k in miss_keys)
            if total_hits + total_misses > 0:
                stats["hit_rate"] = total_hits / (total_hits + total_misses)
        
        return stats
    
    @staticmethod
    def _parse_sglang_server_info_cache(cache_stats: dict) -> dict:
        """Parse radix_cache_stats from SGLang's /get_server_info.
        
        Possible field names across SGLang versions:
        - hit_rate (direct)
        - total, hit, miss (counters)
        - token_usage, total_token_num (capacity info)
        """
        result = {}
        
        # Direct hit_rate field
        if "hit_rate" in cache_stats:
            result["hit_rate"] = float(cache_stats["hit_rate"])
            return result
        
        # Hit/miss counters
        hits = cache_stats.get("hit", cache_stats.get("hits", cache_stats.get("hit_count", 0)))
        misses = cache_stats.get("miss", cache_stats.get("misses", cache_stats.get("miss_count", 0)))
        total = hits + misses
        if total > 0:
            result["hit_rate"] = hits / total
            result["hit_count"] = hits
            result["miss_count"] = misses
            return result
        
        # Token-level stats
        total_tokens = cache_stats.get("total_token_num", cache_stats.get("total_tokens", 0))
        used_tokens = cache_stats.get("token_usage", cache_stats.get("used_tokens", 0))
        if total_tokens > 0 and used_tokens > 0:
            result["token_usage_rate"] = used_tokens / total_tokens
        
        return result
    
    async def measure_streaming_ttft(self, messages: list[dict], max_tokens: int = 256, num_probes: int = 5) -> float:
        """Measure actual TTFT using streaming requests.
        
        This bypasses the non-streaming TTFT=total_ms issue in generate_for_training,
        giving accurate prefill time measurement for cache hit rate estimation.
        """
        import asyncio
        ttft_values = []
        for _ in range(num_probes):
            start = time.perf_counter()
            first_token_time = None
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                stream=True,
            )
            async for chunk in stream:
                if first_token_time is None and chunk.choices[0].delta.content:
                    first_token_time = time.perf_counter()
                    break  # Only need first token for TTFT
            
            if first_token_time:
                ttft_values.append((first_token_time - start) * 1000)
            await asyncio.sleep(0.01)  # Small delay between probes
        
        if ttft_values:
            return sum(ttft_values) / len(ttft_values)
        return 0.0
    
    async def cleanup(self):
        """Cleanup backend resources."""
        import asyncio
        
        # Give pending tasks a moment to complete
        await asyncio.sleep(0.1)
        
        if self.service:
            try:
                await self.service.stop_openai_server()
            except Exception:
                pass
        
        # Clear references
        self.service = None
        self.client = None
        self.backend = None
        self._vllm_engine = None
        self._is_sleeping = False
        
        # Allow cleanup tasks to finish
        await asyncio.sleep(0.2)
    
    # =========================================================================
    # vLLM Sleep/Wake Operations for ART Time-Multiplexed GPU Sharing
    # =========================================================================
    
    @staticmethod
    def gc_and_empty_cuda_cache():
        """
        Force garbage collection and empty CUDA cache.
        
        This is called after do_sleep() to ensure GPU memory is fully released
        for training to use.
        
        Returns:
            tuple: (memory_before_gb, memory_after_gb)
        """
        import gc
        
        memory_before = 0.0
        memory_after = 0.0
        
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / 1e9
            
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            # Synchronize all CUDA streams
            torch.cuda.synchronize()
            # Empty the cache
            torch.cuda.empty_cache()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            memory_after = torch.cuda.memory_allocated() / 1e9
        
        return memory_before, memory_after
    
    async def sleep(self, level: int = 2) -> dict:
        """
        Put vLLM engine to sleep, offloading weights to CPU.
        
        This implements the ART architecture's time-multiplexed GPU sharing:
        - Pauses generation
        - Offloads model weights to CPU (pinned memory)
        - At level=2, discards KV cache to free maximum GPU memory
        
        Args:
            level: Sleep level (1=keep cache, 2=discard cache for max memory)
                   Level 2 is recommended for training as it frees more GPU memory.
        
        Returns:
            dict with timing information:
                - pause_time_s: Time to pause generation
                - offload_time_s: Time to offload weights to CPU
                - gc_time_s: Time for garbage collection
                - total_time_s: Total sleep time
                - memory_freed_gb: GPU memory freed
        
        Raises:
            RuntimeError: If not using vLLM backend or sleep_wake cache method
        """
        if self.backend_type != "vllm":
            raise RuntimeError("sleep() is only supported for vLLM backend")
        
        if self._is_sleeping:
            print("  Warning: vLLM engine already sleeping")
            return {"total_time_s": 0, "memory_freed_gb": 0}
        
        metrics = {
            "pause_time_s": 0.0,
            "offload_time_s": 0.0,
            "gc_time_s": 0.0,
            "total_time_s": 0.0,
            "memory_freed_gb": 0.0,
        }
        
        total_start = time.perf_counter()
        memory_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        # Get the vLLM AsyncLLM engine
        engine = self._get_vllm_engine()
        if engine is None:
            print("  Warning: Could not get vLLM engine for sleep")
            return metrics
        
        try:
            # Step 1: Pause generation
            print(f"  Pausing vLLM generation...")
            pause_start = time.perf_counter()
            await self._pause_vllm_generation(engine)
            metrics["pause_time_s"] = time.perf_counter() - pause_start
            
            # Step 2: Offload weights to CPU (do_sleep)
            print(f"  Offloading weights to CPU (level={level})...")
            offload_start = time.perf_counter()
            await self._do_sleep(engine, level=level)
            metrics["offload_time_s"] = time.perf_counter() - offload_start
            
            # Step 3: Garbage collection and empty CUDA cache
            print(f"  Running gc_and_empty_cuda_cache()...")
            gc_start = time.perf_counter()
            _, memory_after = self.gc_and_empty_cuda_cache()
            metrics["gc_time_s"] = time.perf_counter() - gc_start
            
            self._is_sleeping = True
            metrics["total_time_s"] = time.perf_counter() - total_start
            metrics["memory_freed_gb"] = memory_before - memory_after
            
            print(f"  ✓ vLLM sleeping: freed {metrics['memory_freed_gb']:.2f}GB in {metrics['total_time_s']:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Sleep failed: {e}")
            import traceback
            traceback.print_exc()
        
        return metrics
    
    async def wake_up(self, lora_checkpoint_path: str | None = None) -> dict:
        """
        Wake up vLLM engine, reloading weights from CPU to GPU.
        
        This implements the ART architecture's wake-up phase:
        - Reloads model weights from CPU (pinned memory) to GPU
        - Optionally loads a new LoRA adapter (from training)
        - Resumes generation
        
        Args:
            lora_checkpoint_path: Optional path to new LoRA checkpoint to load
        
        Returns:
            dict with timing information:
                - reload_time_s: Time to reload weights from CPU
                - lora_load_time_s: Time to load LoRA adapter (if any)
                - resume_time_s: Time to resume generation
                - total_time_s: Total wake time
                - memory_used_gb: GPU memory after wake
        
        Raises:
            RuntimeError: If not using vLLM backend
        """
        if self.backend_type != "vllm":
            raise RuntimeError("wake_up() is only supported for vLLM backend")
        
        if not self._is_sleeping:
            print("  Warning: vLLM engine not sleeping")
            return {"total_time_s": 0, "memory_used_gb": 0}
        
        metrics = {
            "reload_time_s": 0.0,
            "lora_load_time_s": 0.0,
            "resume_time_s": 0.0,
            "total_time_s": 0.0,
            "memory_used_gb": 0.0,
        }
        
        total_start = time.perf_counter()
        
        # Get the vLLM AsyncLLM engine
        engine = self._get_vllm_engine()
        if engine is None:
            print("  Warning: Could not get vLLM engine for wake_up")
            return metrics
        
        try:
            # Step 1: Wake up (reload weights from CPU to GPU)
            print(f"  Reloading weights from CPU to GPU...")
            reload_start = time.perf_counter()
            await self._do_wake_up(engine)
            metrics["reload_time_s"] = time.perf_counter() - reload_start
            
            # Step 2: Load new LoRA adapter if provided
            if lora_checkpoint_path:
                print(f"  Loading new LoRA adapter: {lora_checkpoint_path}")
                lora_start = time.perf_counter()
                await self._add_lora(engine, lora_checkpoint_path)
                metrics["lora_load_time_s"] = time.perf_counter() - lora_start
            
            # Step 3: Resume generation
            print(f"  Resuming vLLM generation...")
            resume_start = time.perf_counter()
            await self._resume_vllm_generation(engine)
            metrics["resume_time_s"] = time.perf_counter() - resume_start
            
            self._is_sleeping = False
            metrics["total_time_s"] = time.perf_counter() - total_start
            metrics["memory_used_gb"] = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            
            print(f"  ✓ vLLM awake: {metrics['memory_used_gb']:.2f}GB used in {metrics['total_time_s']:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Wake up failed: {e}")
            import traceback
            traceback.print_exc()
            self._is_sleeping = False  # Reset state even on failure
        
        return metrics
    
    def _get_vllm_engine(self):
        """Get the vLLM AsyncLLM engine instance."""
        if self._vllm_engine is not None:
            return self._vllm_engine
        
        # Try to get engine from service
        if self.service is not None:
            # Check if service has engine attribute
            if hasattr(self.service, 'engine'):
                self._vllm_engine = self.service.engine
                return self._vllm_engine
            if hasattr(self.service, '_engine'):
                self._vllm_engine = self.service._engine
                return self._vllm_engine
            if hasattr(self.service, 'llm'):
                self._vllm_engine = self.service.llm
                return self._vllm_engine
        
        # Try to get from backend
        if self.backend is not None:
            if hasattr(self.backend, '_engine'):
                self._vllm_engine = self.backend._engine
                return self._vllm_engine
            if hasattr(self.backend, 'engine'):
                self._vllm_engine = self.backend.engine
                return self._vllm_engine
        
        return None
    
    async def _pause_vllm_generation(self, engine):
        """Pause vLLM generation before sleep."""
        if hasattr(engine, 'pause_generation'):
            # vLLM AsyncLLM.pause_generation()
            await engine.pause_generation()
        elif hasattr(engine, 'stop'):
            # Alternative API
            await engine.stop()
        else:
            print("    Warning: Engine does not have pause_generation method")
    
    async def _resume_vllm_generation(self, engine):
        """Resume vLLM generation after wake."""
        if hasattr(engine, 'resume_generation'):
            # vLLM AsyncLLM.resume_generation()
            await engine.resume_generation()
        elif hasattr(engine, 'start'):
            # Alternative API
            await engine.start()
        else:
            print("    Warning: Engine does not have resume_generation method")
    
    async def _do_sleep(self, engine, level: int = 2):
        """
        Call vLLM's do_sleep to offload weights.
        
        Args:
            engine: vLLM AsyncLLM engine
            level: Sleep level
                   0 = Keep everything in GPU (minimal sleep)
                   1 = Offload weights, keep KV cache
                   2 = Offload weights, discard KV cache (maximum memory freed)
        """
        if hasattr(engine, 'do_sleep'):
            # vLLM AsyncLLM.do_sleep(level)
            # level=2 discards KV cache for maximum memory freeing
            await engine.do_sleep(level=level)
        elif hasattr(engine, 'sleep'):
            # Alternative API
            await engine.sleep(level=level)
        else:
            # Fallback: try to access model and move to CPU
            print("    Warning: Engine does not have do_sleep method, trying fallback...")
            await self._fallback_offload_to_cpu(engine)
    
    async def _do_wake_up(self, engine):
        """Call vLLM's do_wake_up to reload weights."""
        if hasattr(engine, 'do_wake_up'):
            # vLLM AsyncLLM.do_wake_up()
            await engine.do_wake_up()
        elif hasattr(engine, 'wake_up'):
            # Alternative API
            await engine.wake_up()
        else:
            # Fallback: try to reload model to GPU
            print("    Warning: Engine does not have do_wake_up method, trying fallback...")
            await self._fallback_reload_to_gpu(engine)
    
    async def _add_lora(self, engine, lora_path: str):
        """Load a new LoRA adapter after training."""
        if hasattr(engine, 'add_lora'):
            # vLLM AsyncLLM.add_lora()
            await engine.add_lora(lora_path)
        elif hasattr(engine, 'load_lora_adapter'):
            # Alternative API
            await engine.load_lora_adapter(lora_path)
        else:
            print(f"    Warning: Engine does not have add_lora method")
    
    async def _fallback_offload_to_cpu(self, engine):
        """Fallback method to offload model to CPU if do_sleep is not available."""
        # This is a simplified fallback - real implementation depends on vLLM internals
        print("    Using fallback CPU offload (may not preserve state)")
        if hasattr(engine, 'model'):
            model = engine.model
            if hasattr(model, 'to'):
                model.to('cpu')
    
    async def _fallback_reload_to_gpu(self, engine):
        """Fallback method to reload model to GPU if do_wake_up is not available."""
        print("    Using fallback GPU reload (may not restore state)")
        if hasattr(engine, 'model'):
            model = engine.model
            if hasattr(model, 'to'):
                model.to('cuda')
    
    def is_sleeping(self) -> bool:
        """Check if vLLM engine is currently sleeping."""
        return self._is_sleeping
    
    async def sleep_wake_cycle(
        self,
        training_fn,
        lora_checkpoint_path: str | None = None,
        sleep_level: int = 2,
    ) -> dict:
        """
        Execute a complete sleep → train → wake cycle.
        
        This is the main entry point for ART time-multiplexed training:
        1. Sleep: Offload vLLM weights to CPU, free GPU memory
        2. Train: Execute training function (user-provided)
        3. Wake: Reload weights to GPU, optionally load new LoRA
        
        Args:
            training_fn: Async function that performs training.
                         Called with no arguments, should return training metrics.
            lora_checkpoint_path: Optional path to LoRA checkpoint after training
            sleep_level: Sleep level (2 recommended for training)
        
        Returns:
            dict with complete cycle metrics:
                - sleep_metrics: From sleep()
                - training_metrics: From training_fn
                - wake_metrics: From wake_up()
                - total_cycle_time_s: Total time for complete cycle
        
        Example:
            async def train():
                # Your Megatron/Unsloth training code here
                return {"loss": 0.5, "steps": 100}
            
            metrics = await manager.sleep_wake_cycle(
                training_fn=train,
                lora_checkpoint_path="/path/to/new/lora",
            )
        """
        cycle_metrics = {
            "sleep_metrics": {},
            "training_metrics": {},
            "wake_metrics": {},
            "total_cycle_time_s": 0.0,
        }
        
        cycle_start = time.perf_counter()
        
        # Phase 1: Sleep
        print("\n  [ART] Phase 1: Sleeping vLLM...")
        cycle_metrics["sleep_metrics"] = await self.sleep(level=sleep_level)
        
        # Phase 2: Training (GPU is now free)
        print("\n  [ART] Phase 2: Training (GPU free)...")
        try:
            training_start = time.perf_counter()
            training_result = await training_fn()
            training_time = time.perf_counter() - training_start
            
            if isinstance(training_result, dict):
                cycle_metrics["training_metrics"] = training_result
            else:
                cycle_metrics["training_metrics"] = {"result": training_result}
            cycle_metrics["training_metrics"]["training_time_s"] = training_time
            
        except Exception as e:
            print(f"  ✗ Training failed: {e}")
            cycle_metrics["training_metrics"] = {"error": str(e)}
        
        # Phase 3: Wake up
        print("\n  [ART] Phase 3: Waking up vLLM...")
        cycle_metrics["wake_metrics"] = await self.wake_up(
            lora_checkpoint_path=lora_checkpoint_path
        )
        
        cycle_metrics["total_cycle_time_s"] = time.perf_counter() - cycle_start
        
        # Summary
        sleep_time = cycle_metrics["sleep_metrics"].get("total_time_s", 0)
        train_time = cycle_metrics["training_metrics"].get("training_time_s", 0)
        wake_time = cycle_metrics["wake_metrics"].get("total_time_s", 0)
        overhead = sleep_time + wake_time
        
        print(f"\n  [ART] Cycle complete:")
        print(f"    Sleep:    {sleep_time:.2f}s")
        print(f"    Training: {train_time:.2f}s")
        print(f"    Wake:     {wake_time:.2f}s")
        print(f"    Overhead: {overhead:.2f}s ({overhead/(overhead+train_time)*100:.1f}% of cycle)")
        
        return cycle_metrics


# =============================================================================
# Benchmark Scenarios
# =============================================================================

class BenchmarkRunner:
    """Runs benchmark scenarios."""
    
    def __init__(
        self,
        backend_type: str,
        model: str,
        work_dir: str,
        num_samples: int = 50,
        num_rollouts: int = 4,
        max_tokens: int = 256,
        num_iterations: int = 5,
        vllm_tp_size: int | None = None,
        enable_training: bool = False,
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        temperature: float = 1.0,
        preserve_cache: bool = False,
        sglang_gpu_ids: list[int] | None = None,
        sglang_cache_method: str = "freeze",
        vllm_cache_method: str = "http",
        vllm_gpu_ids: list[int] | None = None,
        training_gpu_ids: list[int] | None = None,
        in_process: bool = True,
    ):
        self.backend_type = backend_type
        self.model = model
        self.work_dir = work_dir
        self.num_samples = num_samples
        self.num_rollouts = num_rollouts
        self.max_tokens = max_tokens
        self.num_iterations = num_iterations
        self.vllm_tp_size = vllm_tp_size
        self.enable_training = enable_training
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.temperature = temperature
        self.preserve_cache = preserve_cache
        self.sglang_gpu_ids = sglang_gpu_ids
        self.sglang_cache_method = sglang_cache_method
        self.vllm_cache_method = vllm_cache_method
        self.vllm_gpu_ids = vllm_gpu_ids
        self.training_gpu_ids = training_gpu_ids
        self.in_process = in_process
        
        self.dataset = GSM8KDataset(max_samples=num_samples)
        self.results: list[BenchmarkResult] = []
    
    async def run_scenario_a(self) -> BenchmarkResult:
        """
        Scenario A: Inference-Only Benchmark
        
        Tests pure inference throughput without training.
        Measures prefix caching effectiveness with repeated system prompts.
        """
        print("\n" + "=" * 60)
        print("SCENARIO A: Inference-Only Benchmark")
        print("=" * 60)
        
        result = BenchmarkResult(
            backend=self.backend_type,
            scenario="A_inference_only",
            model=self.model,
            timestamp=datetime.now().isoformat(),
            config={
                "num_samples": self.num_samples,
                "max_tokens": self.max_tokens,
                "num_rollouts": self.num_rollouts,
            },
        )
        
        manager = BackendManager(
            self.backend_type, self.model, self.work_dir,
            vllm_tp_size=self.vllm_tp_size,
            preserve_cache=self.preserve_cache,
            sglang_gpu_ids=self.sglang_gpu_ids,
            sglang_cache_method=self.sglang_cache_method,
            vllm_cache_method=self.vllm_cache_method,
            vllm_gpu_ids=self.vllm_gpu_ids,
            training_gpu_ids=self.training_gpu_ids,
            in_process=self.in_process,
        )
        
        try:
            print(f"Initializing {self.backend_type} backend...")
            await manager.initialize()
            
            prompts = self.dataset.get_prompts(self.num_samples, with_system=True)
            
            # Warmup
            print("Warmup run...")
            await manager.generate(prompts[0]["messages"], self.max_tokens)
            
            # Run benchmark with batching
            print(f"Running inference on {len(prompts)} prompts (batch_size={self.batch_size})...")
            
            total_tokens = 0
            ttft_values = []
            latency_values = []
            
            start_time = time.perf_counter()
            
            # Build all tasks
            all_tasks = []
            for prompt in prompts:
                for rollout in range(self.num_rollouts):
                    all_tasks.append(prompt["messages"])
            
            # Process in batches
            total_tasks = len(all_tasks)
            completed = 0
            
            for batch_start in range(0, total_tasks, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_tasks)
                batch_messages = all_tasks[batch_start:batch_end]
                
                # Run batch concurrently
                batch_tasks = [
                    manager.generate(messages, self.max_tokens)
                    for messages in batch_messages
                ]
                batch_results = await asyncio.gather(*batch_tasks)
                
                for response, ttft_ms, latency_ms in batch_results:
                    total_tokens += len(response.split())  # Approximate
                    ttft_values.append(ttft_ms)
                    latency_values.append(latency_ms)
                
                completed += len(batch_messages)
                prompts_done = completed // self.num_rollouts
                if prompts_done % 10 == 0 or completed == total_tasks:
                    print(f"  Completed {prompts_done}/{len(prompts)} prompts ({completed}/{total_tasks} rollouts)")
            
            total_time = time.perf_counter() - start_time
            
            # Get accurate cache hit rate: prefer real server stats > estimation
            cache_hit_rate = 0.0
            cache_stats = await manager.get_real_cache_stats()
            if "hit_rate" in cache_stats:
                cache_hit_rate = cache_stats["hit_rate"]
            elif ttft_values:
                cache_hit_rate = self._estimate_cache_hit_rate(ttft_values)
            
            # Calculate metrics
            result.inference_metrics = InferenceMetrics(
                tokens_per_second=total_tokens / total_time,
                time_to_first_token_ms=sum(ttft_values) / len(ttft_values),
                end_to_end_latency_ms=sum(latency_values) / len(latency_values),
                prefix_cache_hit_rate=cache_hit_rate,
                memory_utilization_gb=manager.get_memory_usage(),
                total_tokens_generated=total_tokens,
                total_requests=len(prompts) * self.num_rollouts,
            )
            
            print(f"\nResults:")
            print(f"  Throughput: {result.inference_metrics.tokens_per_second:.2f} tokens/s")
            print(f"  Avg TTFT: {result.inference_metrics.time_to_first_token_ms:.2f} ms")
            print(f"  Avg Latency: {result.inference_metrics.end_to_end_latency_ms:.2f} ms")
            print(f"  Cache Hit Rate: {result.inference_metrics.prefix_cache_hit_rate:.2%}")
            
        finally:
            await manager.cleanup()
        
        self.results.append(result)
        return result
    
    async def run_scenario_b(self) -> BenchmarkResult:
        """
        Scenario B: Single Training Iteration
        
        Measures: Rollouts → Pack tensors → Train → Weight sync
        """
        print("\n" + "=" * 60)
        print("SCENARIO B: Single Training Iteration")
        print("=" * 60)
        
        result = BenchmarkResult(
            backend=self.backend_type,
            scenario="B_single_iteration",
            model=self.model,
            timestamp=datetime.now().isoformat(),
            config={
                "num_samples": self.num_samples,
                "num_rollouts": self.num_rollouts,
                "max_tokens": self.max_tokens,
            },
        )
        
        manager = BackendManager(
            self.backend_type, self.model, self.work_dir,
            vllm_tp_size=self.vllm_tp_size,
            preserve_cache=self.preserve_cache,
            sglang_gpu_ids=self.sglang_gpu_ids,
            sglang_cache_method=self.sglang_cache_method,
            vllm_cache_method=self.vllm_cache_method,
            vllm_gpu_ids=self.vllm_gpu_ids,
            training_gpu_ids=self.training_gpu_ids,
            in_process=self.in_process,
        )
        
        try:
            print(f"Initializing {self.backend_type} backend...")
            await manager.initialize()
            
            prompts = self.dataset.get_prompts(self.num_samples)
            
            # Phase 1: Inference (Rollouts) - BATCHED for performance
            # Use generate_for_training if training is enabled to get Choice objects
            use_training_gen = self.enable_training
            gen_method = "generate_for_training" if use_training_gen else "generate"
            batch_size = self.batch_size
            print(f"Phase 1: Generating rollouts ({len(prompts)} prompts × {self.num_rollouts} rollouts) [mode: {gen_method}, batch={batch_size}]...")
            inference_start = time.perf_counter()
            
            trajectories = []  # For training: list of Trajectory objects
            trajectory_dicts = []  # For metrics: list of dicts
            ttft_values = []
            latency_values = []
            
            # Build all tasks for batched execution
            all_tasks_metadata = []  # (prompt_idx, rollout_idx, messages, expected_answer)
            for prompt_idx, prompt in enumerate(prompts):
                for rollout_idx in range(self.num_rollouts):
                    all_tasks_metadata.append((prompt_idx, rollout_idx, prompt["messages"], prompt["expected_answer"]))
            
            total_tasks = len(all_tasks_metadata)
            completed = 0
            
            # Process in batches concurrently
            for batch_start in range(0, total_tasks, batch_size):
                batch_end = min(batch_start + batch_size, total_tasks)
                batch_metadata = all_tasks_metadata[batch_start:batch_end]
                
                # Create async tasks for this batch
                if use_training_gen:
                    batch_coros = [
                        manager.generate_for_training(messages, self.max_tokens, self.temperature)
                        for _, _, messages, _ in batch_metadata
                    ]
                else:
                    batch_coros = [
                        manager.generate(messages, self.max_tokens, self.temperature)
                        for _, _, messages, _ in batch_metadata
                    ]
                
                # Execute batch concurrently
                batch_results = await asyncio.gather(*batch_coros)
                
                # Process results
                from art.trajectories import Trajectory
                for idx, gen_result in enumerate(batch_results):
                    prompt_idx, rollout_idx, messages, expected_answer = batch_metadata[idx]
                    
                    if use_training_gen:
                        choice, response, ttft_ms, latency_ms, _cache_info = gen_result
                        reward = self.dataset.compute_reward(response, expected_answer)
                        
                        traj = Trajectory(
                            messages_and_choices=[
                                *messages,
                                choice,
                            ],
                            reward=reward,
                        )
                        trajectories.append(traj)
                        trajectory_dicts.append({
                            "prompt": messages,
                            "response": choice.message.content,
                            "reward": reward,
                        })
                    else:
                        response, ttft_ms, latency_ms = gen_result
                        reward = self.dataset.compute_reward(response, expected_answer)
                        trajectory_dicts.append({
                            "prompt": messages,
                            "response": response,
                            "reward": reward,
                        })
                    
                    ttft_values.append(ttft_ms)
                    latency_values.append(latency_ms)
                
                completed += len(batch_metadata)
                prompts_done = (completed + self.num_rollouts - 1) // self.num_rollouts
                print(f"  Completed {min(prompts_done, len(prompts))}/{len(prompts)} prompts ({completed}/{total_tasks} rollouts)")
            
            inference_time = time.perf_counter() - inference_start
            
            # Phase 2: Training
            print("Phase 2: Training step...")
            training_start = time.perf_counter()
            
            if self.enable_training and trajectories:
                # Perform actual training with proper Trajectory objects
                from art.trajectories import TrajectoryGroup
                
                # Group trajectories (all rollouts for same prompt go together)
                trajectory_groups = []
                for i in range(0, len(trajectories), self.num_rollouts):
                    group_trajs = trajectories[i:i + self.num_rollouts]
                    if group_trajs:
                        trajectory_groups.append(TrajectoryGroup(trajectories=group_trajs))
                
                print(f"  Training on {len(trajectory_groups)} trajectory groups...")
                try:
                    train_result = await manager.backend.train(
                        manager._model_obj,
                        trajectory_groups,
                        learning_rate=self.learning_rate,
                        skip_advantage_filter=True,  # Force training even with equal rewards
                        verbose=True,  # Enable verbose output for debugging
                    )
                    result.training_metrics.loss = train_result.metrics.get("loss", 0.0)
                    result.training_metrics.grad_norm = train_result.metrics.get("grad_norm", 0.0)
                    print(f"  ✓ Training complete: step={train_result.step}, loss={result.training_metrics.loss:.4f}")
                except Exception as e:
                    print(f"  ✗ Training failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Skip training - just measure overhead
                print("  (Training disabled - use --enable-training to run actual training)")
            
            avg_reward = sum(t["reward"] for t in trajectory_dicts) / len(trajectory_dicts) if trajectory_dicts else 0
            print(f"  Generated {len(trajectory_dicts)} trajectories, avg reward: {avg_reward:.2f}")
            
            training_time = time.perf_counter() - training_start
            
            # Phase 3: Weight sync
            print("Phase 3: Weight sync...")
            sync_start = time.perf_counter()
            
            # After training, the model step has advanced - update model_name to point
            # to the new checkpoint. The old model name (e.g., "model@7") no longer exists
            # after vLLM restarts; only the new one (e.g., "model@8") is registered.
            if self.enable_training:
                manager.model_name = manager.backend._model_inference_name(manager._model_obj)
                print(f"  Updated model name to: {manager.model_name}")
            
            # Weight sync happens automatically in training, measure verification
            # Retry loop in case vLLM is still starting up after training
            for retry in range(10):
                try:
                    await manager.generate(prompts[0]["messages"], 10)  # Quick inference to verify
                    break
                except Exception as e:
                    if retry < 9:
                        print(f"  Waiting for server to be ready (attempt {retry + 1}/10)...")
                        await asyncio.sleep(3)
                    else:
                        raise e
            
            sync_time = time.perf_counter() - sync_start
            
            total_time = inference_time + training_time + sync_time
            
            result.training_metrics.inference_time_s = inference_time
            result.training_metrics.training_step_time_s = training_time
            result.training_metrics.weight_sync_time_s = sync_time
            result.training_metrics.total_iteration_time_s = total_time
            result.training_metrics.gpu_memory_training_gb = manager.get_memory_usage()
            
            print(f"\nResults:")
            print(f"  Inference time: {inference_time:.2f}s")
            print(f"  Training time: {training_time:.2f}s")
            print(f"  Weight sync time: {sync_time:.2f}s")
            print(f"  Total iteration: {total_time:.2f}s")
            
        finally:
            await manager.cleanup()
        
        self.results.append(result)
        return result
    
    async def run_scenario_c(self) -> BenchmarkResult:
        """
        Scenario C: Full RL Training Loop
        
        Runs multiple training iterations to measure amortized costs.
        """
        print("\n" + "=" * 60)
        print(f"SCENARIO C: Full RL Training Loop ({self.num_iterations} iterations)")
        print("=" * 60)
        
        result = BenchmarkResult(
            backend=self.backend_type,
            scenario="C_full_training_loop",
            model=self.model,
            timestamp=datetime.now().isoformat(),
            config={
                "num_iterations": self.num_iterations,
                "num_samples": self.num_samples,
                "num_rollouts": self.num_rollouts,
                "max_tokens": self.max_tokens,
            },
        )
        
        manager = BackendManager(
            self.backend_type, self.model, self.work_dir,
            vllm_tp_size=self.vllm_tp_size,
            preserve_cache=self.preserve_cache,
            sglang_gpu_ids=self.sglang_gpu_ids,
            sglang_cache_method=self.sglang_cache_method,
            vllm_cache_method=self.vllm_cache_method,
            vllm_gpu_ids=self.vllm_gpu_ids,
            training_gpu_ids=self.training_gpu_ids,
            in_process=self.in_process,
        )
        
        try:
            print(f"Initializing {self.backend_type} backend...")
            await manager.initialize()
            
            prompts = self.dataset.get_prompts(self.num_samples)
            
            iteration_times = []
            inference_times = []
            training_times = []
            sync_times = []
            losses = []
            all_ttft_values = []
            # Per-request cache tracking (same method for both backends)
            total_prompt_tokens = 0
            total_cached_tokens = 0
            per_iteration_cache_rates = []  # Track cache rate per iteration
            
            use_training_gen = self.enable_training
            gen_method = "generate_for_training" if use_training_gen else "generate"
            batch_size = self.batch_size
            
            for iteration in range(self.num_iterations):
                print(f"\n--- Iteration {iteration + 1}/{self.num_iterations} [mode: {gen_method}, batch={batch_size}] ---")
                iter_start = time.perf_counter()
                
                # Phase 1: Generate rollouts (with batching support)
                inference_start = time.perf_counter()
                trajectories = []  # For training: list of Trajectory objects
                trajectory_dicts = []  # For metrics
                ttft_values = []
                
                # Build all tasks for batched execution
                all_tasks = []
                task_metadata = []  # Track (prompt_idx, rollout_idx, prompt, expected_answer)
                iter_prompt_tokens = 0
                iter_cached_tokens = 0
                
                for i, prompt in enumerate(prompts):
                    for rollout in range(self.num_rollouts):
                        task_metadata.append((i, rollout, prompt["messages"], prompt["expected_answer"]))
                
                # Process in batches
                total_tasks = len(task_metadata)
                completed = 0
                
                for batch_start in range(0, total_tasks, batch_size):
                    batch_end = min(batch_start + batch_size, total_tasks)
                    batch_metadata = task_metadata[batch_start:batch_end]
                    
                    # Create async tasks for this batch
                    if use_training_gen:
                        batch_tasks = [
                            manager.generate_for_training(messages, self.max_tokens, self.temperature)
                            for _, _, messages, _ in batch_metadata
                        ]
                    else:
                        batch_tasks = [
                            manager.generate(messages, self.max_tokens, self.temperature)
                            for _, _, messages, _ in batch_metadata
                        ]
                    
                    # Execute batch concurrently
                    batch_results = await asyncio.gather(*batch_tasks)
                    
                    # Process results
                    for idx, gen_result in enumerate(batch_results):
                        prompt_idx, rollout_idx, messages, expected_answer = batch_metadata[idx]
                        
                        if use_training_gen:
                            choice, response, ttft_ms, latency_ms, cache_info = gen_result
                            reward = self.dataset.compute_reward(response, expected_answer)
                            
                            # Accumulate per-request cache tokens
                            iter_prompt_tokens += cache_info["prompt_tokens"]
                            iter_cached_tokens += cache_info["cached_tokens"]
                            
                            from art.trajectories import Trajectory
                            traj = Trajectory(
                                messages_and_choices=[
                                    *messages,
                                    choice,
                                ],
                                reward=reward,
                            )
                            trajectories.append(traj)
                            trajectory_dicts.append({
                                "prompt": messages,
                                "response": choice.message.content,
                                "reward": reward,
                            })
                        else:
                            response, ttft_ms, latency_ms = gen_result
                            reward = self.dataset.compute_reward(response, expected_answer)
                            trajectory_dicts.append({
                                "prompt": messages,
                                "response": response,
                                "reward": reward,
                            })
                        
                        ttft_values.append(ttft_ms)
                    
                    completed += len(batch_metadata)
                    prompts_done = (completed + self.num_rollouts - 1) // self.num_rollouts
                    if prompts_done % 10 == 0 or completed == total_tasks:
                        print(f"    Completed {min(prompts_done, len(prompts))}/{len(prompts)} prompts ({completed}/{total_tasks} rollouts)")
                
                inference_time = time.perf_counter() - inference_start
                inference_times.append(inference_time)
                all_ttft_values.extend(ttft_values)
                
                # Track per-iteration cache stats from response usage
                total_prompt_tokens += iter_prompt_tokens
                total_cached_tokens += iter_cached_tokens
                if iter_prompt_tokens > 0:
                    iter_cache_rate = iter_cached_tokens / iter_prompt_tokens
                    per_iteration_cache_rates.append(iter_cache_rate)
                    print(f"  Cache (from response): {iter_cache_rate:.2%} ({iter_cached_tokens}/{iter_prompt_tokens} tokens cached)")
                else:
                    per_iteration_cache_rates.append(0.0)
                
                # Phase 2: Training
                training_start = time.perf_counter()
                iteration_loss = 0.0
                
                if self.enable_training and trajectories:
                    from art.trajectories import TrajectoryGroup
                    
                    # Group trajectories
                    trajectory_groups = []
                    for i in range(0, len(trajectories), self.num_rollouts):
                        group_trajs = trajectories[i:i + self.num_rollouts]
                        if group_trajs:
                            trajectory_groups.append(TrajectoryGroup(trajectories=group_trajs))
                    
                    try:
                        train_result = await manager.backend.train(
                            manager._model_obj,
                            trajectory_groups,
                            learning_rate=self.learning_rate,
                            skip_advantage_filter=True,  # Force training even with equal rewards
                            verbose=True,  # Enable verbose output for debugging
                        )
                        iteration_loss = train_result.metrics.get("loss", 0.0)
                        losses.append(iteration_loss)
                        print(f"  ✓ Trained: loss={iteration_loss:.4f}")
                    except Exception as e:
                        print(f"  ✗ Training failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    if iteration == 0:
                        print("  (Training disabled - use --enable-training)")
                
                training_time = time.perf_counter() - training_start
                training_times.append(training_time)
                
                # Phase 3: Weight sync verification
                sync_start = time.perf_counter()
                
                # After training, the model step has advanced - update model_name to point
                # to the new checkpoint. The old model name no longer exists after vLLM restarts.
                if self.enable_training:
                    manager.model_name = manager.backend._model_inference_name(manager._model_obj)
                    if iteration == 0:  # Only log on first iteration to reduce noise
                        print(f"  Updated model name to: {manager.model_name}")
                
                # Retry loop in case vLLM is still starting up after training
                for retry in range(10):
                    try:
                        await manager.generate(prompts[0]["messages"], 10)  # Quick inference to verify
                        break
                    except Exception as e:
                        if retry < 9:
                            print(f"  Waiting for server to be ready (attempt {retry + 1}/10)...")
                            await asyncio.sleep(3)
                        else:
                            raise e
                sync_time = time.perf_counter() - sync_start
                sync_times.append(sync_time)
                
                iter_time = time.perf_counter() - iter_start
                iteration_times.append(iter_time)
                
                avg_reward = sum(t["reward"] for t in trajectory_dicts) / len(trajectory_dicts) if trajectory_dicts else 0
                print(f"  Inference: {inference_time:.2f}s, Training: {training_time:.2f}s, Sync: {sync_time:.2f}s")
                print(f"  Total: {iter_time:.2f}s, Avg Reward: {avg_reward:.2f}, Trajectories: {len(trajectory_dicts)}")
            
            avg_iter_time = sum(iteration_times) / len(iteration_times)
            total_rollouts = len(prompts) * self.num_rollouts
            samples_per_second = total_rollouts / avg_iter_time
            
            # Store timing breakdowns
            result.training_metrics.inference_time_s = sum(inference_times) / len(inference_times)
            result.training_metrics.training_step_time_s = sum(training_times) / len(training_times)
            result.training_metrics.weight_sync_time_s = sum(sync_times) / len(sync_times)
            result.training_metrics.total_iteration_time_s = sum(iteration_times)
            result.training_metrics.gpu_memory_training_gb = manager.get_memory_usage()
            result.training_metrics.loss = sum(losses) / len(losses) if losses else 0.0
            
            result.inference_metrics.tokens_per_second = samples_per_second
            result.inference_metrics.time_to_first_token_ms = sum(all_ttft_values) / len(all_ttft_values) if all_ttft_values else 0
            result.inference_metrics.total_requests = total_rollouts * self.num_iterations
            
            # === Cache Hit Rate (unified measurement for both backends) ===
            # Priority 1: Per-request cached_tokens from response usage (same API, same metric)
            # Priority 2: Server-side Prometheus/metrics counters
            # Priority 3: TTFT estimation (last resort, clearly labeled)
            cache_hit_rate = 0.0
            cache_source = "none"
            
            if total_prompt_tokens > 0:
                # Priority 1: Per-request tracking from API response usage
                # This is the SAME measurement for both SGLang and vLLM
                cache_hit_rate = total_cached_tokens / total_prompt_tokens
                cache_source = "per_request_usage"
                print(f"\n  Cache stats (per-request, unified):")
                print(f"    Total prompt tokens: {total_prompt_tokens:,}")
                print(f"    Cached tokens:       {total_cached_tokens:,}")
                print(f"    Cache hit rate:       {cache_hit_rate:.2%}")
                if per_iteration_cache_rates:
                    print(f"    Per-iteration rates:  {', '.join(f'{r:.1%}' for r in per_iteration_cache_rates)}")
            
            if cache_hit_rate == 0.0:
                # Priority 2: Server-side counters (Prometheus /metrics or /get_server_info)
                cache_stats = await manager.get_real_cache_stats()
                if "hit_rate" in cache_stats:
                    cache_hit_rate = cache_stats["hit_rate"]
                    cache_source = f"server_{cache_stats.get('source', 'unknown')}"
                    print(f"\n  Cache hit rate (from server {cache_source}): {cache_hit_rate:.2%}")
                elif all_ttft_values:
                    # Priority 3: TTFT estimation (fallback, clearly labeled as estimate)
                    cache_hit_rate = self._estimate_cache_hit_rate(all_ttft_values)
                    cache_source = "ttft_estimate"
                    print(f"\n  Cache hit rate (TTFT estimate - less reliable): {cache_hit_rate:.2%}")
            
            result.inference_metrics.prefix_cache_hit_rate = cache_hit_rate
            
            print(f"\nResults:")
            print(f"  Total time: {sum(iteration_times):.2f}s")
            print(f"  Avg iteration time: {avg_iter_time:.2f}s")
            print(f"  Avg inference time: {sum(inference_times) / len(inference_times):.2f}s")
            print(f"  Avg training time: {sum(training_times) / len(training_times):.2f}s")
            print(f"  Avg sync time: {sum(sync_times) / len(sync_times):.2f}s")
            print(f"  Training throughput: {samples_per_second:.2f} samples/s")
            print(f"  Cache hit rate: {cache_hit_rate:.2%} (source: {cache_source})")
            if losses:
                print(f"  Avg loss: {result.training_metrics.loss:.4f}")
            
        finally:
            await manager.cleanup()
        
        self.results.append(result)
        return result
    
    async def run_scenario_d(self) -> BenchmarkResult:
        """
        Scenario D: Prefix Cache Utilization Test
        
        Specifically tests prefix caching with repeated system prompts.
        """
        print("\n" + "=" * 60)
        print("SCENARIO D: Prefix Cache Utilization")
        print("=" * 60)
        
        result = BenchmarkResult(
            backend=self.backend_type,
            scenario="D_prefix_cache",
            model=self.model,
            timestamp=datetime.now().isoformat(),
            config={
                "num_samples": 20,
                "num_repetitions": 5,
            },
        )
        
        manager = BackendManager(
            self.backend_type, self.model, self.work_dir,
            vllm_tp_size=self.vllm_tp_size,
            preserve_cache=self.preserve_cache,
            sglang_gpu_ids=self.sglang_gpu_ids,
            sglang_cache_method=self.sglang_cache_method,
            vllm_cache_method=self.vllm_cache_method,
            vllm_gpu_ids=self.vllm_gpu_ids,
            training_gpu_ids=self.training_gpu_ids,
            in_process=self.in_process,
        )
        
        try:
            print(f"Initializing {self.backend_type} backend...")
            await manager.initialize()
            
            prompts = self.dataset.get_prompts(20)
            
            # Test 1: Cold cache (first request for each prompt)
            print("\nTest 1: Cold cache (first requests)...")
            cold_ttft = []
            for prompt in prompts:
                _, ttft_ms, _ = await manager.generate(prompt["messages"], 50)
                cold_ttft.append(ttft_ms)
            
            avg_cold_ttft = sum(cold_ttft) / len(cold_ttft)
            print(f"  Avg TTFT (cold): {avg_cold_ttft:.2f} ms")
            
            # Test 2: Warm cache (repeat same prompts)
            print("\nTest 2: Warm cache (repeated requests)...")
            warm_ttft = []
            for _ in range(5):  # Repeat 5 times
                for prompt in prompts:
                    _, ttft_ms, _ = await manager.generate(prompt["messages"], 50)
                    warm_ttft.append(ttft_ms)
            
            avg_warm_ttft = sum(warm_ttft) / len(warm_ttft)
            print(f"  Avg TTFT (warm): {avg_warm_ttft:.2f} ms")
            
            # Calculate cache hit rate (speedup)
            speedup = avg_cold_ttft / avg_warm_ttft if avg_warm_ttft > 0 else 1.0
            cache_hit_rate = max(0, 1 - (avg_warm_ttft / avg_cold_ttft)) if avg_cold_ttft > 0 else 0
            
            result.inference_metrics.time_to_first_token_ms = avg_warm_ttft
            result.inference_metrics.prefix_cache_hit_rate = cache_hit_rate
            
            print(f"\nResults:")
            print(f"  Cold TTFT: {avg_cold_ttft:.2f} ms")
            print(f"  Warm TTFT: {avg_warm_ttft:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Estimated Cache Hit Rate: {cache_hit_rate:.2%}")
            
        finally:
            await manager.cleanup()
        
        self.results.append(result)
        return result
    
    async def run_scenario_e(self) -> BenchmarkResult:
        """
        Scenario E: ART Sleep/Wake Benchmark (vLLM only)
        
        Tests the ART architecture's time-multiplexed GPU sharing:
        - Measures sleep (offload to CPU) timing
        - Measures wake (reload from CPU) timing
        - Measures training with freed GPU memory
        - Compares overhead vs. full restart approach
        
        This scenario only works with vLLM backend.
        """
        print("\n" + "=" * 60)
        print("SCENARIO E: ART Sleep/Wake Benchmark (vLLM)")
        print("=" * 60)
        
        if self.backend_type != "vllm":
            print(f"  Skipping: Scenario E only supports vLLM backend (got {self.backend_type})")
            return BenchmarkResult(
                backend=self.backend_type,
                scenario="E_sleep_wake_skipped",
                model=self.model,
                timestamp=datetime.now().isoformat(),
            )
        
        result = BenchmarkResult(
            backend=self.backend_type,
            scenario="E_sleep_wake",
            model=self.model,
            timestamp=datetime.now().isoformat(),
            config={
                "num_samples": self.num_samples,
                "num_iterations": self.num_iterations,
                "sleep_level": 2,  # Discard KV cache for maximum memory
            },
        )
        
        manager = BackendManager(
            self.backend_type, self.model, self.work_dir,
            vllm_tp_size=self.vllm_tp_size,
            preserve_cache=True,  # Required for sleep/wake
            sglang_gpu_ids=self.sglang_gpu_ids,
            sglang_cache_method=self.sglang_cache_method,
            vllm_cache_method="sleep_wake",  # Force sleep/wake method
            vllm_gpu_ids=self.vllm_gpu_ids,
            training_gpu_ids=self.training_gpu_ids,
            in_process=self.in_process,
        )
        
        try:
            print(f"Initializing vLLM backend with sleep/wake support...")
            await manager.initialize()
            
            prompts = self.dataset.get_prompts(self.num_samples)
            
            # Track metrics across iterations
            sleep_times = []
            wake_times = []
            training_times = []
            inference_times = []
            memory_freed = []
            
            print(f"\nRunning {self.num_iterations} sleep/wake cycles...")
            print("=" * 60)
            
            for iteration in range(self.num_iterations):
                print(f"\n--- Iteration {iteration + 1}/{self.num_iterations} ---")
                
                # Phase 1: Inference (generate some rollouts)
                print("  Phase 1: Inference")
                inference_start = time.perf_counter()
                
                ttft_values = []
                for i, prompt in enumerate(prompts[:5]):  # Use subset for speed
                    _, ttft_ms, _ = await manager.generate(prompt["messages"], self.max_tokens)
                    ttft_values.append(ttft_ms)
                
                inference_time = time.perf_counter() - inference_start
                inference_times.append(inference_time)
                print(f"    Inference: {inference_time:.2f}s, Avg TTFT: {sum(ttft_values)/len(ttft_values):.2f}ms")
                
                # Phase 2: Sleep (offload to CPU)
                print("  Phase 2: Sleep (offload to CPU)")
                sleep_metrics = await manager.sleep(level=2)
                sleep_times.append(sleep_metrics.get("total_time_s", 0))
                memory_freed.append(sleep_metrics.get("memory_freed_gb", 0))
                
                # Phase 3: Simulated training (GPU is now free)
                print("  Phase 3: Training (simulated)")
                training_start = time.perf_counter()
                
                # Simulate training workload - allocate tensors to verify GPU is free
                if torch.cuda.is_available():
                    # Try to allocate a large tensor (simulating training)
                    try:
                        # Allocate ~2GB to simulate training model
                        training_tensor = torch.randn(512, 1024, 1024, device='cuda')
                        # Simulate some computation
                        for _ in range(10):
                            training_tensor = training_tensor * 0.99 + torch.randn_like(training_tensor) * 0.01
                        del training_tensor
                        torch.cuda.empty_cache()
                        print("    ✓ Successfully allocated and used GPU memory for training")
                    except torch.cuda.OutOfMemoryError:
                        print("    ✗ GPU memory not fully freed - OOM during simulated training")
                
                training_time = time.perf_counter() - training_start
                training_times.append(training_time)
                print(f"    Training time: {training_time:.2f}s")
                
                # Phase 4: Wake up (reload from CPU)
                print("  Phase 4: Wake up (reload from CPU)")
                wake_metrics = await manager.wake_up()
                wake_times.append(wake_metrics.get("total_time_s", 0))
                
                # Phase 5: Verify inference works after wake
                print("  Phase 5: Verify inference")
                try:
                    response, ttft_ms, latency_ms = await manager.generate(
                        prompts[0]["messages"], 50
                    )
                    print(f"    ✓ Inference working: TTFT={ttft_ms:.2f}ms")
                except Exception as e:
                    print(f"    ✗ Inference failed after wake: {e}")
                
                # Iteration summary
                iter_overhead = sleep_times[-1] + wake_times[-1]
                print(f"  Iteration overhead: {iter_overhead:.2f}s (sleep={sleep_times[-1]:.2f}s, wake={wake_times[-1]:.2f}s)")
            
            # Calculate aggregate metrics
            avg_sleep = sum(sleep_times) / len(sleep_times) if sleep_times else 0
            avg_wake = sum(wake_times) / len(wake_times) if wake_times else 0
            avg_training = sum(training_times) / len(training_times) if training_times else 0
            avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
            avg_memory_freed = sum(memory_freed) / len(memory_freed) if memory_freed else 0
            total_overhead = avg_sleep + avg_wake
            
            result.training_metrics = TrainingMetrics(
                sleep_time_s=avg_sleep,
                wake_time_s=avg_wake,
                sleep_wake_overhead_s=total_overhead,
                training_step_time_s=avg_training,
                inference_time_s=avg_inference,
                total_iteration_time_s=sum(sleep_times) + sum(wake_times) + sum(training_times) + sum(inference_times),
                memory_freed_during_sleep_gb=avg_memory_freed,
                gpu_memory_training_gb=manager.get_memory_usage(),
            )
            
            print("\n" + "=" * 60)
            print("SCENARIO E RESULTS: ART Sleep/Wake Performance")
            print("=" * 60)
            print(f"  Average sleep time:        {avg_sleep:.2f}s")
            print(f"  Average wake time:         {avg_wake:.2f}s")
            print(f"  Average overhead per iter: {total_overhead:.2f}s")
            print(f"  Average memory freed:      {avg_memory_freed:.2f}GB")
            print(f"  Average training time:     {avg_training:.2f}s")
            print(f"  Average inference time:    {avg_inference:.2f}s")
            print()
            print("  Breakdown by phase:")
            print(f"    Sleep:    {avg_sleep:.2f}s ({avg_sleep/total_overhead*100:.1f}% of overhead)")
            print(f"    Wake:     {avg_wake:.2f}s ({avg_wake/total_overhead*100:.1f}% of overhead)")
            print()
            if avg_training > 0:
                overhead_ratio = total_overhead / (total_overhead + avg_training)
                print(f"  Overhead ratio: {overhead_ratio*100:.1f}% of cycle time")
            
        finally:
            await manager.cleanup()
        
        self.results.append(result)
        return result
    
    def _estimate_cache_hit_rate(self, ttft_values: list[float]) -> float:
        """
        Estimate cache hit rate from TTFT variance.
        
        Uses warmup requests to establish a cold baseline, then counts
        requests that are significantly faster as cache hits.
        """
        if len(ttft_values) < 10:
            # Not enough data for reliable estimation
            if len(ttft_values) < 2:
                return 0.0
            # Fallback to simple comparison for small samples
            first_ttft = ttft_values[0]
            subsequent_avg = sum(ttft_values[1:]) / len(ttft_values[1:])
            if first_ttft > 0:
                return max(0, 1 - (subsequent_avg / first_ttft))
            return 0.0
        
        # Skip first few requests as warmup to establish cold baseline
        warmup_count = min(3, len(ttft_values) // 4)
        warmup_ttft = ttft_values[:warmup_count]
        warm_ttft_list = ttft_values[warmup_count:]
        
        # Cold baseline is the average of warmup requests
        cold_ttft = sum(warmup_ttft) / len(warmup_ttft) if warmup_ttft else 0
        
        if cold_ttft <= 0:
            return 0.0
        
        # A cache hit is when TTFT is faster than cold baseline
        # Using 80% threshold - if request is 20%+ faster, it's likely a cache hit
        # NOTE: 50% was too aggressive - prefix caching only speeds up prefill,
        # which is a fraction of total time. When non-streaming, TTFT=total_ms,
        # so the speedup from caching is much smaller than 50%.
        speedup_threshold = 0.8
        cache_hits = sum(1 for t in warm_ttft_list if t < cold_ttft * speedup_threshold)
        
        return cache_hits / len(warm_ttft_list) if warm_ttft_list else 0.0
    
    def save_results(self, output_path: str):
        """Save results to JSON and CSV."""
        # Save JSON
        json_path = output_path + ".json"
        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"\nSaved JSON results to: {json_path}")
        
        # Save CSV
        csv_path = output_path + ".csv"
        if self.results:
            fieldnames = [
                "backend", "scenario", "timestamp",
                "tokens_per_second", "time_to_first_token_ms",
                "end_to_end_latency_ms", "prefix_cache_hit_rate",
                "training_step_time_s", "weight_sync_time_s",
                "total_iteration_time_s", "loss",
                # ART sleep/wake metrics
                "sleep_time_s", "wake_time_s", "sleep_wake_overhead_s",
                "memory_freed_during_sleep_gb",
            ]
            
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for r in self.results:
                    row = {
                        "backend": r.backend,
                        "scenario": r.scenario,
                        "timestamp": r.timestamp,
                        "tokens_per_second": r.inference_metrics.tokens_per_second,
                        "time_to_first_token_ms": r.inference_metrics.time_to_first_token_ms,
                        "end_to_end_latency_ms": r.inference_metrics.end_to_end_latency_ms,
                        "prefix_cache_hit_rate": r.inference_metrics.prefix_cache_hit_rate,
                        "training_step_time_s": r.training_metrics.training_step_time_s,
                        "weight_sync_time_s": r.training_metrics.weight_sync_time_s,
                        "total_iteration_time_s": r.training_metrics.total_iteration_time_s,
                        "loss": r.training_metrics.loss,
                        # ART sleep/wake metrics
                        "sleep_time_s": r.training_metrics.sleep_time_s,
                        "wake_time_s": r.training_metrics.wake_time_s,
                        "sleep_wake_overhead_s": r.training_metrics.sleep_wake_overhead_s,
                        "memory_freed_during_sleep_gb": r.training_metrics.memory_freed_during_sleep_gb,
                    }
                    writer.writerow(row)
            
            print(f"Saved CSV results to: {csv_path}")


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang vs vLLM with Megatron Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all scenarios with SGLang
    python benchmark_sglang_vs_vllm_megatron.py --backend sglang --scenario all
    
    # Run inference-only with both backends
    python benchmark_sglang_vs_vllm_megatron.py --backend both --scenario A
    
    # Run prefix cache test with custom samples
    python benchmark_sglang_vs_vllm_megatron.py --backend sglang --scenario D --num-samples 100
        """,
    )
    
    parser.add_argument(
        "--backend",
        choices=["sglang", "vllm", "both"],
        default="sglang",
        help="Backend to benchmark (default: sglang)",
    )
    parser.add_argument(
        "--scenario",
        choices=["A", "B", "C", "D", "E", "all"],
        default="all",
        help="Scenario to run: A=inference, B=single iter, C=full loop, D=cache test, E=sleep/wake (default: all)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model to use (default: Qwen/Qwen3-30B-A3B-Instruct-2507)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples per benchmark (default: 50)",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=4,
        help="Number of rollouts per prompt (default: 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per generation (default: 256)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of training iterations for scenario C (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results",
        help="Output file prefix (default: benchmark_results)",
    )
    parser.add_argument(
        "--work-dir",
        default=".art/benchmark",
        help="Working directory (default: .art/benchmark)",
    )
    parser.add_argument(
        "--vllm-tp-size",
        type=int,
        default=None,
        help="Tensor parallel size for vLLM (default: 1 for fair comparison with SGLang)",
    )
    parser.add_argument(
        "--enable-training",
        action="store_true",
        help="Enable actual training in Scenarios B and C (requires proper backend setup)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for training (default: 1e-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for concurrent inference requests (default: 32)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation (default: 1.0, higher = more variance)",
    )
    parser.add_argument(
        "--preserve-cache",
        action="store_true",
        help="Keep inference engine running during training to preserve prefix cache. "
             "Works for both SGLang (RadixAttention) and vLLM (automatic prefix caching). "
             "Requires spare GPUs for Megatron (inference uses GPU 0 by default, Megatron uses rest). "
             "This can significantly improve performance in multi-iteration scenarios.",
    )
    parser.add_argument(
        "--sglang-gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for SGLang (e.g., '0' or '0,1'). "
             "Only used with --preserve-cache. Remaining GPUs are used for Megatron.",
    )
    parser.add_argument(
        "--sglang-cache-method",
        type=str,
        choices=["freeze", "sleep_wake", "hot_reload", "restart", "sleep"],
        default="freeze",
        help="Cache method for SGLang during training (default: freeze). "
             "freeze: (RECOMMENDED) SIGSTOP the SGLang process during training, SIGCONT after. "
             "  Zero CPU contention, cache 100%% preserved, instant resume. "
             "  With 4+ GPUs, training speed matches vLLM since training GPUs are free. "
             "sleep_wake: (EXPERIMENTAL) Native SGLang weight offload via "
             "  /release_memory_occupation + /resume_memory_occupation. "
             "  Requires SGLang >=0.6. Falls back to freeze if unsupported. "
             "hot_reload: Keep server active. Cache preserved but ~2x slower on 2-GPU. "
             "restart: Kill server. Cache LOST. Fallback only. "
             "sleep: (DEPRECATED) Old custom launcher. Use freeze instead.",
    )
    parser.add_argument(
        "--vllm-cache-method",
        type=str,
        choices=["http", "sleep_wake", "none"],
        default="http",
        help="Cache preservation method for vLLM (default: http). "
             "http: Use /v1/load_lora_adapter API (requires GPU isolation). "
             "sleep_wake: Use do_sleep/do_wake_up with GPU isolation (keeps vLLM on dedicated GPUs). "
             "none: Always restart server (no cache preservation).",
    )
    parser.add_argument(
        "--vllm-gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for vLLM (e.g., '0' or '0,1'). "
             "Only used with --preserve-cache and --vllm-cache-method=http. "
             "Remaining GPUs are used for Megatron.",
    )
    parser.add_argument(
        "--training-gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for Megatron training (e.g., '1' or '0,1'). "
             "Use with sleep_wake to limit training to specific GPUs for fair comparison. "
             "If not set, sleep_wake uses all GPUs, HTTP uses GPUs not used by vLLM.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        default=True,
        help="Run vLLM service in-process (default: True). "
             "This avoids multiprocessing pickle errors.",
    )
    parser.add_argument(
        "--no-in-process",
        action="store_true",
        help="Run vLLM service in a child process (may cause pickle errors).",
    )
    
    args = parser.parse_args()
    
    # Parse sglang_gpu_ids
    sglang_gpu_ids = None
    if args.sglang_gpu_ids:
        sglang_gpu_ids = [int(x.strip()) for x in args.sglang_gpu_ids.split(",")]
    
    # Parse vllm_gpu_ids
    vllm_gpu_ids = None
    if args.vllm_gpu_ids:
        vllm_gpu_ids = [int(x.strip()) for x in args.vllm_gpu_ids.split(",")]
    
    # Parse training_gpu_ids
    training_gpu_ids = None
    if args.training_gpu_ids:
        training_gpu_ids = [int(x.strip()) for x in args.training_gpu_ids.split(",")]
    
    print("=" * 60)
    print("SGLang vs vLLM Megatron Benchmark")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"Scenario: {args.scenario}")
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print(f"Rollouts: {args.num_rollouts}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    if args.vllm_tp_size:
        print(f"vLLM TP size: {args.vllm_tp_size}")
    if args.enable_training:
        print(f"Training: ENABLED (lr={args.learning_rate})")
    else:
        print("Training: DISABLED (use --enable-training to enable)")
    if args.preserve_cache:
        sglang_gpus = sglang_gpu_ids or [0]
        vllm_gpus = vllm_gpu_ids or sglang_gpu_ids or [0]
        print(f"SGLang Cache Method: {args.sglang_cache_method} (GPUs: {sglang_gpus})")
        print(f"vLLM Cache Method: {args.vllm_cache_method} (GPUs: {vllm_gpus})")
    else:
        print(f"Cache Preservation: DISABLED")
        print(f"SGLang Cache Method: {args.sglang_cache_method}")
        print(f"vLLM Cache Method: {args.vllm_cache_method}")
    print("=" * 60)
    
    backends = ["sglang", "vllm"] if args.backend == "both" else [args.backend]
    scenarios = ["A", "B", "C", "D", "E"] if args.scenario == "all" else [args.scenario]
    
    all_results = []
    
    for backend in backends:
        print(f"\n{'=' * 60}")
        print(f"BENCHMARKING: {backend.upper()}")
        print(f"{'=' * 60}")
        
        # Determine in_process mode (default True, --no-in-process disables)
        in_process = not args.no_in_process
        
        runner = BenchmarkRunner(
            backend_type=backend,
            model=args.model,
            work_dir=f"{args.work_dir}/{backend}",
            num_samples=args.num_samples,
            num_rollouts=args.num_rollouts,
            max_tokens=args.max_tokens,
            num_iterations=args.num_iterations,
            vllm_tp_size=args.vllm_tp_size,
            enable_training=args.enable_training,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            temperature=args.temperature,
            preserve_cache=args.preserve_cache,
            sglang_gpu_ids=sglang_gpu_ids,
            sglang_cache_method=args.sglang_cache_method,
            vllm_cache_method=args.vllm_cache_method,
            vllm_gpu_ids=vllm_gpu_ids,
            training_gpu_ids=training_gpu_ids,
            in_process=in_process,
        )
        
        for scenario in scenarios:
            try:
                if scenario == "A":
                    await runner.run_scenario_a()
                elif scenario == "B":
                    await runner.run_scenario_b()
                elif scenario == "C":
                    await runner.run_scenario_c()
                elif scenario == "D":
                    await runner.run_scenario_d()
                elif scenario == "E":
                    await runner.run_scenario_e()
            except Exception as e:
                print(f"Error in scenario {scenario}: {e}")
                import traceback
                traceback.print_exc()
        
        # Clean up between backends
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        all_results.extend(runner.results)
        runner.save_results(f"{args.output}_{backend}")
    
    # Save combined results if both backends
    if args.backend == "both":
        combined_path = args.output + "_comparison"
        with open(combined_path + ".json", "w") as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        print(f"\nSaved comparison results to: {combined_path}.json")
        
        # Print comparison table
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Scenario':<20} {'Metric':<25} {'SGLang':<15} {'vLLM':<15} {'Winner':<10}")
        print("-" * 80)
        
        for scenario in scenarios:
            sglang_results = [r for r in all_results if r.backend == "sglang" and scenario in r.scenario]
            vllm_results = [r for r in all_results if r.backend == "vllm" and scenario in r.scenario]
            
            if sglang_results and vllm_results:
                sr = sglang_results[0]
                vr = vllm_results[0]
                
                # Compare TTFT
                s_ttft = sr.inference_metrics.time_to_first_token_ms
                v_ttft = vr.inference_metrics.time_to_first_token_ms
                winner = "SGLang" if s_ttft < v_ttft else "vLLM"
                print(f"{scenario:<20} {'TTFT (ms)':<25} {s_ttft:<15.2f} {v_ttft:<15.2f} {winner:<10}")
                
                # Compare throughput
                s_tps = sr.inference_metrics.tokens_per_second
                v_tps = vr.inference_metrics.tokens_per_second
                winner = "SGLang" if s_tps > v_tps else "vLLM"
                print(f"{'':<20} {'Tokens/sec':<25} {s_tps:<15.2f} {v_tps:<15.2f} {winner:<10}")
    
    print("\n✓ Benchmark complete!")
    
    # Final cleanup to suppress async warnings
    await asyncio.sleep(0.5)
    
    # Kill any remaining sglang processes
    import subprocess
    try:
        subprocess.run(["pkill", "-f", "sglang.launch_server"], 
                      capture_output=True, timeout=5)
    except Exception:
        pass


if __name__ == "__main__":
    # Suppress asyncio cleanup warnings
    import warnings
    import logging
    import sys
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*CancelledError.*")
    
    # Suppress asyncio error logs
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    
    # Redirect stderr temporarily for cleaner output
    class SuppressAsyncErrors:
        def __enter__(self):
            self._stderr = sys.stderr
            return self
        def __exit__(self, *args):
            pass
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
    except Exception as e:
        import traceback
        print(f"\nBenchmark error: {e}")
        traceback.print_exc()
    finally:
        # Force kill any remaining processes
        import subprocess
        subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], 
                      capture_output=True, timeout=5)
        subprocess.run(["pkill", "-9", "-f", "sglang"], 
                      capture_output=True, timeout=5)
