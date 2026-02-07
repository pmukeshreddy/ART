"""SGLang + Megatron service for inference and distributed training.

This service combines:
- SGLang for inference (RadixAttention prefix caching, fast scheduling)
- Megatron-Core for distributed training (expert parallelism, tensor parallelism)

Architecture:
    GPU 0: SGLang inference server
    GPU 1+: Megatron distributed training
    Weight sync: Hot-reload LoRA via SGLang API (no restart needed)

This is similar to MegatronService but replaces vLLM with SGLang for inference.
"""

import asyncio
from dataclasses import asdict, dataclass
import datetime
from functools import cached_property
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import AsyncIterator, Literal

import aiohttp
from peft.tuners.lora.config import LoraConfig
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from .. import dev, types
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.pack import DiskPackedTensors
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir


def gc_and_empty_cuda_cache(n: int = 3) -> None:
    """Run garbage collection and empty CUDA cache."""
    for _ in range(n):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# GPU health detection — cached per process lifetime
# ---------------------------------------------------------------------------
_healthy_gpu_cache: list[int] | None = None


def _get_healthy_gpu_ids() -> list[int]:
    """Return GPU indices that are actually usable, excluding dead/stuck ones.

    A GPU is considered dead if nvidia-smi reports high utilisation (>90 %)
    but zero memory usage and no running processes — a typical symptom of a
    hardware fault on cloud H100 instances.

    The result is cached so the (cheap) nvidia-smi call only happens once.
    """
    global _healthy_gpu_cache
    if _healthy_gpu_cache is not None:
        return list(_healthy_gpu_cache)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            _healthy_gpu_cache = list(range(torch.cuda.device_count()))
            return list(_healthy_gpu_cache)

        healthy: list[int] = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            idx, mem_used, util = int(parts[0]), int(parts[1]), int(parts[2])
            if mem_used == 0 and util > 90:
                print(
                    f"  WARNING: GPU {idx} appears dead/stuck "
                    f"(util={util}%, mem=0 MiB) — auto-excluding from training"
                )
                continue
            healthy.append(idx)

        _healthy_gpu_cache = healthy if healthy else list(range(torch.cuda.device_count()))
        return list(_healthy_gpu_cache)
    except Exception:
        _healthy_gpu_cache = list(range(torch.cuda.device_count()))
        return list(_healthy_gpu_cache)


class MegatronTrainingJob(BaseModel):
    """Job format for communication with train.py subprocess."""

    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    experimental_config: dev.TrainConfig
    log_file_path: str = "/tmp/megatron_training_log.jsonl"


SGLangCacheMethod = Literal["sleep_wake", "sleep", "freeze", "hot_reload", "restart"]


@dataclass
class SGLangConfig:
    """Configuration for SGLang server."""
    
    # Set to 0.85 to maximize KV cache pool while leaving 5-8GB for activations
    # and CUDA graph buffers. This improves prefix cache hit rate significantly.
    mem_fraction_static: float = 0.85
    disable_radix_cache: bool = False
    max_loras_per_batch: int = 4
    context_length: int | None = None
    server_timeout: float = 300.0  # 5 min default for large models
    log_level: str = "info"  # More verbose for debugging
    tensor_parallel_size: int | None = None  # None = auto-detect based on GPU count
    sglang_python_path: str | None = None  # Path to SGLang server venv python
    
    # GPU isolation settings for keeping server running during training
    # When enabled, SGLang runs on dedicated GPUs and Megatron uses the rest
    preserve_cache_during_training: bool = True  # Keep server running if possible
    sglang_gpu_ids: list[int] | None = None  # Explicit GPU IDs for SGLang (e.g., [0])
    training_gpu_ids: list[int] | None = None  # Explicit GPU IDs for Megatron training (e.g., [1])
    
    # Cache method for SGLang during training:
    #   "sleep_wake" (RECOMMENDED): Uses SGLang's native /release_memory_occupation
    #                 to offload model weights + CUDA graphs to CPU before training.
    #                 KV cache stays on GPU → prefix cache PRESERVED.
    #                 After training: /update_weights_from_disk reloads ALL weights
    #                 fresh from disk (NOT resume_memory_occupation which is broken
    #                 for MoE models — SGLang issue #6367). Then /load_lora_adapter
    #                 with the new LoRA checkpoint.
    #                 Requires: --enable-memory-saver on server.
    #                 Frees ~48GB (weights + CUDA graphs), keeps ~17GB (KV cache) on GPU.
    #                 Training speed should match vLLM sleep_wake.
    #   "freeze":     SIGSTOP the SGLang process group during training, SIGCONT after.
    #                 Suspends ALL threads → zero CPU/PCIe contention.
    #                 GPU memory stays allocated (model weights + KV cache on GPU 0).
    #                 After SIGCONT: server resumes instantly, cache is 100% preserved.
    #                 Training may be slower due to 65GB frozen allocation on GPU 0.
    #   "hot_reload": Keep server fully active, hot-reload LoRA via HTTP API.
    #                 Preserves RadixAttention cache but active process causes
    #                 CPU/PCIe contention with Megatron (~2x slower training on 2-GPU).
    #                 Best for: 4+ GPU setups where contention is minimal.
    #   "restart":    Stop server before training, restart after with new LoRA.
    #                 Cache is LOST. Training runs at full speed but server restart
    #                 adds ~30-60s overhead per iteration.
    #                 Use only as fallback when freeze/hot_reload don't work.
    #   "sleep":      (DEPRECATED) Old custom launcher approach. Use "sleep_wake" instead.
    cache_method: SGLangCacheMethod = "freeze"
    
    def get_tensor_parallel_size(self) -> int:
        """Get tensor parallel size, auto-detecting if not set."""
        if self.tensor_parallel_size is not None:
            return self.tensor_parallel_size
        # If explicit GPU IDs are set, use that count
        if self.sglang_gpu_ids is not None:
            return len(self.sglang_gpu_ids)
        # Auto-detect: use all available GPUs for inference
        gpu_count = torch.cuda.device_count()
        return gpu_count if gpu_count > 0 else 1
    
    def get_sglang_gpu_ids(self) -> list[int]:
        """Get GPU IDs for SGLang server."""
        if self.sglang_gpu_ids is not None:
            return self.sglang_gpu_ids
        # Default: use GPUs 0 to (tp_size - 1)
        tp_size = self.get_tensor_parallel_size()
        return list(range(tp_size))
    
    def get_megatron_gpu_ids(self) -> list[int]:
        """Get GPU IDs for Megatron training.
        
        If training_gpu_ids is set, use those explicitly.
        Otherwise, auto-detect healthy GPUs not used by SGLang.
        Dead/stuck GPUs (high util, zero memory) are automatically excluded.
        """
        if self.training_gpu_ids is not None:
            # Warn if user explicitly picked a dead GPU
            healthy = set(_get_healthy_gpu_ids())
            for gpu in self.training_gpu_ids:
                if gpu not in healthy:
                    print(
                        f"  WARNING: Specified training GPU {gpu} may be "
                        f"dead/stuck — consider removing it from training_gpu_ids"
                    )
            return self.training_gpu_ids
        # Auto-detect: healthy GPUs minus SGLang GPUs
        healthy_gpus = _get_healthy_gpu_ids()
        sglang_gpus = set(self.get_sglang_gpu_ids())
        return [i for i in healthy_gpus if i not in sglang_gpus]
    
    def can_preserve_cache(self) -> bool:
        """Check if we can keep SGLang process alive during training.
        
        Returns True for "sleep_wake", "sleep", "freeze", and "hot_reload" modes (all preserve cache).
        Returns False for "restart" (server is killed, cache lost).
        
        Requires GPUs available for Megatron that aren't used by SGLang.
        """
        if not self.preserve_cache_during_training:
            return False
        if self.cache_method == "restart":
            return False
        megatron_gpus = self.get_megatron_gpu_ids()
        return len(megatron_gpus) > 0
    
    def get_server_python(self) -> str:
        """Get Python executable for SGLang server subprocess."""
        import sys
        
        if self.sglang_python_path:
            path = os.path.abspath(self.sglang_python_path)
            if os.path.exists(path):
                return path
        
        # Auto-detect .venv-sglang-server
        search_paths = [
            ".venv-sglang-server/bin/python",
            "../.venv-sglang-server/bin/python",
        ]
        
        for rel_path in search_paths:
            abs_path = os.path.abspath(rel_path)
            if os.path.exists(abs_path):
                print(f"Auto-detected SGLang server venv: {abs_path}")
                return abs_path
        
        return sys.executable


@dataclass
class SGLangMegatronService:
    """Service using SGLang for inference and Megatron for distributed training.
    
    This is the SGLang equivalent of MegatronService which uses vLLM.
    
    Key differences from MegatronService:
    - Uses SGLang subprocess for inference instead of vLLM AsyncLLM
    - Hot-reloads LoRA via SGLang HTTP API instead of vLLM LoRARequest
    - RadixAttention cache preserved across training steps
    """
    
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    sglang_config: SGLangConfig | None = None
    
    _is_sleeping: bool = False
    _latest_step: int = 0
    _lora_id_counter: int = 1
    _server_process: subprocess.Popen | None = None
    _server_port: int = 8000
    _server_host: str = "127.0.0.1"
    _control_port: int | None = None  # Port for sleep/wake control (sglang_sleep_server)
    _megatron_process: asyncio.subprocess.Process | None = None
    _optimizer_state_path: str | None = None

    def __post_init__(self):
        if self.sglang_config is None:
            self.sglang_config = SGLangConfig()

    def _next_lora_id(self) -> int:
        self._lora_id_counter += 1
        return self._lora_id_counter

    def _get_fixed_lora_name(self) -> str:
        """Get fixed LoRA adapter name for RadixCache consistency.
        
        SGLang's RadixCache keeps separate tree branches per adapter name.
        Using a fixed name ensures cache hits across epochs:
        - Same adapter name + same prefix = cache HIT (~80% hit rate)
        - Different adapter name + same prefix = cache MISS (new branch)
        
        As of PR #7216 (August 2025), SGLang LoRA + RadixCache is supported.
        """
        return f"{self.model_name}@latest"

    def _get_optimizer_state_path(self) -> str:
        if self._optimizer_state_path is not None:
            return self._optimizer_state_path
        self._optimizer_state_path = os.path.join(self.output_dir, "optimizer_states")
        os.makedirs(self._optimizer_state_path, exist_ok=True)
        return self._optimizer_state_path

    def _default_lora_adapter_config(self) -> LoraConfig:
        """Default LoRA config matching megatron/train.py settings."""
        return LoraConfig(
            r=1,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        )

    def _adapter_has_weights(self, lora_path: str) -> bool:
        """Check if LoRA adapter has non-zero weights."""
        adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            return False
        try:
            with safe_open(adapter_path, framework="pt") as adapter_file:
                for key in adapter_file.keys():
                    tensor = adapter_file.get_tensor(key)
                    if torch.any(tensor != 0):
                        return True
        except Exception:
            return False
        return False

    def _sanitize_lora_config(self, lora_path: str) -> None:
        """Remove unsupported fields from adapter_config.json for vLLM 0.15.x compatibility."""
        config_path = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_path):
            return
        
        unsupported_fields = {'long_lora_max_len'}
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            fields_to_remove = unsupported_fields & set(config.keys())
            if not fields_to_remove:
                return
            
            for field in fields_to_remove:
                del config[field]
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except (json.JSONDecodeError, IOError):
            pass

    def _create_identity_lora(self, lora_path: str) -> None:
        """Create an identity (zero) LoRA for initial inference.
        
        For MoE models, we skip PEFT-based LoRA creation since the shapes
        don't match Megatron's expectations. Instead, we just create an
        empty checkpoint directory and let Megatron initialize LoRA.
        """
        os.makedirs(lora_path, exist_ok=True)
        # Just create the adapter config, no weights
        # SGLang can start without LoRA weights for initial inference
        # Megatron will create proper LoRA weights during first training
        self._ensure_lora_adapter_config(lora_path)
        print(f"Created empty LoRA checkpoint at {lora_path} (Megatron will initialize)")

    def _ensure_identity_lora(self, lora_path: str) -> None:
        """Ensure LoRA checkpoint directory exists for server startup."""
        if os.path.exists(lora_path):
            return
        self._create_identity_lora(lora_path)

    def _ensure_lora_adapter_config(
        self, lora_path: str, *, source_path: str | None = None
    ) -> None:
        """Ensure adapter_config.json exists."""
        config_path = os.path.join(lora_path, "adapter_config.json")
        if os.path.exists(config_path):
            # Sanitize existing config for vLLM 0.15.x compatibility
            self._sanitize_lora_config(lora_path)
            return
        os.makedirs(lora_path, exist_ok=True)
        if source_path is not None:
            source_config = os.path.join(source_path, "adapter_config.json")
            if os.path.exists(source_config):
                shutil.copy(source_config, config_path)
                # Sanitize copied config
                self._sanitize_lora_config(lora_path)
                return
        # Convert to dict and handle sets (target_modules is a set in LoraConfig)
        config_dict = asdict(self._default_lora_adapter_config())
        for key, value in config_dict.items():
            if isinstance(value, set):
                config_dict[key] = list(value)
        # Remove long_lora_max_len if present (vLLM 0.15.x doesn't support it)
        config_dict.pop('long_lora_max_len', None)
        with open(config_path, "w") as f:
            json.dump(config_dict, f)

    # ========================================================================
    # SGLang Server Management (replaces vLLM in MegatronService)
    # ========================================================================

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        """Start SGLang OpenAI-compatible server.
        
        This replaces vLLM's openai_server_task in MegatronService.
        """
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            self._latest_step = 0
        else:
            self._latest_step = get_step_from_dir(self.output_dir)
        
        self._ensure_identity_lora(lora_path)
        self._ensure_lora_adapter_config(lora_path)

        # Get server configuration
        server_config = config or {}
        server_args = server_config.get("server_args", {})
        
        self._server_host = server_args.get("host", "127.0.0.1")
        self._server_port = server_args.get("port", 8000)
        
        # Create logs directory
        log_dir = f"{self.output_dir}/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Only include LoRA path if adapter has actual weights
        # For initial run (step 0), there are no weights yet - SGLang starts without LoRA
        lora_path_for_server = lora_path if self._adapter_has_weights(lora_path) else None
        if lora_path_for_server:
            print(f"Starting SGLang with LoRA from {lora_path_for_server}")
        else:
            print("Starting SGLang without LoRA (initial run, Megatron will create weights)")
        
        # Start SGLang server subprocess
        await self._start_sglang_server(lora_path_for_server)
        
        return self._server_host, self._server_port

    async def _start_sglang_server(self, lora_path: str | None = None) -> None:
        """Start SGLang server as subprocess."""
        # Kill any existing SGLang processes
        subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "sglang_sleep_server"], capture_output=True)
        
        # Get Python executable for SGLang server
        server_python = self.sglang_config.get_server_python()
        
        # Get tensor parallel size (auto-detect for large models)
        tp_size = self.sglang_config.get_tensor_parallel_size()
        
        # Get GPU IDs for SGLang
        sglang_gpu_ids = self.sglang_config.get_sglang_gpu_ids()
        
        # Use custom sleep launcher if cache_method is "sleep" (deprecated)
        if self.sglang_config.cache_method == "sleep":
            sleep_server_path = str(Path(__file__).parent / "sglang_sleep_server.py")
            self._control_port = self._server_port + 100
            cmd = [
                server_python, sleep_server_path,
                "--art-control-port", str(self._control_port),
                "--model-path", self.base_model,
                "--host", self._server_host,
                "--port", str(self._server_port),
                "--mem-fraction-static", str(self.sglang_config.mem_fraction_static),
                "--log-level", self.sglang_config.log_level,
                "--enable-lora",
            ]
            print(f"  Using ART sleep launcher (control port: {self._control_port})")
        else:
            self._control_port = None
        cmd = [
            server_python, "-m", "sglang.launch_server",
            "--model-path", self.base_model,
            "--host", self._server_host,
            "--port", str(self._server_port),
            "--mem-fraction-static", str(self.sglang_config.mem_fraction_static),
            "--log-level", self.sglang_config.log_level,
            "--enable-lora",
        ]
        
        # For sleep_wake mode, try to enable native memory saver so we can
        # call /release_memory_occupation and /resume_memory_occupation.
        # NOTE: These flags may not exist in all SGLang versions (e.g. 0.5.8).
        # If the server fails to start, fall back to freeze mode.
        if self.sglang_config.cache_method == "sleep_wake":
            # Validate --enable-memory-saver exists by checking sglang help output
            # NOTE: We do NOT need --enable-weights-cpu-backup because we use
            # /update_weights_from_disk to reload weights after training (not
            # resume_memory_occupation which is broken for MoE models).
            try:
                help_output = subprocess.run(
                    [server_python, "-m", "sglang.launch_server", "--help"],
                    capture_output=True, text=True, timeout=15
                ).stdout
                has_memory_saver = "--enable-memory-saver" in help_output
            except Exception:
                has_memory_saver = False
            
            if has_memory_saver:
                cmd.extend(["--enable-memory-saver"])
                print(f"  Native sleep/wake enabled (memory-saver={has_memory_saver})")
                print(f"  Post-training: update_weights_from_disk (MoE-safe, no CPU backup needed)")
            else:
                print(f"  WARNING: SGLang version does not support --enable-memory-saver.")
                print(f"  Falling back to 'freeze' mode (cache still preserved).")
                self.sglang_config.cache_method = "freeze"
        
        # Add tensor parallelism
        if tp_size > 1:
            cmd.extend(["--tp-size", str(tp_size)])
            print(f"  Using tensor parallelism: TP={tp_size}")
        
        # Performance optimizations for SGLang
        # Tested various flags - minimal config performs best for RL workloads
        # Removed: --schedule-conservativeness, --chunked-prefill-size
        # Note: --enable-memory-saver is added above only for sleep_wake mode
        
        # 1. LPM (Longest Prefix Match) scheduler - reorders requests for prefix sharing
        cmd.extend(["--schedule-policy", "lpm"])
        
        # 2. CUDA graph and attention backend
        cmd.extend(["--cuda-graph-max-bs", "128"])
        cmd.extend(["--attention-backend", "flashinfer"])
        
        # Add context length if specified
        if self.sglang_config.context_length:
            cmd.extend(["--context-length", str(self.sglang_config.context_length)])
        
        # Add LoRA configuration
        if lora_path and os.path.exists(lora_path) and self._adapter_has_weights(lora_path):
            cmd.extend(["--lora-paths", lora_path])
            cmd.extend(["--max-loras-per-batch", str(self.sglang_config.max_loras_per_batch)])
        else:
            # No LoRA adapter yet - provide LoRA config for initialization
            # SGLang requires these when --enable-lora is set without --lora-paths
            lora_config = self._default_lora_adapter_config()
            cmd.extend(["--max-lora-rank", str(lora_config.r)])
            # SGLang expects space-separated modules, not comma-separated
            cmd.extend(["--lora-target-modules"] + list(lora_config.target_modules))
        
        # Disable radix cache only if explicitly requested
        if self.sglang_config.disable_radix_cache:
            cmd.append("--disable-radix-cache")
        
        print(f"Starting SGLang server: {' '.join(cmd)}")
        
        # Set up environment with GPU isolation
        env = os.environ.copy()
        
        # =====================================================================
        # FIX: Ensure SGLang subprocess can find CUDA + system libraries
        # 
        # Problem: On RunPod/containers, system CUDA may be 13.0 but PyTorch
        # in the SGLang venv was built for CUDA 12.x. The CUDA 12 runtime is
        # bundled inside pip packages (nvidia-cuda-runtime-cu12 etc.) inside
        # the venv. SGLang's child processes need these on LD_LIBRARY_PATH.
        #
        # Also: flashinfer JIT needs 'ninja' from the venv's bin/ on PATH.
        # =====================================================================
        
        # 1. Find the SGLang venv's site-packages WITHOUT importing torch
        #    (import torch itself fails without LD_LIBRARY_PATH set first!)
        sglang_venv_site = None
        if server_python:
            try:
                result = subprocess.run(
                    [server_python, "-c", 
                     "import site; print(site.getsitepackages()[0])"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    sglang_venv_site = result.stdout.strip()
            except Exception:
                pass
            
            # Fallback: derive from server_python path
            # e.g. /root/ART/.venv-sglang-server/bin/python
            #   → /root/ART/.venv-sglang-server/lib/python3.11/site-packages
            if not sglang_venv_site or not os.path.isdir(sglang_venv_site):
                venv_root = os.path.dirname(os.path.dirname(
                    os.path.realpath(server_python)
                ))
                # Find the python version dir
                lib_dir = os.path.join(venv_root, "lib")
                if os.path.isdir(lib_dir):
                    for d in os.listdir(lib_dir):
                        if d.startswith("python"):
                            candidate = os.path.join(lib_dir, d, "site-packages")
                            if os.path.isdir(candidate):
                                sglang_venv_site = candidate
                                break
        
        venv_nvidia_lib_paths = []
        if sglang_venv_site and os.path.isdir(sglang_venv_site):
            nvidia_dir = os.path.join(sglang_venv_site, "nvidia")
            if os.path.isdir(nvidia_dir):
                # Scan all nvidia packages for lib directories
                for pkg in sorted(os.listdir(nvidia_dir)):
                    lib_dir = os.path.join(nvidia_dir, pkg, "lib")
                    if os.path.isdir(lib_dir):
                        venv_nvidia_lib_paths.append(lib_dir)
            # Also add torch/lib itself (has libc10_cuda.so etc.)
            torch_lib = os.path.join(sglang_venv_site, "torch", "lib")
            if os.path.isdir(torch_lib):
                venv_nvidia_lib_paths.append(torch_lib)
        
        # 2. System CUDA paths (fallback)
        system_lib_paths = [
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64",
        ]
        
        # Venv nvidia paths go FIRST (correct CUDA 12.x version)
        all_lib_paths = venv_nvidia_lib_paths + system_lib_paths
        existing_ld_path = env.get("LD_LIBRARY_PATH", "")
        new_ld_path = ":".join(all_lib_paths)
        if existing_ld_path:
            new_ld_path = f"{new_ld_path}:{existing_ld_path}"
        env["LD_LIBRARY_PATH"] = new_ld_path
        
        if venv_nvidia_lib_paths:
            print(f"  SGLang CUDA fix: added {len(venv_nvidia_lib_paths)} nvidia lib paths from venv")
        else:
            print(f"  WARNING: Could not find nvidia lib paths in SGLang venv (site={sglang_venv_site})")
        
        # 3. Add venv bin/ to PATH so SGLang child processes can find 'ninja'
        #    (needed by flashinfer JIT compilation)
        if server_python:
            venv_bin = os.path.dirname(os.path.realpath(server_python))
            existing_path = env.get("PATH", "")
            env["PATH"] = f"{venv_bin}:{existing_path}"
        
        # 4. Force preload libnuma for sgl_kernel H100 support
        libnuma_paths = [
            "/usr/lib/x86_64-linux-gnu/libnuma.so.1",
            "/lib/x86_64-linux-gnu/libnuma.so.1",
            "/usr/lib64/libnuma.so.1",
        ]
        libnuma_found = None
        for path in libnuma_paths:
            if os.path.exists(path):
                libnuma_found = path
                break
        
        if libnuma_found:
            existing_preload = env.get("LD_PRELOAD", "")
            env["LD_PRELOAD"] = f"{libnuma_found}:{existing_preload}" if existing_preload else libnuma_found
        else:
            print("  WARNING: libnuma.so.1 not found. Install with: apt-get install libnuma1")
        
        # 5. Set CUDA environment
        env["CUDA_HOME"] = "/usr/local/cuda"
        # =====================================================================
        
        if self.sglang_config.can_preserve_cache():
            # Use only the designated GPUs for SGLang
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, sglang_gpu_ids))
            print(f"  SGLang GPU isolation: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        
        # Start server
        log_file = open(f"{self.output_dir}/logs/sglang.log", "a")
        self._server_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=env,
        )
        
        # Wait for server to be ready
        await self._wait_for_server()

    async def _wait_for_server(self) -> None:
        """Wait for SGLang server to be ready."""
        timeout = self.sglang_config.server_timeout
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if self._server_process and self._server_process.poll() is not None:
                raise RuntimeError(
                    f"SGLang server process died with code {self._server_process.returncode}. "
                    f"Check logs at {self.output_dir}/logs/sglang.log"
                )
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{self._server_host}:{self._server_port}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            print(f"SGLang server ready at http://{self._server_host}:{self._server_port}")
                            return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        
        raise TimeoutError(
            f"SGLang server did not start within {timeout} seconds. "
            f"Check logs at {self.output_dir}/logs/sglang.log"
        )

    async def _stop_sglang_server(self) -> None:
        """Stop SGLang server subprocess."""
        if self._server_process is None:
            return
        
        try:
            try:
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                self._server_process.kill()
            
            for _ in range(10):
                if self._server_process.poll() is not None:
                    break
                await asyncio.sleep(0.1)
        except Exception:
            pass
        finally:
            self._server_process = None
        
        gc_and_empty_cuda_cache()

    def _freeze_server(self) -> bool:
        """Freeze (SIGSTOP) the SGLang server using its process group.
        
        Simple and safe: we started SGLang with preexec_fn=os.setsid, so it has
        its own process group. os.killpg() stops the entire group — parent and
        all children that inherited the group. Cannot accidentally stop ourselves.
        
        Returns True if successful, False if process not running.
        """
        if self._server_process is None or self._server_process.poll() is not None:
            print(f"[SGLang] DEBUG _freeze: server not running "
                  f"(is_none={self._server_process is None}, "
                  f"poll={self._server_process.poll() if self._server_process else 'N/A'})", flush=True)
            return False
        
        try:
            pid = self._server_process.pid
            pgid = os.getpgid(pid)
            print(f"[SGLang] DEBUG _freeze: pid={pid}, pgid={pgid}, my_pid={os.getpid()}", flush=True)
            os.killpg(pgid, signal.SIGSTOP)
            print(f"[SGLang] Server frozen (SIGSTOP pgid={pgid})", flush=True)
            print(f"  GPU memory stays allocated — model weights + KV cache intact", flush=True)
            return True
        except (ProcessLookupError, OSError) as e:
            print(f"[SGLang] ⚠️ Failed to freeze server: {e}", flush=True)
            return False

    async def _thaw_server(self, timeout: float = 30.0) -> bool:
        """Thaw (SIGCONT) the SGLang server using its process group.
        
        Mirrors _freeze_server: SIGCONT the same process group.
        
        Returns True if server is healthy after thaw, False otherwise.
        """
        if self._server_process is None or self._server_process.poll() is not None:
            print(f"[SGLang] DEBUG _thaw: server not running", flush=True)
            return False
        
        try:
            pid = self._server_process.pid
            pgid = os.getpgid(pid)
            print(f"[SGLang] DEBUG _thaw: SIGCONT pgid={pgid}", flush=True)
            os.killpg(pgid, signal.SIGCONT)
            print(f"[SGLang] Server thawed (SIGCONT pgid={pgid})", flush=True)
        except (ProcessLookupError, OSError) as e:
            print(f"[SGLang] ⚠️ Failed to thaw server: {e}", flush=True)
            return False
        
        # Wait for server to respond to health checks after thaw
        start = asyncio.get_event_loop().time()
        check_count = 0
        while asyncio.get_event_loop().time() - start < timeout:
            check_count += 1
            healthy = await self._is_server_healthy()
            if check_count <= 5 or check_count % 10 == 0:
                print(f"[SGLang] DEBUG _thaw: health check #{check_count} = {healthy}", flush=True)
            if healthy:
                elapsed = asyncio.get_event_loop().time() - start
                print(f"[SGLang] Server responsive after thaw ({elapsed:.1f}s)", flush=True)
                return True
            await asyncio.sleep(0.5)
        
        print(f"[SGLang] ⚠️ Server not responding after thaw (waited {timeout}s, {check_count} checks)", flush=True)
        return False

    async def _sleep_server(self) -> dict | None:
        """Call /art/sleep on the custom launcher's control port.
        
        Offloads model weights from GPU to CPU (~25GB freed).
        Returns the response dict if successful, None if unavailable.
        """
        if self._control_port is None:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{self._control_port}/art/sleep",
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "sleeping":
                            return data
                        elif data.get("status") == "error":
                            print(f"[SGLang] Sleep endpoint error: {data.get('error')}")
                            return None
                    return None
        except Exception as e:
            print(f"[SGLang] Sleep endpoint unavailable: {e}")
            return None

    async def _wake_server(self) -> dict | None:
        """Call /art/wake on the custom launcher's control port.
        
        Reloads model weights from CPU back to GPU.
        Returns the response dict if successful, None if unavailable.
        """
        if self._control_port is None:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{self._control_port}/art/wake",
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "awake":
                            return data
                    return None
        except Exception as e:
            print(f"[SGLang] Wake endpoint unavailable: {e}")
            return None

    async def _native_release_memory(self, tags: list[str] | None = None) -> bool:
        """Call SGLang's native /release_memory_occupation endpoint.
        
        Offloads specified memory types from GPU to CPU.
        
        Args:
            tags: List of memory types to release. Options:
                  "weights" — model weights (~48GB for 30B model)
                  "kv_cache" — KV cache (WARNING: flushes radix cache!)
                  "cuda_graph" — CUDA graph buffers
                  None — release ALL types
                  
        For cache preservation, use tags=["weights", "cuda_graph"] to keep KV cache.
        
        Returns True if successful.
        """
        if tags is None:
            tags = ["weights", "cuda_graph"]  # Safe default: keep KV cache
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{self._server_host}:{self._server_port}/release_memory_occupation",
                    json={"tags": tags},
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 200:
                        print(f"[SGLang] ✓ Memory released: tags={tags}")
                        return True
                    else:
                        error_text = await resp.text()
                        print(f"[SGLang] ⚠️ release_memory_occupation failed ({resp.status}): {error_text}")
                        return False
        except Exception as e:
            print(f"[SGLang] ⚠️ release_memory_occupation error: {e}")
            return False

    async def _update_weights_from_disk(self, model_path: str | None = None) -> bool:
        """Reload ALL model weights from disk via /update_weights_from_disk.
        
        This is the correct post-training resume path for MoE models.
        
        Why NOT resume_memory_occupation:
            resume_memory_occupation restores weights from CPU backup, but this is
            broken for MoE models (SGLang issue #6367) — expert layers don't get
            properly placed on GPU, causing garbage output.
        
        Why update_weights_from_disk:
            - Reloads ALL weights fresh from disk, properly placing MoE expert layers
            - Bypasses the broken CPU backup restore entirely
            - KV cache stays on GPU (prefix cache preserved)
            - Slightly slower than CPU→GPU copy (disk I/O) but actually works for MoE
            - Still way faster than full server restart (no process spawn, no CUDA
              graph recapture, no KV cache rebuild)
        
        This pattern is used by veRL and Slime for the same reason.
        
        Args:
            model_path: Path to load weights from. Defaults to self.base_model.
        
        Returns True if successful.
        """
        if model_path is None:
            model_path = self.base_model
        
        import time as _time
        _t0 = _time.perf_counter()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{self._server_host}:{self._server_port}/update_weights_from_disk",
                    json={
                        "model_path": model_path,
                        "load_format": "auto",
                    },
                    timeout=aiohttp.ClientTimeout(total=300)  # Large models take time
                ) as resp:
                    elapsed = _time.perf_counter() - _t0
                    if resp.status == 200:
                        print(f"[SGLang] ✓ Weights reloaded from disk in {elapsed:.1f}s (model={model_path})")
                        return True
                    else:
                        error_text = await resp.text()
                        print(f"[SGLang] ⚠️ update_weights_from_disk failed ({resp.status}): {error_text}")
                        return False
        except Exception as e:
            elapsed = _time.perf_counter() - _t0
            print(f"[SGLang] ⚠️ update_weights_from_disk error after {elapsed:.1f}s: {e}")
            return False

    async def _hot_reload_lora(self, checkpoint_dir: str, step: int) -> None:
        """Hot-reload LoRA weights without restarting server.
        
        Key optimization for RadixCache preservation:
        - Uses FIXED adapter name (_get_fixed_lora_name) across all epochs
        - SGLang keeps same radix tree branch for same adapter name
        - Same prefix + same adapter = cache HIT (~80% hit rate)
        
        As of PR #7216 (August 2025), SGLang LoRA + RadixCache is supported.
        """
        # Use FIXED adapter name for RadixCache consistency across epochs
        # This is critical: different names = different cache branches = no hits
        lora_name = self._get_fixed_lora_name()
        
        async with aiohttp.ClientSession() as session:
            # Try load_lora_adapter endpoint first (REPLACES weights for same name)
            try:
                async with session.post(
                    f"http://{self._server_host}:{self._server_port}/load_lora_adapter",
                    json={"lora_path": checkpoint_dir, "lora_name": lora_name},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        print(f"  Hot-reloaded LoRA '{lora_name}' step {step} (RadixCache preserved)")
                        return
            except Exception:
                pass
            
            # Fallback: try add_lora endpoint
            try:
                async with session.post(
                    f"http://{self._server_host}:{self._server_port}/add_lora",
                    json={
                        "lora_path": checkpoint_dir,
                        "lora_name": lora_name,
                        "lora_int_id": self._next_lora_id(),
                    },
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        print(f"  Added LoRA '{lora_name}' step {step} (RadixCache preserved)")
                        return
                    error_text = await resp.text()
                    print(f"Warning: Failed to hot-reload LoRA: {error_text}")
            except Exception as e:
                print(f"Warning: Failed to hot-reload LoRA: {e}")

        # NOTE: We intentionally do NOT add step-specific aliases (model@1, model@2, etc.)
        # Each unique adapter name creates a separate radix tree branch in SGLang,
        # which kills cache hit rates. The fixed name (model@latest) is sufficient.
        # Step tracking is done via _latest_step and checkpoint directories.

    # ========================================================================
    # Megatron Training (same as MegatronService)
    # ========================================================================

    async def _ensure_megatron_running(self, use_gpu_isolation: bool = False) -> None:
        """Lazily start Megatron training process if not running.
        
        Args:
            use_gpu_isolation: If True, run Megatron only on GPUs not used by SGLang.
                              This allows SGLang to keep running and preserve cache.
        """
        if self._megatron_process is not None:
            if self._megatron_process.returncode is None:
                return
            self._megatron_process = None

        # Check if megatron.bridge is installed by looking at the file system.
        # We do NOT import it (importing loads CUDA extensions that poison
        # fork()ed children) and we do NOT use a subprocess with
        # CUDA_VISIBLE_DEVICES="" (that makes the import fail every time,
        # causing setup.sh to re-run on every iteration).
        need_setup = not any(
            os.path.isdir(os.path.join(p, "megatron", "bridge"))
            for p in sys.path
            if os.path.isdir(p)
        )

        # Kill ALL stale training processes — not just "megatron-service".
        # torch.distributed.run spawns processes named "train.py" / "torchrun"
        # that hold port 29500. If we only pkill "megatron-service", these
        # survive and block the next run with EADDRINUSE.
        subprocess.run(["pkill", "-9", "megatron-service"], check=False)
        subprocess.run(["pkill", "-9", "-f", "megatron/train.py"], check=False)
        subprocess.run(["pkill", "-9", "-f", "torchrun"], check=False)
        subprocess.run(["pkill", "-9", "-f", "torch.distributed.run"], check=False)
        # Wait for ports (29500) to be released by the kernel
        import time as _time_mod
        _time_mod.sleep(1.0)
        train_script = Path(__file__).parent / "train.py"

        # Ensure GPUs are in DEFAULT compute mode (not EXCLUSIVE_PROCESS).
        smi_result = subprocess.run(
            ["nvidia-smi", "-c", "0"],
            capture_output=True, text=True, check=False,
        )
        print(f"  nvidia-smi -c 0: rc={smi_result.returncode}")
        if smi_result.stdout.strip():
            print(f"    {smi_result.stdout.strip()}")
        if smi_result.returncode != 0 and smi_result.stderr.strip():
            print(f"    stderr: {smi_result.stderr.strip()[:200]}")

        # Build a CLEAN environment for the training subprocess.
        train_env = dict(os.environ)
        train_env["MODEL_IDENTIFIER"] = self.base_model
        train_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # Remove any stale CUDA variables that the parent may have set
        train_env.pop("CUDA_VISIBLE_DEVICES", None)

        # ── FIX: LD_LIBRARY_PATH for CUDA 12.x / 13.0 mismatch ──
        # The system driver is CUDA 13.0 but PyTorch was compiled for CUDA 12.8.
        # Without LD_LIBRARY_PATH, the training subprocess loads the system's
        # libcudart.so.13 instead of the venv's bundled libcudart.so.12, which
        # causes cudaErrorDevicesUnavailable on set_device().
        # Prioritize the main venv's nvidia libs (CUDA 12.x) over the system's.
        venv_site = None
        try:
            import site as _site
            venv_site = _site.getsitepackages()[0]
        except Exception:
            pass
        if not venv_site or not os.path.isdir(str(venv_site)):
            # Derive from sys.executable
            _venv_root = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
            _lib = os.path.join(_venv_root, "lib")
            if os.path.isdir(_lib):
                for _d in os.listdir(_lib):
                    if _d.startswith("python"):
                        _cand = os.path.join(_lib, _d, "site-packages")
                        if os.path.isdir(_cand):
                            venv_site = _cand
                            break

        train_nvidia_lib_paths: list[str] = []
        if venv_site and os.path.isdir(str(venv_site)):
            _nvidia_dir = os.path.join(venv_site, "nvidia")
            if os.path.isdir(_nvidia_dir):
                for _pkg in sorted(os.listdir(_nvidia_dir)):
                    _lib_dir = os.path.join(_nvidia_dir, _pkg, "lib")
                    if os.path.isdir(_lib_dir):
                        train_nvidia_lib_paths.append(_lib_dir)
            _torch_lib = os.path.join(venv_site, "torch", "lib")
            if os.path.isdir(_torch_lib):
                train_nvidia_lib_paths.append(_torch_lib)

        _sys_lib_paths = [
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64",
        ]
        _all_paths = train_nvidia_lib_paths + _sys_lib_paths
        _existing = train_env.get("LD_LIBRARY_PATH", "")
        _new_ld = ":".join(_all_paths)
        if _existing:
            _new_ld = f"{_new_ld}:{_existing}"
        train_env["LD_LIBRARY_PATH"] = _new_ld

        if train_nvidia_lib_paths:
            print(f"  Training CUDA fix: added {len(train_nvidia_lib_paths)} nvidia lib paths from venv")
        else:
            print(f"  WARNING: Could not find nvidia lib paths in training venv (site={venv_site})")
        
        # Determine which GPUs to use for Megatron
        if use_gpu_isolation:
            megatron_gpu_ids = self.sglang_config.get_megatron_gpu_ids()
            num_gpus = len(megatron_gpu_ids)
            cuda_devices = ",".join(map(str, megatron_gpu_ids))
            train_env["CUDA_VISIBLE_DEVICES"] = cuda_devices
            print(f"  Megatron GPU isolation: using GPUs {megatron_gpu_ids}")
        else:
            num_gpus = torch.cuda.device_count()

        # Strong GPU diagnostic — actually create CUDA contexts and tensors,
        # not just query device names.  Also check compute mode.
        diag_script = (
            "import subprocess, torch, sys\n"
            "# 1. Print compute mode for each GPU\n"
            "r = subprocess.run(['nvidia-smi', '--query-gpu=index,compute_mode', '--format=csv,noheader'],\n"
            "                   capture_output=True, text=True)\n"
            "print('Compute modes:', r.stdout.strip())\n"
            "# 2. Check visible devices\n"
            "import os; print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)'))\n"
            "n = torch.cuda.device_count()\n"
            "print(f'visible_devices={n}')\n"
            "# 3. Actually claim each device (creates CUDA context)\n"
            "for i in range(n):\n"
            "    try:\n"
            "        torch.cuda.set_device(i)\n"
            "        t = torch.tensor([1.0], device=f'cuda:{i}')\n"
            "        print(f'  dev{i}: {torch.cuda.get_device_name(i)} — set_device OK, tensor OK')\n"
            "        del t\n"
            "    except Exception as e:\n"
            "        print(f'  dev{i}: FAILED — {e}')\n"
            "torch.cuda.empty_cache()\n"
        )
        diag = subprocess.run(
            [sys.executable, "-c", diag_script],
            env=train_env, capture_output=True, text=True, timeout=30,
        )
        print(f"  GPU diagnostic stdout:\n{diag.stdout.strip()}")
        if diag.returncode != 0:
            err_tail = diag.stderr.strip()[-500:] if diag.stderr else "(no stderr)"
            print(f"  GPU diagnostic FAILED (rc={diag.returncode}): {err_tail}")

        cv = train_env.get("CUDA_VISIBLE_DEVICES", "(all)")

        # Pick a random master port to avoid EADDRINUSE on 29500.
        # Port 29500 is torch.distributed's default and stale processes
        # or slow kernel cleanup can leave it bound for seconds.
        import random
        master_port = random.randint(29500, 29999)

        if need_setup:
            # First run: need setup.sh, must use shell
            setup_script = Path(__file__).parent / "setup.sh"
            command = (
                f"bash {setup_script} && "
                f"{sys.executable} -m torch.distributed.run "
                f"--nproc_per_node {num_gpus} --master_port {master_port} {train_script}"
            )
            print(f"Starting Megatron training (with setup): {command}")
            print(f"  CUDA_VISIBLE_DEVICES={cv}, master_port={master_port}")
            self._megatron_process = await asyncio.create_subprocess_shell(
                command, env=train_env,
            )
        else:
            # Subsequent runs: use direct exec (no shell, no uv) — most
            # reliable way to ensure CUDA_VISIBLE_DEVICES propagates.
            args = [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node", str(num_gpus),
                "--master_port", str(master_port),
                str(train_script),
            ]
            print(f"Starting Megatron training: {' '.join(args)}")
            print(f"  CUDA_VISIBLE_DEVICES={cv}, master_port={master_port}")
            self._megatron_process = await asyncio.create_subprocess_exec(
                *args, env=train_env,
            )

    async def vllm_engine_is_sleeping(self) -> bool:
        """Check if server is sleeping (for compatibility with LocalBackend)."""
        return self._is_sleeping

    async def _is_server_healthy(self) -> bool:
        """Check if SGLang server is running and healthy."""
        if self._server_process is None:
            return False
        poll_result = self._server_process.poll()
        if poll_result is not None:
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{self._server_host}:{self._server_port}/health",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    return resp.status == 200
        except Exception as e1:
            # Try /v1/models as fallback
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{self._server_host}:{self._server_port}/v1/models",
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        return resp.status == 200
            except Exception:
                return False

    async def _unload_old_loras(self, keep_step: int | None = None) -> None:
        """Unload old LoRA adapters to free memory.
        
        SGLang may accumulate step-aliased LoRAs across training steps.
        The fixed-name adapter (model@latest) is always kept since it's
        the one used for inference. Step aliases (model@1, model@2, ...)
        are unloaded to prevent memory creep.
        
        Uses SGLang's /unload_lora_adapter or /delete_lora_adapter endpoint
        if available.
        """
        if keep_step is None:
            return
        
        fixed_name = self._get_fixed_lora_name()
        
        # Try to unload any step-specific aliases that may have accumulated
        # We don't know exactly which steps exist, so try recent ones
        for old_step in range(max(0, keep_step - 10), keep_step):
            old_name = f"{self.model_name}@{old_step}"
            if old_name == fixed_name:
                continue  # Never unload the fixed adapter
            
            for endpoint in ("/unload_lora_adapter", "/delete_lora_adapter"):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"http://{self._server_host}:{self._server_port}{endpoint}",
                            json={"lora_name": old_name},
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as resp:
                            if resp.status == 200:
                                print(f"  Unloaded old LoRA: {old_name}")
                                break  # Success, no need to try other endpoint
                except Exception:
                    pass  # Endpoint may not exist in this SGLang version

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Run Megatron training step.
        
        Cache methods:
        - "sleep_wake" (RECOMMENDED): Uses SGLang's native /release_memory_occupation
              to offload weights + CUDA graphs to CPU. KV cache stays on GPU.
              Server stays alive (idle) with GPU isolation — no SIGSTOP.
              After training: /update_weights_from_disk (NOT resume_memory_occupation
              which is broken for MoE — issue #6367) → hot-reload LoRA.
              Cache PRESERVED. Training speed matches vLLM sleep_wake.
        - "freeze": GPU isolation only — server stays alive and idle.
              GPU memory stays allocated (weights + KV cache on SGLang GPUs).
              Megatron trains on separate GPUs. Cache 100% preserved.
              NOTE: No SIGSTOP — it causes cudaErrorDevicesUnavailable system-wide.
        - "hot_reload": Keep server fully active during training.
              Preserves cache but causes CPU/PCIe contention. Best for 4+ GPUs.
        - "restart": Kill server before training, restart after.
              Cache LOST. Fallback only.
        - "sleep": (DEPRECATED) Old custom launcher approach.
        """
        import time as _time
        _t0 = _time.perf_counter()
        
        cache_method = self.sglang_config.cache_method
        server_alive = (
            self._server_process is not None
            and self._server_process.poll() is None
        )
        
        # Determine strategy based on cache_method
        if cache_method == "sleep_wake" and server_alive:
            strategy = "sleep_wake"
        elif cache_method == "sleep" and server_alive:
            strategy = "sleep"
        elif cache_method in ("sleep_wake", "sleep", "freeze") and server_alive:
            # sleep_wake/sleep fall back to freeze if endpoint fails
            strategy = "freeze"
        elif cache_method == "hot_reload" and self.sglang_config.can_preserve_cache() and server_alive:
            if await self._is_server_healthy():
                strategy = "hot_reload"
            else:
                strategy = "restart"
        else:
            strategy = "restart"
        _t_health = _time.perf_counter()
        
        # === PRE-TRAINING: Prepare GPU for Megatron ===
        _slept = False
        _native_sleep_tags = ["weights", "cuda_graph"]  # Keep KV cache for prefix caching!
        
        if strategy == "sleep_wake":
            print(f"[SGLang] Method: Native Sleep/Wake (offload weights+CUDA graphs, keep KV cache)")
            print(f"  SGLang GPUs: {self.sglang_config.get_sglang_gpu_ids()}")
            print(f"  Megatron GPUs: {self.sglang_config.get_megatron_gpu_ids()}")
            
            # Offload weights + CUDA graphs via native endpoint
            # KV cache stays on GPU -> prefix cache preserved!
            release_ok = await self._native_release_memory(tags=_native_sleep_tags)
            if release_ok:
                _slept = True
                print(f"  \u2713 Weights + CUDA graphs offloaded to CPU (KV cache stays on GPU)")
                print(f"  Server idle with minimal GPU footprint")
            else:
                print(f"  \u26a0\ufe0f Native release_memory failed, falling back to restart")
                strategy = "restart"
                await self._stop_sglang_server()
                self._is_sleeping = True
                gc_and_empty_cuda_cache()
            use_gpu_isolation = True
        elif strategy == "sleep":
            print(f"[SGLang] Method: Sleep (DEPRECATED — offload weights to CPU)")
            print(f"  SGLang GPUs: {self.sglang_config.get_sglang_gpu_ids()}")
            print(f"  Megatron GPUs: {self.sglang_config.get_megatron_gpu_ids()}")
            
            # Offload weights via /art/sleep endpoint (old custom launcher)
            sleep_result = await self._sleep_server()
            if sleep_result is not None:
                _slept = True
                print(f"  ✓ Model weights offloaded: {sleep_result.get('gpu_freed_gb', '?')}GB "
                      f"in {sleep_result.get('offload_time_s', '?')}s")
                print(f"  Server idle with minimal GPU footprint (no SIGSTOP needed)")
            else:
                print(f"  ⚠️ Sleep endpoint unavailable, falling back to restart")
                strategy = "restart"
                await self._stop_sglang_server()
                self._is_sleeping = True
                gc_and_empty_cuda_cache()
            # NOTE: Do NOT SIGSTOP — see sleep_wake comment above.
            use_gpu_isolation = True
        elif strategy == "freeze":
            # GPU isolation only — do NOT SIGSTOP the server.
            # SIGSTOP causes the NVIDIA driver to report
            # cudaErrorDevicesUnavailable for ALL new CUDA contexts,
            # even on completely separate GPUs.  The driver needs
            # inter-process coordination for memory management and a
            # SIGSTOP'd process can't respond.
            #
            # With separate GPUs, contention is negligible — SGLang
            # sits idle on GPU 0 while Megatron trains on GPUs 2,3.
            print(f"[SGLang] Method: Freeze (GPU isolation — cache PRESERVED)", flush=True)
            print(f"  SGLang GPUs: {self.sglang_config.get_sglang_gpu_ids()} (server idle, memory stays)", flush=True)
            print(f"  Megatron GPUs: {self.sglang_config.get_megatron_gpu_ids()}", flush=True)
            use_gpu_isolation = True
        elif strategy == "hot_reload":
            print(f"[SGLang] Method: Hot-Reload (cache PRESERVED, ⚠️ process stays active)")
            print(f"  SGLang GPUs: {self.sglang_config.get_sglang_gpu_ids()}")
            print(f"  Megatron GPUs: {self.sglang_config.get_megatron_gpu_ids()}")
            print(f"  ⚠️ Server stays active → training may be slower due to contention")
            use_gpu_isolation = True
        else:
            # restart
            if cache_method != "restart":
                print(f"[SGLang] Method: Restart (fallback — {cache_method} not available)")
            else:
                print(f"[SGLang] Method: Restart (server stopped → cache LOST)")
            await self._stop_sglang_server()
            self._is_sleeping = True
            gc_and_empty_cuda_cache()
            print(f"  GPU memory freed.")
            use_gpu_isolation = self.sglang_config.training_gpu_ids is not None
        _t_pre = _time.perf_counter()

        # Start Megatron training
        await self._ensure_megatron_running(use_gpu_isolation=use_gpu_isolation)
        _t_megatron_start = _time.perf_counter()

        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
        self._ensure_lora_adapter_config(lora_path)

        self._optimizer_state_path = self._get_optimizer_state_path()

        # Submit job to Megatron
        jobs_dir = "/tmp/megatron_training_jobs"
        os.makedirs(jobs_dir, exist_ok=True)
        for job_name in os.listdir(jobs_dir):
            if job_name.endswith(".json"):
                os.remove(os.path.join(jobs_dir, job_name))
        
        log_file_path = "/tmp/megatron_training_log.jsonl"
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        
        job = MegatronTrainingJob(
            lora_path=lora_path,
            optimizer_state_path=self._optimizer_state_path,
            disk_packed_tensors=disk_packed_tensors,
            config=config,
            experimental_config=_config,
        )
        job_path = os.path.join(jobs_dir, f"{datetime.datetime.now().isoformat()}.json")
        with open(job_path, "w") as f:
            f.write(job.model_dump_json())
        _t_job_submit = _time.perf_counter()

        # Wait for training to complete
        num_lines = 0
        log_found = False
        print(f"[SGLang] Waiting for training, reading from: {log_file_path}", flush=True)
        while True:
            await asyncio.sleep(0.1)
            try:
                with open(log_file_path, "r") as log_file:
                    if not log_found:
                        print(f"[SGLang] Log file found, monitoring...", flush=True)
                        log_found = True
                    lines = log_file.readlines()[num_lines:]
                    for line in lines:
                        if line := line.strip():
                            if line == "all done":
                                print("[SGLang] DEBUG: Found 'all done' in log file", flush=True)
                                print("[SGLang] Training complete, merging LoRA adapters...", flush=True)
                                self._merge_lora_adapter(lora_path)
                                print("[SGLang] LoRA merge complete", flush=True)
                                try:
                                    os.remove(log_file_path)
                                except Exception:
                                    pass
                                break
                            num_lines += 1
                            yield json.loads(line)
                    else:
                        continue
                    break
            except FileNotFoundError:
                continue
        print(f"[SGLang] DEBUG: Exited training loop", flush=True)
        _t_training_done = _time.perf_counter()

        # Save new checkpoint
        next_step = self._latest_step + 1
        new_checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        print(f"[SGLang] DEBUG: Saving checkpoint to {new_checkpoint_dir}", flush=True)
        os.makedirs(new_checkpoint_dir, exist_ok=True)
        shutil.copy(
            f"{lora_path}/adapter_model.safetensors",
            f"{new_checkpoint_dir}/adapter_model.safetensors",
        )
        self._ensure_lora_adapter_config(new_checkpoint_dir, source_path=lora_path)
        print(f"[SGLang] DEBUG: Checkpoint saved", flush=True)
        _t_checkpoint = _time.perf_counter()

        # === AFTER TRAINING: Resume inference + reload weights ===
        print(f"[SGLang] DEBUG: Post-training phase starting. strategy={strategy}", flush=True)
        _reload_method = strategy
        if strategy in ("sleep_wake", "sleep", "freeze"):
            # Server was NOT frozen (no SIGSTOP), just check health
            print(f"[SGLang] DEBUG: Checking server health...", flush=True)
            thaw_ok = await self._is_server_healthy()
            print(f"[SGLang] DEBUG: Server health check result: {thaw_ok}", flush=True)
            
            if thaw_ok:
                try:
                    # Resume weights if we used native sleep/wake
                    # NOTE: We use update_weights_from_disk instead of resume_memory_occupation.
                    # resume_memory_occupation is BROKEN for MoE models (SGLang issue #6367):
                    # it restores from CPU backup but expert layers don't get properly placed
                    # on GPU, causing garbage output. update_weights_from_disk reloads ALL
                    # weights fresh from disk, which works correctly for MoE models.
                    # The KV cache stays on GPU throughout (prefix cache preserved).
                    if _slept and strategy == "sleep_wake":
                        print(f"[SGLang] DEBUG: Reloading base weights from disk (MoE-safe)...", flush=True)
                        reload_ok = await self._update_weights_from_disk()
                        if reload_ok:
                            print(f"  ✓ Base weights reloaded from disk (MoE-safe)", flush=True)
                        else:
                            print(f"  ⚠️ update_weights_from_disk failed — server may need restart", flush=True)
                    elif _slept and strategy == "sleep":
                        # Old custom launcher wake
                        print(f"[SGLang] DEBUG: Calling wake (old launcher)...", flush=True)
                        wake_result = await self._wake_server()
                        if wake_result:
                            print(f"  ✓ Weights reloaded: {wake_result.get('gpu_reloaded_gb', '?')}GB "
                                  f"in {wake_result.get('reload_time_s', '?')}s", flush=True)
                        else:
                            print(f"  ⚠️ Wake failed — server may need restart", flush=True)
                    
                    # Wait a moment for server to stabilize after resume
                    if _slept:
                        await asyncio.sleep(1.0)
                        # Verify server is healthy after memory resume
                        for i in range(10):
                            if await self._is_server_healthy():
                                break
                            await asyncio.sleep(1.0)
                    
                    print(f"[SGLang] DEBUG: About to hot-reload LoRA...", flush=True)
                    await asyncio.wait_for(
                        self._hot_reload_lora(new_checkpoint_dir, next_step),
                        timeout=120.0
                    )
                    print(f"[SGLang] DEBUG: Hot-reload complete", flush=True)
                    await self._unload_old_loras(keep_step=next_step)
                    if _slept:
                        _reload_method = f"{strategy}+hot_reload"
                    else:
                        _reload_method = "gpu_isolation+hot_reload"
                    print(f"[SGLang] ✓ Server resumed + LoRA reloaded (cache preserved)", flush=True)
                except (asyncio.TimeoutError, Exception) as e:
                    print(f"[SGLang] ⚠️ Post-thaw reload failed: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    print(f"[SGLang] ⚠️ RESTARTING SERVER — cache will be LOST", flush=True)
                    self.sglang_config.cache_method = "freeze"
                    await self._stop_sglang_server()
                    gc_and_empty_cuda_cache()
                    await self._start_sglang_server(new_checkpoint_dir)
                    await self._hot_reload_lora(new_checkpoint_dir, next_step)
                    _reload_method = f"{strategy}_failed→restart"
            else:
                print(f"[SGLang] ⚠️ Server dead after thaw attempts", flush=True)
                print(f"[SGLang] ⚠️ RESTARTING SERVER — cache will be LOST", flush=True)
                self.sglang_config.cache_method = "freeze"
                await self._stop_sglang_server()
                gc_and_empty_cuda_cache()
                await self._start_sglang_server(new_checkpoint_dir)
                await self._hot_reload_lora(new_checkpoint_dir, next_step)
                _reload_method = "thaw_failed→restart"
        elif strategy == "hot_reload":
            # Server was active the whole time, just hot-reload
            print(f"[SGLang] Hot-reloading LoRA for step {next_step} (cache preserved)")
            try:
                if not await self._is_server_healthy():
                    raise RuntimeError("Server became unhealthy during training")
                await asyncio.wait_for(
                    self._hot_reload_lora(new_checkpoint_dir, next_step),
                    timeout=120.0
                )
                await self._unload_old_loras(keep_step=next_step)
                _reload_method = "hot_reload"
                print(f"[SGLang] Hot-reload successful (cache preserved)")
            except (asyncio.TimeoutError, Exception) as e:
                _reload_method = "hot_reload_failed→restart"
                print(f"[SGLang] ⚠️ Hot-reload FAILED: {e}, restarting server")
                await self._stop_sglang_server()
                gc_and_empty_cuda_cache()
                await self._start_sglang_server(new_checkpoint_dir)
                await self._hot_reload_lora(new_checkpoint_dir, next_step)
        else:
            # restart — server was killed, start fresh
            _reload_method = "server_restart"
            print(f"[SGLang] Restarting server with LoRA step {next_step}...")
            await self._start_sglang_server(new_checkpoint_dir)
            await self._hot_reload_lora(new_checkpoint_dir, next_step)
        _t_reload = _time.perf_counter()
        
        self._latest_step = next_step
        self._is_sleeping = False

        # Print detailed timing breakdown
        print(f"\n[SGLang] ═══ Training Timing Breakdown ═══")
        print(f"  Health check:       {(_t_health - _t0):.2f}s")
        print(f"  Pre-training:       {(_t_pre - _t_health):.2f}s (method={cache_method})")
        print(f"  Megatron startup:   {(_t_megatron_start - _t_pre):.2f}s")
        print(f"  Job submission:     {(_t_job_submit - _t_megatron_start):.2f}s")
        print(f"  Megatron training:  {(_t_training_done - _t_job_submit):.2f}s")
        print(f"  Checkpoint save:    {(_t_checkpoint - _t_training_done):.2f}s")
        print(f"  Weight reload:      {(_t_reload - _t_checkpoint):.2f}s ({_reload_method})")
        print(f"  ─────────────────────────────────────")
        print(f"  TOTAL:              {(_t_reload - _t0):.2f}s")
        print(f"═══════════════════════════════════════\n")

        if verbose:
            print(f"SGLangMegatronService.train complete (step {next_step})")

    def _merge_lora_adapter(self, lora_path: str) -> None:
        """Merge sharded LoRA adapters from distributed training."""
        base_dir = Path(lora_path)
        shard_filenames = sorted(base_dir.glob("adapter_model-*-of-*.safetensors"))
        if not shard_filenames:
            return

        adapter_model_path = base_dir / "adapter_model.safetensors"
        sharded_tensors: dict[str, list[torch.Tensor]] = {}

        for filename in shard_filenames:
            with safe_open(filename, framework="pt") as file:
                for key in file.keys():
                    tensor = file.get_tensor(key)
                    sharded_tensors.setdefault(key, []).append(tensor)

        adapter_model: dict[str, torch.Tensor] = {}
        if adapter_model_path.exists():
            adapter_model = load_file(adapter_model_path)

        for key, tensors in sharded_tensors.items():
            tensor = torch.cat(tensors, dim=1 if "lora_A" in key else 0)
            adapter_model[key] = tensor

        save_file(adapter_model, adapter_model_path)
        for filename in shard_filenames:
            filename.unlink()

    async def shutdown(self) -> None:
        """Clean shutdown of service."""
        await self._stop_sglang_server()
        
        if self._megatron_process:
            self._megatron_process.terminate()
            try:
                await asyncio.wait_for(self._megatron_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._megatron_process.kill()
            self._megatron_process = None
