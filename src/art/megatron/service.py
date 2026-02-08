import asyncio
from dataclasses import asdict, dataclass
import datetime
from functools import cached_property
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import AsyncIterator

import aiohttp
from peft.tuners.lora.config import LoraConfig
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from vllm import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev, types
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.pack import DiskPackedTensors
from ..unsloth.service import do_sleep, do_wake_up, gc_and_empty_cuda_cache
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir
from ..vllm import get_llm, openai_server_task, run_on_workers


class MegatronTrainingJob(BaseModel):
    """Job format for communication with train.py"""

    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    experimental_config: dev.TrainConfig
    log_file_path: str = "/tmp/megatron_training_log.jsonl"


from typing import Literal

CacheMethod = Literal["http", "sleep_wake", "none"]


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


@dataclass
class VLLMConfig:
    """Configuration for vLLM engine with cache preservation support.
    
    Attributes:
        tensor_parallel_size: TP size for vLLM (None = auto-detect)
        preserve_cache_during_training: Enable cache preservation
        vllm_gpu_ids: Explicit GPU IDs for vLLM (e.g., [0])
        training_gpu_ids: Explicit GPU IDs for Megatron training (e.g., [1])
            If None, uses all GPUs not used by vLLM (for HTTP) or all GPUs (for sleep_wake)
        cache_method: Method for cache preservation:
            - "http": Use /v1/load_lora_adapter API (vLLM 0.8+, requires GPU isolation)
            - "sleep_wake": Use do_sleep/do_wake_up with level=2 (offload to CPU)
            - "none": Always restart server (no cache preservation)
    """
    
    tensor_parallel_size: int | None = None
    preserve_cache_during_training: bool = True
    vllm_gpu_ids: list[int] | None = None
    training_gpu_ids: list[int] | None = None  # Explicit GPUs for Megatron training
    cache_method: CacheMethod = "http"  # Default to HTTP hot-reload
    
    def get_tensor_parallel_size(self) -> int:
        """Get tensor parallel size, auto-detecting if not set."""
        if self.tensor_parallel_size is not None:
            return self.tensor_parallel_size
        if self.vllm_gpu_ids is not None:
            return len(self.vllm_gpu_ids)
        gpu_count = torch.cuda.device_count()
        return gpu_count if gpu_count > 0 else 1
    
    def get_vllm_gpu_ids(self) -> list[int]:
        """Get GPU IDs for vLLM engine."""
        if self.vllm_gpu_ids is not None:
            return self.vllm_gpu_ids
        tp_size = self.get_tensor_parallel_size()
        return list(range(tp_size))
    
    def get_megatron_gpu_ids(self) -> list[int]:
        """Get GPU IDs for Megatron training.
        
        If training_gpu_ids is set, use those explicitly.
        Otherwise, auto-detect healthy GPUs not used by vLLM.
        Dead/stuck GPUs (high util, zero memory) are automatically excluded.
        For sleep_wake: use all healthy GPUs (since vLLM is offloaded).
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
        # Auto-detect: use only healthy GPUs
        healthy_gpus = _get_healthy_gpu_ids()
        if self.cache_method == "sleep_wake":
            # sleep_wake frees all GPUs, so Megatron can use all healthy ones
            return healthy_gpus
        # HTTP method: vLLM stays on its GPUs
        vllm_gpus = set(self.get_vllm_gpu_ids())
        return [i for i in healthy_gpus if i not in vllm_gpus]
    
    def can_preserve_cache(self) -> bool:
        """Check if we can preserve cache during training."""
        if not self.preserve_cache_during_training:
            return False
        if self.cache_method == "none":
            return False
        # HTTP method requires GPU isolation (vLLM stays running on dedicated GPUs)
        if self.cache_method == "http":
            megatron_gpus = self.get_megatron_gpu_ids()
            return len(megatron_gpus) > 0
        # Sleep/wake doesn't require GPU isolation (GPU freed via do_sleep + gc)
        return True


@dataclass
class MegatronService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    vllm_config: VLLMConfig | None = None
    _is_sleeping: bool = False
    _latest_step: int = 0
    _lora_id_counter: int = 1
    _megatron_process: asyncio.subprocess.Process | None = None
    _optimizer_state_path: str | None = None
    _server_host: str = "127.0.0.1"
    _server_port: int = 8000

    def __post_init__(self):
        if self.vllm_config is None:
            self.vllm_config = VLLMConfig()
        # Enable runtime LoRA loading for cache preservation (vLLM 0.8+)
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    def _next_lora_id(self) -> int:
        self._lora_id_counter += 1
        return self._lora_id_counter

    def _get_optimizer_state_path(self) -> str:
        if self._optimizer_state_path is not None:
            return self._optimizer_state_path
        self._optimizer_state_path = os.path.join(self.output_dir, "optimizer_states")
        os.makedirs(self._optimizer_state_path, exist_ok=True)
        return self._optimizer_state_path

    def _default_lora_adapter_config(self) -> LoraConfig:
        # Keep in sync with LoRA settings in megatron/train.py.
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
        # Create an identity (zero) LoRA using PEFT so vLLM can load it.
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM

        lora_config = self._default_lora_adapter_config()
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        peft_model = get_peft_model(model, lora_config)
        # Keep LoRA A initialized (trainable) and zero only B for identity.
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.data.zero_()
        os.makedirs(lora_path, exist_ok=True)
        peft_model.save_pretrained(lora_path)
        # Sanitize the adapter config for vLLM 0.15.x compatibility
        self._sanitize_lora_config(lora_path)
        del peft_model, model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _ensure_identity_lora(self, lora_path: str) -> None:
        if self._adapter_has_weights(lora_path):
            return
        self._create_identity_lora(lora_path)

    def _ensure_lora_adapter_config(
        self, lora_path: str, *, source_path: str | None = None
    ) -> None:
        config_path = os.path.join(lora_path, "adapter_config.json")
        if os.path.exists(config_path):
            # Sanitize existing config
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
        with open(config_path, "w") as f:
            json.dump(asdict(self._default_lora_adapter_config()), f)

    async def _add_lora_aliases(
        self, llm: AsyncLLM, step: int, checkpoint_dir: str
    ) -> None:
        """Add LoRA adapter for a new training step during hot-reload.
        
        Only adds the versioned name (e.g., "model@1"). The base alias is already
        registered at server startup via get_openai_server_config and cannot be
        replaced (vLLM has no remove_lora API). The benchmark uses versioned names.
        """
        versioned_name = f"{self.model_name}@{step}"
        added = await llm.add_lora(
            LoRARequest(
                lora_name=versioned_name,
                lora_int_id=self._next_lora_id(),
                lora_path=checkpoint_dir,
            )
        )
        if not added:
            raise RuntimeError(f"Failed to add LoRA adapter for step {step}")
        print(f"[vLLM] Added LoRA adapter: {versioned_name}")
        self._latest_step = step

    async def register_lora_for_step(self, step: int, checkpoint_dir: str) -> None:
        llm = await self.llm
        await llm.pause_generation()
        await self._add_lora_aliases(llm, step, checkpoint_dir)
        await llm.resume_generation()

    async def _hot_reload_lora(self, checkpoint_dir: str, step: int) -> None:
        """Hot-reload LoRA weights via vLLM's /v1/load_lora_adapter API.
        
        This allows updating LoRA weights without restarting the server,
        preserving the KV cache. Requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=True.
        Available in vLLM 0.8+.
        """
        lora_name = f"{self.model_name}@{step}"
        base_url = f"http://{self._server_host}:{self._server_port}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{base_url}/v1/load_lora_adapter",
                    json={"lora_name": lora_name, "lora_path": checkpoint_dir},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        print(f"[vLLM] Hot-reloaded LoRA: {lora_name}")
                        self._latest_step = step
                        return
                    error_text = await resp.text()
                    raise RuntimeError(f"Failed to hot-reload LoRA: {error_text}")
            except aiohttp.ClientError as e:
                raise RuntimeError(f"Failed to connect to vLLM server for hot-reload: {e}")

    async def _unload_old_loras(self, keep_step: int) -> None:
        """Unload old LoRA adapters to free memory."""
        base_url = f"http://{self._server_host}:{self._server_port}"
        
        async with aiohttp.ClientSession() as session:
            if keep_step > 0:
                old_name = f"{self.model_name}@{keep_step - 1}"
                try:
                    async with session.post(
                        f"{base_url}/v1/unload_lora_adapter",
                        json={"lora_name": old_name},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            print(f"[vLLM] Unloaded old LoRA: {old_name}")
                except Exception:
                    pass  # Ignore errors - old LoRA might not exist

    async def _is_server_healthy(self) -> bool:
        """Check if vLLM server is responding."""
        base_url = f"http://{self._server_host}:{self._server_port}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def _ensure_megatron_running(self, use_gpu_isolation: bool = False) -> None:
        """Lazily start Megatron training process if not running.
        
        Args:
            use_gpu_isolation: If True, run Megatron only on GPUs not used by vLLM.
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

        subprocess.run(["pkill", "-9", "megatron-service"], check=False)
        train_script = Path(__file__).parent / "train.py"

        # Ensure GPUs are in DEFAULT compute mode (not EXCLUSIVE_PROCESS).
        subprocess.run(["nvidia-smi", "-c", "0"], capture_output=True, check=False)

        # Build a CLEAN environment for the training subprocess.
        train_env = dict(os.environ)
        train_env["MODEL_IDENTIFIER"] = self.base_model
        train_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        train_env.pop("CUDA_VISIBLE_DEVICES", None)

        # ── FIX: LD_LIBRARY_PATH for CUDA 12.x / 13.0 mismatch ──
        # The system driver is CUDA 13.0 but PyTorch was compiled for CUDA 12.8.
        # Without LD_LIBRARY_PATH, the training subprocess loads the system's
        # libcudart.so.13 instead of the venv's bundled libcudart.so.12, which
        # causes cudaErrorDevicesUnavailable on set_device().
        venv_site = None
        try:
            import site as _site
            venv_site = _site.getsitepackages()[0]
        except Exception:
            pass
        if not venv_site or not os.path.isdir(str(venv_site)):
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

        # Determine which GPUs to use for Megatron
        if use_gpu_isolation:
            megatron_gpu_ids = self.vllm_config.get_megatron_gpu_ids()
            num_gpus = len(megatron_gpu_ids)
            cuda_devices = ",".join(map(str, megatron_gpu_ids))
            train_env["CUDA_VISIBLE_DEVICES"] = cuda_devices
            print(f"  Megatron GPU isolation: using GPUs {megatron_gpu_ids}")
        else:
            num_gpus = torch.cuda.device_count()

        # Quick GPU diagnostic — verify GPUs are reachable from a clean subprocess
        diag_script = (
            "import torch; n=torch.cuda.device_count(); "
            "print(f'visible={n}'); "
            "[print(f'  dev{i}={torch.cuda.get_device_name(i)}') for i in range(n)]"
        )
        diag = subprocess.run(
            [sys.executable, "-c", diag_script],
            env=train_env, capture_output=True, text=True, timeout=30,
        )
        print(f"  GPU diagnostic: {diag.stdout.strip()}")
        if diag.returncode != 0:
            err_tail = diag.stderr.strip()[-500:] if diag.stderr else "(no stderr)"
            print(f"  GPU diagnostic FAILED (rc={diag.returncode}): {err_tail}")

        cv = train_env.get("CUDA_VISIBLE_DEVICES", "(all)")

        if need_setup:
            setup_script = Path(__file__).parent / "setup.sh"
            command = (
                f"bash {setup_script} && "
                f"{sys.executable} -m torch.distributed.run "
                f"--nproc_per_node {num_gpus} {train_script}"
            )
            print(f"Starting Megatron training (with setup): {command}")
            print(f"  CUDA_VISIBLE_DEVICES={cv}")
            self._megatron_process = await asyncio.create_subprocess_shell(
                command, env=train_env,
            )
        else:
            args = [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node", str(num_gpus), str(train_script),
            ]
            print(f"Starting Megatron training: {' '.join(args)}")
            print(f"  CUDA_VISIBLE_DEVICES={cv}")
            self._megatron_process = await asyncio.create_subprocess_exec(
                *args, env=train_env,
            )

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            self._latest_step = 0
        else:
            self._latest_step = get_step_from_dir(self.output_dir)
        self._ensure_identity_lora(lora_path)
        self._ensure_lora_adapter_config(lora_path)

        lora_path_for_server = (
            lora_path if self._adapter_has_weights(lora_path) else None
        )
        server_config = dev.get_openai_server_config(
            model_name=self.model_name,
            base_model=self.base_model,
            log_file=f"{self.output_dir}/logs/vllm.log",
            lora_path=lora_path_for_server,
            config=config,
        )
        await openai_server_task(engine=await self.llm, config=server_config)
        
        # Store server host/port for hot-reload API calls
        host = server_config.get("server_args", {}).get("host") or "0.0.0.0"
        self._server_host = "127.0.0.1" if host == "0.0.0.0" else host
        self._server_port = server_config.get("server_args", {}).get("port", 8000)
        
        return (host, self._server_port)

    async def vllm_engine_is_sleeping(self) -> bool:
        return self._is_sleeping

    async def _is_engine_healthy(self) -> bool:
        """Check if vLLM engine is running and healthy."""
        if "llm" not in self.__dict__:
            return False
        try:
            llm = await self.llm
            # Try a simple operation to verify engine is responsive
            return llm is not None
        except Exception:
            return False

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Train using Megatron with configurable cache preservation method.
        
        Cache methods:
        - "http": Keep server running, hot-reload LoRA via /v1/load_lora_adapter API
                  Requires GPU isolation. KV cache stays on GPU.
        - "sleep_wake": Use do_sleep/do_wake_up with level=1. 
                        Offloads weights + KV cache to CPU, restores after training.
        - "none": Always restart server. Cache is lost.
        """
        import time as _time
        _t0 = _time.perf_counter()
        
        cache_method = self.vllm_config.cache_method
        can_preserve = self.vllm_config.can_preserve_cache() and await self._is_engine_healthy()
        
        # For HTTP method, also check server health
        if cache_method == "http" and can_preserve:
            can_preserve = await self._is_server_healthy()
        
        # Get engine reference for sleep/wake method
        llm = await self.llm if cache_method == "sleep_wake" and can_preserve else None
        _t_health = _time.perf_counter()
        
        # === BEFORE TRAINING: Prepare based on method ===
        if can_preserve and cache_method == "http":
            print(f"[vLLM] Method: HTTP Hot-Reload (KV cache PRESERVED on GPU)")
            print(f"  vLLM GPUs: {self.vllm_config.get_vllm_gpu_ids()}")
            print(f"  Megatron GPUs: {self.vllm_config.get_megatron_gpu_ids()}")
            # Server stays running - GPU isolation keeps cache warm
            use_gpu_isolation = True
            
        elif can_preserve and cache_method == "sleep_wake":
            print(f"[vLLM] Method: Sleep/Wake (weights offloaded to CPU, KV cache discarded)")
            training_gpus = self.vllm_config.get_megatron_gpu_ids()
            print(f"  Training GPUs: {training_gpus}")
            llm = await self.llm
            await llm.pause_generation()
            # level=2: offload weights to CPU, DISCARD KV cache (ART architecture)
            await run_on_workers(llm, do_sleep, level=2)
            self._is_sleeping = True
            gc_and_empty_cuda_cache()
            # Use GPU isolation if training_gpu_ids is explicitly set (for fair comparison)
            use_gpu_isolation = self.vllm_config.training_gpu_ids is not None
            
        else:
            if self.vllm_config.preserve_cache_during_training and cache_method != "none":
                total_gpus = torch.cuda.device_count()
                vllm_gpus = len(self.vllm_config.get_vllm_gpu_ids())
                print(f"[vLLM] Cannot preserve cache with {cache_method}: need more GPUs (have {total_gpus}, vLLM uses {vllm_gpus})")
            print(f"[vLLM] Method: Restart (KV cache will be CLEARED)")
            await self._stop_vllm_engine()
            self._is_sleeping = True
            gc_and_empty_cuda_cache()
            use_gpu_isolation = False
            can_preserve = False  # Force restart path
        _t_pre = _time.perf_counter()

        # Start Megatron training
        await self._ensure_megatron_running(use_gpu_isolation=use_gpu_isolation)
        _t_megatron_start = _time.perf_counter()

        # Setup training job
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
        # Use same log file path as SGLang (the default in MegatronTrainingJob)
        log_file_path = "/tmp/megatron_training_log.jsonl"
        # Remove old log file if exists
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        
        job = MegatronTrainingJob(
            lora_path=lora_path,
            optimizer_state_path=self._optimizer_state_path,
            disk_packed_tensors=disk_packed_tensors,
            config=config,
            experimental_config=_config,
            # Use default log_file_path (/tmp/megatron_training_log.jsonl) same as SGLang
        )
        job_path = os.path.join(jobs_dir, f"{datetime.datetime.now().isoformat()}.json")
        with open(job_path, "w") as f:
            f.write(job.model_dump_json())
        _t_job_submit = _time.perf_counter()

        # Wait for training to complete
        num_lines = 0
        log_found = False
        print(f"[vLLM] Waiting for training, reading from: {log_file_path}")
        while True:
            await asyncio.sleep(0.1)
            try:
                with open(log_file_path, "r") as log_file:
                    if not log_found:
                        print(f"[vLLM] Log file found, monitoring...")
                        log_found = True
                    lines = log_file.readlines()[num_lines:]
                    for line in lines:
                        if line := line.strip():
                            if line == "all done":
                                print("[vLLM] Training complete, merging LoRA adapters...")
                                self._merge_lora_adapter(lora_path)
                                print("[vLLM] LoRA merge complete, saving checkpoint...")
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
        _t_training_done = _time.perf_counter()

        # Save new checkpoint
        next_step = self._latest_step + 1
        new_checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        os.makedirs(new_checkpoint_dir, exist_ok=True)
        shutil.copy(
            f"{lora_path}/adapter_model.safetensors",
            f"{new_checkpoint_dir}/adapter_model.safetensors",
        )
        self._ensure_lora_adapter_config(new_checkpoint_dir, source_path=lora_path)
        _t_checkpoint = _time.perf_counter()

        # Note: Don't kill Megatron - it's a long-running service that handles multiple jobs
        # (same as SGLang's approach)

        # === AFTER TRAINING: Restore/reload based on method ===
        _reload_method = "none"
        if can_preserve and cache_method == "http":
            # HTTP Hot-Reload: Use /v1/load_lora_adapter API
            print(f"[vLLM] Hot-reloading LoRA for step {next_step}...")
            try:
                if not await self._is_server_healthy():
                    raise RuntimeError("Server became unhealthy during training")
                
                await asyncio.wait_for(
                    self._hot_reload_lora(new_checkpoint_dir, next_step),
                    timeout=120.0
                )
                await self._unload_old_loras(keep_step=next_step)
                _reload_method = "http_hot_reload"
                print(f"[vLLM] HTTP hot-reload successful (cache preserved)")
            except (asyncio.TimeoutError, Exception) as e:
                _reload_method = "engine_restart"
                print(f"[vLLM] HTTP hot-reload failed: {e}, falling back to restart")
                await self._stop_vllm_engine()
                gc_and_empty_cuda_cache()
                await self._start_vllm_engine(new_checkpoint_dir, next_step)
                
        elif can_preserve and cache_method == "sleep_wake":
            # Sleep/Wake: Megatron already offloaded to CPU (see train.py line 346)
            # Just need to ensure GPU memory is freed before vLLM wakes up
            print(f"[vLLM] Waiting for Megatron GPU memory cleanup...")
            await asyncio.sleep(0.5)  # Allow async GPU cleanup to complete
            gc_and_empty_cuda_cache()
            print(f"[vLLM] Waking up engine and loading LoRA for step {next_step}...")
            await run_on_workers(llm, do_wake_up)
            await llm.resume_generation()
            # Use HTTP /v1/load_lora_adapter to register the new LoRA.
            # Engine-level add_lora alone does NOT update the OpenAI API model
            # list in newer vLLM versions (the _openai_serving_models capture
            # fails for vLLM v1), so the model name would return 404.
            try:
                await asyncio.wait_for(
                    self._hot_reload_lora(new_checkpoint_dir, next_step),
                    timeout=60.0,
                )
            except Exception as e:
                # Fallback: try engine-level add (works if serving_models was captured)
                print(f"[vLLM] HTTP LoRA load failed ({e}), trying engine-level add...")
                await llm.pause_generation()
                await self._add_lora_aliases(llm, next_step, new_checkpoint_dir)
                await llm.resume_generation()
            _reload_method = "sleep_wake"
            print(f"[vLLM] Sleep/wake restore successful")
            
        else:
            # Restart: Start fresh vLLM engine with new LoRA
            _reload_method = "engine_restart"
            print(f"[vLLM] Starting vLLM engine for step {next_step}...")
            await self._start_vllm_engine(new_checkpoint_dir, next_step)
        _t_reload = _time.perf_counter()
        
        self._latest_step = next_step
        self._is_sleeping = False
        
        # Print detailed timing breakdown for debugging training speed
        print(f"\n[vLLM] ═══ Training Timing Breakdown ═══")
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
            print(f"[vLLM] Training complete (step {next_step})")

    async def _stop_vllm_engine(self) -> None:
        """Stop vLLM engine completely (like SGLang's _stop_sglang_server)."""
        if "llm" in self.__dict__:
            try:
                llm = await self.llm
                await llm.shutdown()
            except Exception:
                pass
            del self.__dict__["llm"]
        
        # Kill any vLLM processes
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], check=False)
        subprocess.run(["pkill", "-9", "-f", "vllm.v1"], check=False)
        await asyncio.sleep(1.0)

    async def _kill_megatron(self) -> None:
        """Kill Megatron subprocess."""
        print("[vLLM] Killing Megatron process...")
        if self._megatron_process is not None:
            self._megatron_process.terminate()
            try:
                await asyncio.wait_for(self._megatron_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                print("[vLLM] Megatron didn't terminate, force killing...")
                self._megatron_process.kill()
                try:
                    await asyncio.wait_for(self._megatron_process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    print("[vLLM] Force kill timed out, continuing anyway")
            self._megatron_process = None
        
        # Kill any orphaned processes
        subprocess.run(["pkill", "-9", "-f", "megatron/train.py"], check=False)
        subprocess.run(["pkill", "-9", "-f", "torchrun"], check=False)
        await asyncio.sleep(0.5)
        gc_and_empty_cuda_cache()
        print("[vLLM] Megatron killed")

    async def _start_vllm_engine(self, checkpoint_dir: str, step: int) -> None:
        """Start fresh vLLM engine (like SGLang's _start_sglang_server)."""
        print(f"[vLLM] Starting fresh vLLM engine with checkpoint {checkpoint_dir} (step {step})...")
        
        # Get fresh engine
        llm = await self.llm
        
        # Start the OpenAI server with LoRA config
        # Note: get_openai_server_config already registers BOTH model_name and model_name@step
        # via lora_modules, so we don't need to call _add_lora_aliases separately
        lora_path_for_server = (
            checkpoint_dir if self._adapter_has_weights(checkpoint_dir) else None
        )
        server_config = dev.get_openai_server_config(
            model_name=self.model_name,
            base_model=self.base_model,
            log_file=f"{self.output_dir}/logs/vllm.log",
            lora_path=lora_path_for_server,
            config=None,
        )
        await openai_server_task(engine=llm, config=server_config)
        
        # Update tracking (LoRA is already registered via lora_modules in server config)
        self._latest_step = step
        
        await llm.resume_generation()
        print(f"[vLLM] Engine started successfully with LoRA step {step}")

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
            if len(tensors) == 1:
                adapter_model[key] = tensors[0]
            elif len(tensors) > 1 and torch.equal(tensors[0], tensors[1]):
                # Duplicate: unsharded param saved by multiple TP ranks.
                # This happens when expert LoRA's sharded_lora_state_dict doesn't
                # filter by TP rank for non-sharded params (e.g., lora_A in
                # column-parallel MoE layers). Take one copy, don't concatenate.
                adapter_model[key] = tensors[0]
            else:
                tensor = torch.cat(tensors, dim=1 if "lora_A" in key else 0)
                adapter_model[key] = tensor

        save_file(adapter_model, adapter_model_path)
        for filename in shard_filenames:
            filename.unlink()

    @cached_property
    def llm(self) -> asyncio.Task[AsyncLLM]:
        engine_args = {
            **self.config.get("engine_args", {}),
            "enable_lora": True,
            "max_loras": self.config.get("engine_args", {}).get("max_loras", 2),
            "enable_prefix_caching": True,  # Enable automatic prefix caching
        }
        for key in ["enable_log_requests", "disable_log_requests"]:
            engine_args.pop(key, None)
        return asyncio.create_task(get_llm(AsyncEngineArgs(**engine_args)))  # type: ignore
