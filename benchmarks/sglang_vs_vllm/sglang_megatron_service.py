"""
SGLang + Megatron service — drop-in replacement for MegatronService.

This module mirrors art.megatron.service.MegatronService but uses SGLang
for inference instead of vLLM.  The Megatron training loop (train.py) is
shared between both backends.

Architecture
~~~~~~~~~~~~
1. SGLang server runs as a subprocess (via SGLangServer).
2. When training is requested:
   a. SGLang server is stopped (freeing GPU memory).
   b. Megatron training process runs (exactly the same as vLLM backend).
   c. LoRA shards are merged.
   d. SGLang server is restarted with the new LoRA.
3. The OpenAI-compatible API endpoint is identical so the rollout code
   works with both backends without modification.

Why stop/restart instead of sleep/wake?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
vLLM's in-process design allows fine-grained sleep/wake through its
CuMemAllocator.  SGLang runs as a separate process, so the cleanest
way to free GPU memory is to terminate and restart the server.  This is
a fair comparison because:
- vLLM's sleep/wake has overhead (offload/reload weights)
- SGLang's restart has overhead (model loading)
- Both achieve the same goal: free GPU memory for Megatron training
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from peft.tuners.lora.config import LoraConfig
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from .sglang_server import SGLangServer, SGLangServerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job format (shared with Megatron train.py — MUST stay in sync)
# ---------------------------------------------------------------------------

class SGLangMegatronTrainingJob(BaseModel):
    """Job format for communication with Megatron train.py."""
    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: dict  # DiskPackedTensors
    config: dict  # types.TrainConfig
    experimental_config: dict  # dev.TrainConfig


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

class TimingContext:
    """Simple context manager to time a block of code."""
    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# SGLangMegatronService
# ---------------------------------------------------------------------------

@dataclass
class SGLangMegatronService:
    """
    Manages SGLang inference + Megatron training lifecycle.

    This is the SGLang counterpart of art.megatron.service.MegatronService.
    It implements the same interface so it can be used as a drop-in
    replacement for benchmarking.
    """

    model_name: str
    base_model: str
    output_dir: str
    sglang_python: str = "python"
    port: int = 8200
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 256
    log_dir: str = ""

    # Internal state
    _server: SGLangServer | None = None
    _latest_step: int = 0
    _megatron_process: asyncio.subprocess.Process | None = None
    _optimizer_state_path: str | None = None
    _is_sleeping: bool = False

    # Timing metrics from the last training step
    _last_server_stop_time: float = 0.0
    _last_server_start_time: float = 0.0
    _last_training_time: float = 0.0
    _last_lora_merge_time: float = 0.0

    def __post_init__(self) -> None:
        if not self.log_dir:
            self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # LoRA adapter management (mirrors MegatronService)
    # ------------------------------------------------------------------

    def _default_lora_adapter_config(self) -> LoraConfig:
        """Keep in sync with LoRA settings in megatron/train.py."""
        return LoraConfig(
            r=1,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
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

    def _create_identity_lora(self, lora_path: str) -> None:
        """Create a zero (identity) LoRA adapter."""
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
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.data.zero_()
        os.makedirs(lora_path, exist_ok=True)
        peft_model.save_pretrained(lora_path)
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
            return
        os.makedirs(lora_path, exist_ok=True)
        if source_path is not None:
            source_config = os.path.join(source_path, "adapter_config.json")
            if os.path.exists(source_config):
                shutil.copy(source_config, config_path)
                return
        with open(config_path, "w") as f:
            json.dump(asdict(self._default_lora_adapter_config()), f)

    def _get_optimizer_state_path(self) -> str:
        if self._optimizer_state_path is not None:
            return self._optimizer_state_path
        self._optimizer_state_path = os.path.join(self.output_dir, "optimizer_states")
        os.makedirs(self._optimizer_state_path, exist_ok=True)
        return self._optimizer_state_path

    def _get_checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.output_dir, "checkpoints", f"{step:04d}")

    def _get_last_checkpoint_dir(self) -> str | None:
        checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            return None
        steps = sorted(
            int(d)
            for d in os.listdir(checkpoints_dir)
            if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.isdigit()
        )
        if not steps:
            return None
        return os.path.join(checkpoints_dir, f"{steps[-1]:04d}")

    # ------------------------------------------------------------------
    # Server management
    # ------------------------------------------------------------------

    def _create_server(self, lora_path: str | None = None) -> SGLangServer:
        """Create a new SGLang server instance."""
        lora_paths = []
        if lora_path and self._adapter_has_weights(lora_path):
            lora_paths.append(lora_path)

        config = SGLangServerConfig(
            model_path=self.base_model,
            port=self.port,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=self.max_num_seqs,
            python_executable=self.sglang_python,
            log_file=os.path.join(self.log_dir, "sglang.log"),
            lora_paths=lora_paths,
            trust_remote_code=True,
        )
        return SGLangServer(config)

    async def start_openai_server(self) -> tuple[str, int]:
        """
        Start the SGLang server and return (host, port).

        Mirrors MegatronService.start_openai_server().
        """
        lora_path = self._get_last_checkpoint_dir()
        if lora_path is None:
            lora_path = self._get_checkpoint_dir(0)
            self._latest_step = 0
        else:
            # Extract step from path
            self._latest_step = int(os.path.basename(lora_path))

        self._ensure_identity_lora(lora_path)
        self._ensure_lora_adapter_config(lora_path)

        self._server = self._create_server(lora_path)
        startup_time = await self._server.start()
        logger.info(f"SGLang server started in {startup_time:.2f}s")

        return self._server.config.host, self._server.config.port

    async def vllm_engine_is_sleeping(self) -> bool:
        """Compatibility: check if the inference engine is stopped."""
        return self._is_sleeping

    def get_openai_client(self) -> AsyncOpenAI:
        """Get an OpenAI client pointing at the SGLang server."""
        if self._server is None:
            raise RuntimeError("Server not started. Call start_openai_server() first.")
        return AsyncOpenAI(
            base_url=self._server.openai_base_url,
            api_key="sglang-benchmark",  # SGLang doesn't require a real key
        )

    # ------------------------------------------------------------------
    # Megatron process management (identical to MegatronService)
    # ------------------------------------------------------------------

    async def _ensure_megatron_running(self) -> None:
        """Lazily start Megatron training process if not running."""
        if self._megatron_process is not None:
            if self._megatron_process.returncode is None:
                return
            self._megatron_process = None

        try:
            import megatron.bridge  # type: ignore
            setup_cmd = ""
        except ImportError:
            setup_script = Path(__file__).parent.parent.parent / "src" / "art" / "megatron" / "setup.sh"
            setup_cmd = f"bash {setup_script} && "

        subprocess.run(["pkill", "-9", "megatron-service"], check=False)

        train_script = Path(__file__).parent.parent.parent / "src" / "art" / "megatron" / "train.py"
        num_gpus = torch.cuda.device_count()
        os.environ["MODEL_IDENTIFIER"] = self.base_model

        command = f"{setup_cmd}uv run torchrun --nproc_per_node {num_gpus} {train_script}"
        self._megatron_process = await asyncio.create_subprocess_shell(command)

    # ------------------------------------------------------------------
    # Training step (mirrors MegatronService.train())
    # ------------------------------------------------------------------

    async def train(
        self,
        disk_packed_tensors: dict,
        config: dict,
        experimental_config: dict,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """
        Run a single training step:
        1. Stop SGLang server (free GPU memory)
        2. Run Megatron training
        3. Merge LoRA shards
        4. Restart SGLang with new LoRA

        Yields training metrics dicts (loss, grad_norm, etc.) as they arrive.
        """
        # --- Phase 1: Stop SGLang server ---
        with TimingContext() as stop_timer:
            if self._server is not None and self._server.is_running:
                # Flush cache before stopping
                await self._server.flush_cache()
                await self._server.stop()
            self._is_sleeping = True

        self._last_server_stop_time = stop_timer.elapsed
        logger.info(f"SGLang server stopped in {stop_timer.elapsed:.2f}s")

        # --- Phase 2: Start Megatron training ---
        training_start = time.perf_counter()
        await self._ensure_megatron_running()

        lora_path = self._get_last_checkpoint_dir()
        if lora_path is None:
            lora_path = self._get_checkpoint_dir(0)
        self._ensure_lora_adapter_config(lora_path)

        self._optimizer_state_path = self._get_optimizer_state_path()

        # Write training job (same format as MegatronService)
        jobs_dir = "/tmp/megatron_training_jobs"
        os.makedirs(jobs_dir, exist_ok=True)
        for job_name in os.listdir(jobs_dir):
            if job_name.endswith(".json"):
                os.remove(os.path.join(jobs_dir, job_name))

        job = SGLangMegatronTrainingJob(
            lora_path=lora_path,
            optimizer_state_path=self._optimizer_state_path,
            disk_packed_tensors=disk_packed_tensors,
            config=config,
            experimental_config=experimental_config,
        )
        job_path = os.path.join(jobs_dir, f"{datetime.datetime.now().isoformat()}.json")
        with open(job_path, "w") as f:
            f.write(job.model_dump_json())

        # Monitor training log
        num_lines = 0
        while True:
            await asyncio.sleep(0.1)
            try:
                with open("/tmp/megatron_training_log.jsonl", "a+") as log_file:
                    log_file.seek(0)
                    lines = log_file.readlines()[num_lines:]
                    for line in lines:
                        if line := line.strip():
                            if line == "all done":
                                # --- Phase 3: Merge LoRA shards ---
                                with TimingContext() as merge_timer:
                                    self._merge_lora_adapter(lora_path)
                                self._last_lora_merge_time = merge_timer.elapsed
                                os.remove("/tmp/megatron_training_log.jsonl")
                                break
                            num_lines += 1
                            yield json.loads(line)
                    else:
                        continue
                    break
            except FileNotFoundError:
                continue

        self._last_training_time = time.perf_counter() - training_start

        # --- Phase 4: Create new checkpoint ---
        next_step = self._latest_step + 1
        new_checkpoint_dir = self._get_checkpoint_dir(next_step)
        os.makedirs(new_checkpoint_dir, exist_ok=True)
        shutil.copy(
            f"{lora_path}/adapter_model.safetensors",
            f"{new_checkpoint_dir}/adapter_model.safetensors",
        )
        self._ensure_lora_adapter_config(new_checkpoint_dir, source_path=lora_path)
        self._latest_step = next_step

        # --- Phase 5: Restart SGLang with new LoRA ---
        with TimingContext() as start_timer:
            self._server = self._create_server(new_checkpoint_dir)
            await self._server.start()
            self._is_sleeping = False

        self._last_server_start_time = start_timer.elapsed
        logger.info(f"SGLang server restarted in {start_timer.elapsed:.2f}s")

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

    # ------------------------------------------------------------------
    # Metrics access
    # ------------------------------------------------------------------

    def get_last_step_timings(self) -> dict[str, float]:
        """Return timing breakdown from the last training step."""
        return {
            "server_stop_time": self._last_server_stop_time,
            "training_time": self._last_training_time,
            "lora_merge_time": self._last_lora_merge_time,
            "server_start_time": self._last_server_start_time,
            "total_transition_overhead": (
                self._last_server_stop_time + self._last_server_start_time
            ),
        }
