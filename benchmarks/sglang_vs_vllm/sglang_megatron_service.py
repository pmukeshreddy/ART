"""
SGLang + Megatron service — drop-in for MegatronService.

Uses SGLang for inference, Megatron for training.
Server stop/restart to free GPU memory (vs vLLM's sleep/wake).
Same Megatron train.py for both backends.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from pydantic import BaseModel
import torch

from .sglang_server import SGLangServer, SGLangServerConfig

logger = logging.getLogger(__name__)


class SGLangMegatronTrainingJob(BaseModel):
    """Job format for Megatron train.py — MUST stay in sync."""
    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: dict
    config: dict
    experimental_config: dict


@dataclass
class SGLangMegatronService:
    """SGLang inference + Megatron training lifecycle."""

    model_name: str
    base_model: str
    output_dir: str
    sglang_python: str = "python"
    port: int = 8200
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.7
    max_running_requests: int = 256
    log_dir: str = ""

    _server: SGLangServer | None = None
    _latest_step: int = 0
    _megatron_process: asyncio.subprocess.Process | None = None
    _optimizer_state_path: str | None = None
    _is_sleeping: bool = False

    def __post_init__(self) -> None:
        if not self.log_dir:
            self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Server management
    # ------------------------------------------------------------------

    def _create_server(self, lora_path: str | None = None) -> SGLangServer:
        """Create SGLang server with correct config field names."""
        lora_paths: list[str] = []
        if lora_path and os.path.isdir(lora_path):
            adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
            if os.path.exists(adapter_file):
                # SGLang --lora-paths format: "name=path"
                lora_name = f"{self.model_name}@{self._latest_step}"
                lora_paths.append(f"{lora_name}={lora_path}")

        return SGLangServer(SGLangServerConfig(
            model_path=self.base_model,
            # served_model_name must match what ART uses for inference
            served_model_name=self.base_model,
            port=self.port,
            host="0.0.0.0",
            tensor_parallel_size=self.tensor_parallel_size,
            mem_fraction_static=self.gpu_memory_utilization,
            max_running_requests=self.max_running_requests,
            python_executable=self.sglang_python,
            log_file=os.path.join(self.log_dir, "sglang.log"),
            lora_paths=lora_paths,
            trust_remote_code=True,
            enable_p2p_check=True,
            chunked_prefill_size=32768,
        ))

    async def start_openai_server(
        self, config: Any = None
    ) -> tuple[str, int]:
        """Start SGLang server. Returns (host, port)."""
        # Don't create identity LoRA (loads full 30B model, wastes time).
        # Just start without LoRA — first step trains from base model.
        self._latest_step = 0

        # Check for existing checkpoint
        last_ckpt = self._get_last_checkpoint_dir()
        if last_ckpt:
            self._latest_step = int(os.path.basename(last_ckpt))

        self._server = self._create_server(last_ckpt)
        await self._server.start()
        logger.info(f"SGLang ready — serving {self.base_model} on port {self.port}")

        return "0.0.0.0", self.port

    async def vllm_engine_is_sleeping(self) -> bool:
        return self._is_sleeping

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _get_checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.output_dir, "checkpoints", f"{step:04d}")

    def _get_last_checkpoint_dir(self) -> str | None:
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            return None
        steps = sorted(
            int(d) for d in os.listdir(ckpt_dir)
            if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit()
        )
        return os.path.join(ckpt_dir, f"{steps[-1]:04d}") if steps else None

    def _get_optimizer_state_path(self) -> str:
        if self._optimizer_state_path is None:
            self._optimizer_state_path = os.path.join(self.output_dir, "optimizer_states")
            os.makedirs(self._optimizer_state_path, exist_ok=True)
        return self._optimizer_state_path

    # ------------------------------------------------------------------
    # Megatron process management
    # ------------------------------------------------------------------

    async def _ensure_megatron_running(self) -> None:
        if self._megatron_process is not None:
            if self._megatron_process.returncode is None:
                return
            self._megatron_process = None

        try:
            import megatron.bridge
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
    # Training step
    # ------------------------------------------------------------------

    async def train(
        self,
        disk_packed_tensors: dict,
        config: dict,
        experimental_config: dict,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Stop SGLang → Megatron train → merge LoRA → restart SGLang."""

        # Phase 1: Stop SGLang
        t0 = time.perf_counter()
        if self._server and self._server.is_running:
            await self._server.flush_cache()
            await self._server.stop()
        self._is_sleeping = True
        logger.info(f"SGLang stopped in {time.perf_counter()-t0:.1f}s")

        # Phase 2: Megatron training
        await self._ensure_megatron_running()

        lora_path = self._get_last_checkpoint_dir()
        if lora_path is None:
            lora_path = self._get_checkpoint_dir(0)
            os.makedirs(lora_path, exist_ok=True)

        jobs_dir = "/tmp/megatron_training_jobs"
        os.makedirs(jobs_dir, exist_ok=True)
        for f in os.listdir(jobs_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(jobs_dir, f))

        job = SGLangMegatronTrainingJob(
            lora_path=lora_path,
            optimizer_state_path=self._get_optimizer_state_path(),
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
                with open("/tmp/megatron_training_log.jsonl", "a+") as lf:
                    lf.seek(0)
                    lines = lf.readlines()[num_lines:]
                    for line in lines:
                        if line := line.strip():
                            if line == "all done":
                                self._merge_lora_adapter(lora_path)
                                os.remove("/tmp/megatron_training_log.jsonl")
                                break
                            num_lines += 1
                            yield json.loads(line)
                    else:
                        continue
                    break
            except FileNotFoundError:
                continue

        # Phase 3: New checkpoint
        next_step = self._latest_step + 1
        new_ckpt = self._get_checkpoint_dir(next_step)
        os.makedirs(new_ckpt, exist_ok=True)
        adapter_src = os.path.join(lora_path, "adapter_model.safetensors")
        if os.path.exists(adapter_src):
            shutil.copy(adapter_src, os.path.join(new_ckpt, "adapter_model.safetensors"))
        # Copy adapter config
        for cfg_name in ["adapter_config.json"]:
            src = os.path.join(lora_path, cfg_name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(new_ckpt, cfg_name))
        self._latest_step = next_step

        # Phase 4: Restart SGLang with new LoRA
        t0 = time.perf_counter()
        self._server = self._create_server(new_ckpt)
        await self._server.start()
        self._is_sleeping = False
        logger.info(f"SGLang restarted in {time.perf_counter()-t0:.1f}s with step {next_step}")

    def _merge_lora_adapter(self, lora_path: str) -> None:
        """Merge sharded LoRA adapters from distributed training."""
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        base_dir = Path(lora_path)
        shards = sorted(base_dir.glob("adapter_model-*-of-*.safetensors"))
        if not shards:
            return

        adapter_path = base_dir / "adapter_model.safetensors"
        sharded: dict[str, list[torch.Tensor]] = {}
        for fn in shards:
            with safe_open(fn, framework="pt") as f:
                for k in f.keys():
                    sharded.setdefault(k, []).append(f.get_tensor(k))

        merged: dict[str, torch.Tensor] = {}
        if adapter_path.exists():
            merged = load_file(adapter_path)
        for k, tensors in sharded.items():
            merged[k] = torch.cat(tensors, dim=1 if "lora_A" in k else 0)

        save_file(merged, adapter_path)
        for fn in shards:
            fn.unlink()
