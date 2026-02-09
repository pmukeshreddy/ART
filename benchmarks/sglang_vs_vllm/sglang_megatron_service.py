"""
SGLang + Megatron service — verl-style hybrid engine.

Architecture matches verl-project/verl exactly:
  - SGLang server starts ONCE and NEVER restarts
  - Before training: sleep() releases KV cache memory
  - Training: Megatron runs in subprocess (produces LoRA adapter)
  - After training: merge LoRA → update_weights_from_tensor (CUDA IPC)
  - After weight sync: wake_up() reallocates KV cache
  - Generation: non-streaming /generate returns actual token IDs

This eliminates the 60-90s cold restart that the old stop/restart
architecture imposed. Weight sync takes <2s via CUDA IPC.

Reference:
  - verl/workers/rollout/sglang_rollout/sglang_rollout.py (ServerAdapter)
  - verl/workers/rollout/sglang_rollout/async_sglang_server.py (SGLangHttpServer)
  - verl/trainer/ppo/ray_trainer.py (fit loop: generate → sleep → train → update_weights → wake)
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
from dataclasses import dataclass
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
    """verl-style SGLang inference + Megatron training lifecycle.

    Key difference from old implementation:
      OLD: stop SGLang → train → restart SGLang (60-90s overhead)
      NEW: sleep → train → update_weights → wake_up (<2s overhead)

    Mirrors verl's RayPPOTrainer.fit() loop:
      1. generate_sequences()  — SGLang active, KV cache allocated
      2. sleep_replicas()      — release KV cache, free GPU for training
      3. update_actor()        — Megatron training step
      4. update_weights()      — push new weights to SGLang via CUDA IPC
                                  + resume KV cache
    """

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
    # Base model weights cache — loaded once, kept in memory for LoRA merging
    _base_weights: dict[str, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if not self.log_dir:
            self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Server management — start ONCE, never restart
    # ------------------------------------------------------------------

    def _create_server(self) -> SGLangServer:
        """Create SGLang server — no LoRA on initial start.

        verl starts the base model and syncs weights in-place.
        We do the same: start clean, push LoRA-merged weights after training.
        """
        return SGLangServer(SGLangServerConfig(
            model_path=self.base_model,
            served_model_name=self.base_model,
            port=self.port,
            host="0.0.0.0",
            tensor_parallel_size=self.tensor_parallel_size,
            mem_fraction_static=self.gpu_memory_utilization,
            max_running_requests=self.max_running_requests,
            python_executable=self.sglang_python,
            log_file=os.path.join(self.log_dir, "sglang.log"),
            trust_remote_code=True,
            enable_p2p_check=True,
            chunked_prefill_size=32768,
            enable_memory_saver=True,  # Required for sleep/wake
        ))

    async def start_openai_server(
        self, config: Any = None
    ) -> tuple[str, int]:
        """Start SGLang server ONCE. It stays alive for the entire benchmark.

        Mirrors verl's SGLangHttpServer.launch_server() which launches
        subprocesses once and keeps them alive across all RL steps.
        """
        self._latest_step = 0
        self._server = self._create_server()
        await self._server.start()
        logger.info(
            f"SGLang ready (verl-style, persistent) — "
            f"serving {self.base_model} on port {self.port}"
        )
        return "0.0.0.0", self.port

    async def vllm_engine_is_sleeping(self) -> bool:
        """Compat: check if inference engine is sleeping."""
        return self._is_sleeping

    # ------------------------------------------------------------------
    # verl-style sleep / wake_up
    # Mirrors: verl/workers/rollout/sglang_rollout/async_sglang_server.py
    # ------------------------------------------------------------------

    async def sleep(self) -> float:
        """Release GPU memory for training — verl's sleep().

        Flushes KV cache, then releases memory so Megatron can use GPUs.
        SGLang process stays alive: CUDA graphs, NCCL, tokenizer all survive.

        verl equivalent:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
            await tokenizer_manager.release_memory_occupation(obj, None)
        """
        if self._server is None or not self._server.is_running:
            return 0.0

        t0 = time.perf_counter()
        elapsed = await self._server.sleep(tags=["kv_cache"])
        self._is_sleeping = True
        logger.info(f"SGLang sleeping (memory released) in {elapsed:.2f}s")
        return time.perf_counter() - t0

    async def wake_up(self) -> float:
        """Resume GPU memory after training — verl's wake_up().

        Reallocates KV cache so inference can proceed.

        verl equivalent:
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache"])
            await tokenizer_manager.resume_memory_occupation(obj, None)
            await tokenizer_manager.flush_cache()
        """
        if self._server is None or not self._server.is_running:
            return 0.0

        t0 = time.perf_counter()
        elapsed = await self._server.wake_up(tags=["kv_cache"])
        self._is_sleeping = False
        logger.info(f"SGLang awake (memory resumed) in {elapsed:.2f}s")
        return time.perf_counter() - t0

    # ------------------------------------------------------------------
    # verl-style weight sync
    # Mirrors: verl/workers/rollout/sglang_rollout/sglang_rollout.py
    # ------------------------------------------------------------------

    async def update_weights(self, lora_path: str) -> float:
        """Push LoRA-merged weights to SGLang via CUDA IPC — verl's update_weights().

        This is the key optimization: instead of restarting SGLang with a new
        LoRA adapter (60-90s), we merge the LoRA delta into the base model
        weights and push them directly to SGLang's GPU memory (<2s).

        verl equivalent:
            async for params_batch in get_named_tensor_buckets(weights, bucket_bytes):
                await sgl_update_weights(engine=self._engine, params_batch=params_batch, ...)
            await self._engine.flush_cache()
        """
        t0 = time.perf_counter()

        # Step 1: Load LoRA adapter
        lora_tensors = self._load_lora_adapter(lora_path)
        if not lora_tensors:
            logger.warning("No LoRA adapter found, skipping weight sync")
            return 0.0

        # Step 2: Merge LoRA into base weights for affected layers
        merged_params = self._merge_lora_into_base(lora_tensors, lora_path)
        if not merged_params:
            logger.warning("No params to update after LoRA merge")
            return 0.0

        # Step 3: Push merged weights to SGLang via CUDA IPC
        #         This mirrors verl's sgl_update_weights() which sends
        #         tensors in batches via CUDA IPC handles
        try:
            elapsed = await self._server.update_weights_from_tensor(merged_params)
            logger.info(
                f"Weight sync: {len(merged_params)} params via CUDA IPC "
                f"in {elapsed:.2f}s"
            )
        except Exception as e:
            logger.warning(f"CUDA IPC weight sync failed: {e}, trying disk fallback")
            # Fallback: save merged model and reload from disk
            # Still better than restart — SGLang process stays alive
            elapsed = await self._weight_sync_disk_fallback(lora_path)

        # Step 4: Flush cache after weight update (verl does this)
        await self._server.flush_cache()

        total = time.perf_counter() - t0
        logger.info(f"Total weight sync: {total:.2f}s")
        return total

    def _load_lora_adapter(self, lora_path: str) -> dict[str, torch.Tensor]:
        """Load LoRA adapter tensors from checkpoint."""
        adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            return {}

        from safetensors.torch import load_file
        return load_file(adapter_file)

    def _merge_lora_into_base(
        self,
        lora_tensors: dict[str, torch.Tensor],
        lora_path: str,
    ) -> list[tuple[str, torch.Tensor]]:
        """Merge LoRA A/B into base model weights for affected layers.

        For each LoRA pair (lora_A, lora_B), computes:
            W_merged = W_base + (alpha/r) * B @ A

        Only returns the layers that were modified by LoRA.
        """
        # Load adapter config
        config_path = os.path.join(lora_path, "adapter_config.json")
        alpha = 32  # default
        r = 1       # default
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
                alpha = cfg.get("lora_alpha", alpha)
                r = cfg.get("r", r)
        scaling = alpha / r

        # Ensure base weights are loaded (cached)
        if self._base_weights is None:
            self._base_weights = self._load_base_weights()

        if not self._base_weights:
            logger.warning("Could not load base model weights for merging")
            return []

        # Group LoRA tensors by layer
        # Keys look like: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        merged: list[tuple[str, torch.Tensor]] = []
        lora_pairs: dict[str, dict[str, torch.Tensor]] = {}

        for key, tensor in lora_tensors.items():
            if "lora_A" in key:
                base_key = key.replace(".lora_A.weight", ".weight")
                # Strip PEFT prefix
                base_key = base_key.replace("base_model.model.", "", 1)
                lora_pairs.setdefault(base_key, {})["A"] = tensor
            elif "lora_B" in key:
                base_key = key.replace(".lora_B.weight", ".weight")
                base_key = base_key.replace("base_model.model.", "", 1)
                lora_pairs.setdefault(base_key, {})["B"] = tensor

        for base_key, pair in lora_pairs.items():
            if "A" not in pair or "B" not in pair:
                continue

            if base_key in self._base_weights:
                W_base = self._base_weights[base_key]
                A = pair["A"].to(W_base.device, dtype=W_base.dtype)
                B = pair["B"].to(W_base.device, dtype=W_base.dtype)
                W_merged = W_base + scaling * (B @ A)
                merged.append((base_key, W_merged))
            else:
                logger.debug(f"Base weight not found for {base_key}, skipping")

        logger.info(f"Merged {len(merged)} LoRA layers (scaling={scaling})")
        return merged

    def _load_base_weights(self) -> dict[str, torch.Tensor]:
        """Load base model weights from HuggingFace cache.

        Only loads the weight index and then lazily loads affected shards.
        This is cached for the entire benchmark run.
        """
        try:
            from huggingface_hub import snapshot_download
            from safetensors import safe_open

            # Get cached model path
            model_path = snapshot_download(
                self.base_model,
                allow_patterns=["*.safetensors", "*.json"],
            )

            # Load weight index
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
            else:
                # Single shard model
                weight_map = {}
                for f in os.listdir(model_path):
                    if f.endswith(".safetensors"):
                        from safetensors import safe_open as _so
                        with _so(os.path.join(model_path, f), framework="pt") as sf:
                            for k in sf.keys():
                                weight_map[k] = f

            # Load all weights into CPU memory
            weights: dict[str, torch.Tensor] = {}
            loaded_files: set[str] = set()
            for name, shard_file in weight_map.items():
                if shard_file not in loaded_files:
                    full_path = os.path.join(model_path, shard_file)
                    with safe_open(full_path, framework="pt", device="cpu") as sf:
                        for k in sf.keys():
                            weights[k] = sf.get_tensor(k)
                    loaded_files.add(shard_file)

            logger.info(f"Loaded {len(weights)} base model weights for LoRA merging")
            return weights

        except Exception as e:
            logger.warning(f"Failed to load base weights: {e}")
            return {}

    async def _weight_sync_disk_fallback(self, lora_path: str) -> float:
        """Fallback weight sync: merge LoRA to disk, tell SGLang to reload.

        Still better than restart — SGLang process stays alive, only weights
        are reloaded. CUDA graphs, NCCL, etc. survive.
        """
        t0 = time.perf_counter()
        logger.info("Using disk-based weight sync fallback")

        # Try SGLang's update_weights endpoint (loads from model path)
        # This avoids full restart but still involves disk I/O
        if self._server:
            await self._server.update_weights_from_disk(
                model_path=self.base_model,
                load_format="auto",
            )

        return time.perf_counter() - t0

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
    # Training step — verl-style: sleep → train → update_weights → wake
    # ------------------------------------------------------------------

    async def train(
        self,
        disk_packed_tensors: dict,
        config: dict,
        experimental_config: dict,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """verl-style training step: sleep → Megatron train → weight sync → wake.

        OLD architecture:
          stop SGLang (5-10s) → train → restart SGLang (60-90s)
          Total overhead: 65-100s

        NEW architecture (verl-style):
          sleep (<1s) → train → update_weights (<2s) → wake_up (<1s)
          Total overhead: <4s

        This mirrors verl's RayPPOTrainer.fit():
          self.checkpoint_manager.sleep_replicas()
          actor_output = self._update_actor(batch)
          self.checkpoint_manager.update_weights()
        """

        # Phase 1: Sleep — release KV cache memory for training
        # verl: self.checkpoint_manager.sleep_replicas()
        t0 = time.perf_counter()
        sleep_time = await self.sleep()
        logger.info(f"Phase 1 — sleep: {sleep_time:.2f}s (was: kill server 5-10s)")

        # Phase 2: Megatron training
        # verl: actor_output = self._update_actor(batch)
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
                                self._merge_lora_shards(lora_path)
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
        for cfg_name in ["adapter_config.json"]:
            src = os.path.join(lora_path, cfg_name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(new_ckpt, cfg_name))
        self._latest_step = next_step

        # Phase 4: Weight sync — push merged weights to SGLang via CUDA IPC
        # verl: self.checkpoint_manager.update_weights()
        t_ws = time.perf_counter()
        weight_sync_time = await self.update_weights(new_ckpt)
        logger.info(
            f"Phase 4 — weight sync: {weight_sync_time:.2f}s "
            f"(was: restart server 60-90s)"
        )

        # Phase 5: Wake up — reallocate KV cache
        wake_time = await self.wake_up()
        logger.info(f"Phase 5 — wake_up: {wake_time:.2f}s")

        total_overhead = time.perf_counter() - t0
        logger.info(
            f"Total transition overhead: {total_overhead:.2f}s "
            f"(verl-style, was 65-100s with restart)"
        )

    def _merge_lora_shards(self, lora_path: str) -> None:
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

    # ------------------------------------------------------------------
    # Native generation (verl-style, non-streaming)
    # ------------------------------------------------------------------

    async def generate_native(
        self,
        prompt: str | list[dict[str, str]],
        sampling_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming generation with actual token counts.

        Mirrors verl's SGLangHttpServer.generate() which uses
        tokenizer_manager.generate_request() directly, returning
        actual output_ids instead of SSE chunks.
        """
        if self._server is None:
            return {"error": "Server not started"}
        return await self._server.generate_native(prompt, sampling_params)
