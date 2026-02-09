"""
SGLang + Megatron service — verl-style hybrid engine.

Architecture:
  - SGLang server starts ONCE and NEVER restarts
  - Before training: sleep(kv_cache+weights) releases GPU memory
  - Training: Megatron runs as SEPARATE subprocess (produces LoRA adapter)
  - After training: merge LoRA → save to disk → /update_weights reload
  - After weight sync: wake_up(kv_cache) reallocates KV cache + flushes radix

Key difference from verl: our Megatron subprocess is separate (not shared
process), so we must release BOTH kv_cache and weights during sleep to give
Megatron enough GPU memory. verl's colocated design only releases kv_cache
because training shares the same model weights in GPU memory.

Weight sync: CUDA IPC (verl's sgl_update_weights) requires in-process
SGLang Python API, not available over HTTP. We use disk-based reload instead.

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
      NEW: sleep(kv+weights) → train → update_weights(disk) → wake(kv)

    Key difference from verl:
      verl: training + inference share same process, same GPU memory
      ours: Megatron is a SEPARATE subprocess, needs its own GPU memory
      → we must release weights too, not just KV cache

    Loop:
      1. generate()      — SGLang active, KV cache + weights on GPU
      2. sleep()         — release KV cache AND weights (for Megatron)
      3. Megatron train  — uses freed GPU memory
      4. update_weights()— merge LoRA, save to disk, SGLang reloads
      5. wake_up()       — reallocate KV cache only (weights already loaded)
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
        # Pre-warm HF cache path in background (overlaps with first rollout)
        import threading
        def _prewarm():
            try:
                from huggingface_hub import snapshot_download
                path = snapshot_download(
                    self.base_model,
                    allow_patterns=["*.safetensors", "*.json"],
                )
                logger.info(f"Pre-warmed HF cache: {path}")
            except Exception as e:
                logger.warning(f"HF cache pre-warm failed: {e}")
        threading.Thread(target=_prewarm, daemon=True).start()
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

        Releases BOTH KV cache AND model weights from GPU. This is critical
        because Megatron runs as a SEPARATE subprocess and needs GPU memory
        for its own copy of model weights, optimizer states, and activations.

        Unlike verl (where training and inference share the same process and
        same GPU memory), our architecture runs them in separate processes on
        the same GPUs. Only releasing KV cache (~35GB) is not enough — SGLang's
        model weights (~7.5GB/GPU for Qwen3-30B-A3B) must also be freed.

        SGLang process stays alive: NCCL communicators, tokenizer survive.
        Weights will be reloaded via update_weights_from_disk() after training.

        verl equivalent:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
            await tokenizer_manager.release_memory_occupation(obj, None)
        """
        if self._server is None or not self._server.is_running:
            return 0.0

        t0 = time.perf_counter()
        # Release both KV cache and weights — Megatron needs the GPU memory
        elapsed = await self._server.sleep(tags=["kv_cache", "weights"])
        self._is_sleeping = True
        logger.info(f"SGLang sleeping (kv_cache + weights released) in {elapsed:.2f}s")
        return time.perf_counter() - t0

    async def wake_up(self) -> float:
        """Resume KV cache after training + weight sync — verl's wake_up().

        Only reallocates KV cache. Model weights are already loaded by
        update_weights_from_disk() which was called before wake_up().

        verl equivalent:
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache"])
            await tokenizer_manager.resume_memory_occupation(obj, None)
            await tokenizer_manager.flush_cache()
        """
        if self._server is None or not self._server.is_running:
            return 0.0

        t0 = time.perf_counter()
        # Only resume KV cache — weights already loaded by update_weights
        elapsed = await self._server.wake_up(tags=["kv_cache"])
        self._is_sleeping = False
        logger.info(f"SGLang awake (kv_cache resumed) in {elapsed:.2f}s")
        return time.perf_counter() - t0

    # ------------------------------------------------------------------
    # verl-style weight sync
    # Mirrors: verl/workers/rollout/sglang_rollout/sglang_rollout.py
    # ------------------------------------------------------------------

    async def update_weights(self, lora_path: str) -> float:
        """Merge LoRA into base weights, save to disk, tell SGLang to reload.

        SGLang's /update_weights_from_tensor requires in-process Python API
        (sgl_update_weights) passing actual tensor objects — it does NOT accept
        serialized CUDA IPC handles over HTTP. Since we run SGLang as a
        subprocess (HTTP server), we must use the disk-based path:

          1. Load LoRA adapter from checkpoint
          2. Merge LoRA deltas into base weights (only affected layers)
          3. Save merged weights to a temp directory
          4. POST /update_weights with the merged model path
          5. Flush cache

        This still avoids full server restart — SGLang reloads only the weight
        tensors while keeping CUDA graphs, NCCL, tokenizer alive.
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

        # Step 3: Save merged weights to disk for SGLang to reload
        merged_model_path = await self._save_merged_weights(merged_params)

        # Step 4: Tell SGLang to reload weights from the merged path
        if self._server and merged_model_path:
            elapsed = await self._server.update_weights_from_disk(
                model_path=merged_model_path,
                load_format="auto",
            )
            logger.info(
                f"Weight sync: {len(merged_params)} params via disk reload "
                f"in {elapsed:.2f}s"
            )

        # Step 5: Flush cache after weight update (verl does this)
        if self._server:
            await self._server.flush_cache()

        total = time.perf_counter() - t0
        logger.info(f"Total weight sync: {total:.2f}s")
        return total

    async def _save_merged_weights(
        self, merged_params: list[tuple[str, torch.Tensor]]
    ) -> str:
        """Save merged weights to a directory in safetensors format.

        Copies the full base model structure (config, tokenizer, etc.) and
        overwrites only the LoRA-modified weight shards. SGLang's /update_weights
        expects a complete model directory.

        Uses symlinks for unaffected shards to avoid copying ~16GB. Cleans the merged directory on each call
        to prevent stale data from previous steps.
        """
        from safetensors.torch import save_file
        from huggingface_hub import snapshot_download

        merged_dir = os.path.join(self.output_dir, "merged_weights")

        # Clean previous merged weights to prevent stale data
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
        os.makedirs(merged_dir, exist_ok=True)

        # Get base model path (uses HF cache, no re-download if cached)
        base_path = snapshot_download(
            self.base_model,
            allow_patterns=["*.safetensors", "*.json"],
        )

        # Copy config/tokenizer files so SGLang can load the directory
        for fname in os.listdir(base_path):
            if fname.endswith(".json") or fname.endswith(".py") or fname == "tokenizer.model":
                src = os.path.join(base_path, fname)
                dst = os.path.join(merged_dir, fname)
                shutil.copy2(src, dst)

        # Load the weight index to know which shard each weight lives in
        index_path = os.path.join(base_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
        else:
            weight_map = {}

        # Build a dict of merged param names → tensors
        merged_dict = {name: tensor for name, tensor in merged_params}

        # Figure out which shard files are affected
        affected_shards: dict[str, set[str]] = {}
        for name in merged_dict:
            shard_file = weight_map.get(name)
            if shard_file:
                affected_shards.setdefault(shard_file, set()).add(name)

        # For each affected shard: load original, overwrite changed tensors, save
        from safetensors import safe_open

        for shard_file, changed_keys in affected_shards.items():
            src_path = os.path.join(base_path, shard_file)
            dst_path = os.path.join(merged_dir, shard_file)
            shard_tensors: dict[str, torch.Tensor] = {}
            with safe_open(src_path, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    if k in merged_dict:
                        shard_tensors[k] = merged_dict[k].cpu()
                    else:
                        shard_tensors[k] = sf.get_tensor(k)
            save_file(shard_tensors, dst_path)
            logger.info(f"Saved merged shard {shard_file} ({len(changed_keys)} updated keys)")

        # Symlink unaffected shard files (same filesystem, avoids 16GB copy)
        for fname in os.listdir(base_path):
            if fname.endswith(".safetensors") and fname not in affected_shards:
                src = os.path.realpath(os.path.join(base_path, fname))
                dst = os.path.join(merged_dir, fname)
                os.symlink(src, dst)

        # Copy index file
        if os.path.exists(index_path):
            shutil.copy2(index_path, os.path.join(merged_dir, "model.safetensors.index.json"))

        logger.info(f"Merged model saved to {merged_dir}")
        return merged_dir

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

        # BUG 6 fix: Build the set of needed base keys BEFORE loading weights,
        # so _load_base_weights only loads the shards containing these keys.
        needed_base_keys = {k for k, p in lora_pairs.items() if "A" in p and "B" in p}

        # Ensure base weights are loaded (cached, only LoRA-targeted shards)
        if self._base_weights is None:
            self._base_weights = self._load_base_weights(needed_keys=needed_base_keys)

        if not self._base_weights:
            logger.warning("Could not load base model weights for merging")
            return []

        for base_key, pair in lora_pairs.items():
            if "A" not in pair or "B" not in pair:
                continue

            if base_key in self._base_weights:
                W_base = self._base_weights[base_key]
                A = pair["A"].to(W_base.device, dtype=W_base.dtype)
                B = pair["B"].to(W_base.device, dtype=W_base.dtype)
                delta = B @ A
                # Handle TP shard mismatch: if LoRA shards weren't fully
                # concatenated (e.g. dim0 is half), tile to match base shape
                if delta.shape != W_base.shape:
                    if delta.shape[1] == W_base.shape[1] and W_base.shape[0] % delta.shape[0] == 0:
                        tp = W_base.shape[0] // delta.shape[0]
                        delta = delta.repeat(tp, 1)
                        logger.debug(f"Tiled {base_key} dim0 x{tp} to match base shape")
                    elif delta.shape[0] == W_base.shape[0] and W_base.shape[1] % delta.shape[1] == 0:
                        tp = W_base.shape[1] // delta.shape[1]
                        delta = delta.repeat(1, tp)
                        logger.debug(f"Tiled {base_key} dim1 x{tp} to match base shape")
                    else:
                        logger.debug(f"Shape mismatch for {base_key}: base={W_base.shape}, delta={delta.shape}, skipping")
                        continue
                W_merged = W_base + scaling * delta
                merged.append((base_key, W_merged))
            else:
                logger.debug(f"Base weight not found for {base_key}, skipping")

        logger.info(f"Merged {len(merged)} LoRA layers (scaling={scaling})")
        return merged

    def _load_base_weights(
        self, needed_keys: set[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Load base model weights from HuggingFace cache.

        Only loads the shard files that contain weights targeted by LoRA,
        avoiding loading the full 30GB model into CPU RAM.

        Args:
            needed_keys: Set of base weight names that LoRA targets.
                         If None, loads ALL weights (slow, avoid).
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
                for fn in os.listdir(model_path):
                    if fn.endswith(".safetensors"):
                        with safe_open(os.path.join(model_path, fn), framework="pt") as sf:
                            for k in sf.keys():
                                weight_map[k] = fn

            # Determine which shard files we actually need
            if needed_keys is not None:
                needed_shards: set[str] = set()
                for key in needed_keys:
                    shard = weight_map.get(key)
                    if shard:
                        needed_shards.add(shard)
                logger.info(
                    f"LoRA targets {len(needed_keys)} base weights across "
                    f"{len(needed_shards)} shard files (skipping the rest)"
                )
            else:
                needed_shards = set(weight_map.values())

            # Load only the needed shards, and only the needed keys from each
            weights: dict[str, torch.Tensor] = {}
            for shard_file in needed_shards:
                full_path = os.path.join(model_path, shard_file)
                with safe_open(full_path, framework="pt", device="cpu") as sf:
                    for k in sf.keys():
                        if needed_keys is None or k in needed_keys:
                            weights[k] = sf.get_tensor(k)

            logger.info(f"Loaded {len(weights)} base model weights for LoRA merging")
            return weights

        except Exception as e:
            logger.warning(f"Failed to load base weights: {e}")
            return {}

    # _weight_sync_disk_fallback removed — update_weights() now always uses
    # the disk-based path since CUDA IPC requires in-process SGLang API.

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
        """verl-style training step: sleep → train → update_weights → wake.

        OLD architecture:
          stop SGLang (5-10s) → train → restart SGLang (60-90s)
          Total overhead: 65-100s

        NEW architecture (verl-style):
          sleep(kv_cache+weights) → train → update_weights(disk) → wake(kv_cache)
          Total overhead: ~10-20s (mostly disk I/O for weight reload)

        Key: sleep releases BOTH kv_cache and weights because Megatron runs
        as a separate subprocess and needs GPU memory. update_weights reloads
        the merged weights BEFORE wake_up re-allocates KV cache.
        """

        # Phase 1: Sleep — release KV cache + model weights from GPU
        # Megatron subprocess needs the GPU memory for training
        t0 = time.perf_counter()
        sleep_time = await self.sleep()
        logger.info(f"Phase 1 — sleep(kv_cache+weights): {sleep_time:.2f}s")

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
            config=config if isinstance(config, dict) else config.model_dump(),
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

        # Phase 4: Weight sync — try to reload merged weights from disk
        # If merge fails (e.g. TP shape mismatch), fall back to original weights
        weight_sync_ok = False
        try:
            t_ws = time.perf_counter()
            weight_sync_time = await self.update_weights(new_ckpt)
            logger.info(
                f"Phase 4 — update_weights (disk reload): {weight_sync_time:.2f}s"
            )
            weight_sync_ok = True
        except Exception as e:
            logger.warning(f"Phase 4 — weight sync failed: {e}, waking with original weights")

        # Phase 5: Wake up — reallocate KV cache (+ weights if sync failed)
        if weight_sync_ok:
            wake_time = await self.wake_up()
        else:
            # Fallback: wake with both kv_cache + weights (original from CPU offload)
            t0_wake = time.perf_counter()
            if self._server:
                await self._server.wake_up(tags=["kv_cache", "weights"])
            self._is_sleeping = False
            wake_time = time.perf_counter() - t0_wake
            logger.info(f"SGLang awake (fallback: kv_cache + weights) in {wake_time:.2f}s")
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
            merged[k] = torch.cat(tensors, dim=self._shard_cat_dim(k))

        save_file(merged, adapter_path)
        for fn in shards:
            fn.unlink()

    @staticmethod
    def _shard_cat_dim(key: str) -> int:
        """Determine the correct concat dimension for TP-sharded LoRA weights.

        In Megatron/Transformer TP sharding:
          - Column-parallel layers (gate_proj, up_proj, q_proj, k_proj, v_proj):
              base weight sharded on dim=0 → lora_A on dim=0, lora_B on dim=0
          - Row-parallel layers (down_proj, o_proj):
              base weight sharded on dim=1 → lora_A on dim=1, lora_B on dim=0

        For LoRA:
          - lora_A has shape (r, in_features) or shard thereof
          - lora_B has shape (out_features, r) or shard thereof

        The naive "lora_A → dim=1, lora_B → dim=0" heuristic fails for
        row-parallel layers where lora_A is sharded on dim=1 (input dim)
        but lora_B is NOT sharded (it's the small r-dimension output).
        """
        # Row-parallel layers: down_proj, o_proj (and MoE shared_expert variants)
        is_row_parallel = any(rp in key for rp in ["down_proj", "o_proj"])

        if "lora_A" in key:
            # lora_A shape: (r, in_features)
            # Column-parallel: in_features is NOT sharded → no concat needed (dim=0 is r)
            # Row-parallel: in_features IS sharded on dim=1 → concat on dim=1
            return 1 if is_row_parallel else 0
        else:
            # lora_B shape: (out_features, r)
            # Column-parallel: out_features IS sharded → concat on dim=0
            # Row-parallel: out_features is NOT sharded → no concat (dim=0 safe)
            return 0

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
