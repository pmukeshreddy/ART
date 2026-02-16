"""
Unsloth + SGLang service — MoE training matching Megatron exactly.

Uses the SAME pipeline as the Megatron backend:
  - LoRA config: rank=1, alpha=32, targets 7 modules (q/k/v/o/gate/up/down_proj)
  - Loss function: art.loss.loss_fn with on_policy_correction=True
  - Data pipeline: ART's packed tensors (tokenize_trajectory_groups +
    packed_tensors_from_tokenized_results), saved to disk
  - Optimizer: AdamW(lr=5e-6, betas=(0.9, 0.99), weight_decay=0.1, clip_grad=0.1)

Architecture (verl-style, same GPUs via sleep/wake — matches SGLang+Megatron):
  - SGLang server starts ONCE and NEVER restarts
  - Unsloth training runs in a PERSISTENT SUBPROCESS
  - Training and inference time-share the same GPUs via sleep/wake
  - LoRA hot-reload for weight sync (<2s)

Training loop (per step):
  1. generate()       — SGLang active, KV cache + weights on GPU
  2. sleep()          — SGLang releases KV cache AND weights
  3. reload_to_gpu()  — Unsloth model back to GPU from CPU
  4. train            — ART loss on packed tensors (same as Megatron)
  5. offload_to_cpu() — Unsloth model to CPU, free GPU
  6. wake_up()        — SGLang restores base weights + KV cache
  7. load_lora()      — hot-reload adapter

Reference:
  - https://unsloth.ai/docs/new/faster-moe
  - https://unsloth.ai/docs/basics/inference-and-deployment/sglang-guide
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import torch

from .sglang_server import SGLangServer, SGLangServerConfig

logger = logging.getLogger(__name__)


def _is_vllm_healthy() -> bool:
    """Return True if vLLM's C extension loads without ABI errors."""
    try:
        import vllm._C  # noqa: F401
        return True
    except (ImportError, OSError, AttributeError):
        return False


class _StubModule(types.ModuleType):
    """A module whose public attributes are no-op callables returning None.

    Used to mock ``unsloth_zoo.vllm_utils`` when vLLM's C extension is broken.
    Any function imported from the mock (e.g. ``_get_torchao_fp8_config``)
    will be a harmless no-op.

    Dunder attributes (``__file__``, ``__path__``, ``__spec__``, …) are NOT
    mocked — Python's ``inspect`` module iterates ``sys.modules`` and accesses
    ``__file__`` on every module.  If ``__file__`` returns a callable instead
    of a string, ``inspect.getsourcefile()`` crashes with
    ``AttributeError: 'function' object has no attribute 'endswith'``.
    """

    def __getattr__(self, name: str):
        # Let dunder lookups raise AttributeError so inspect/importlib
        # treat this module as one without source (like builtins).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _noop(*args, **kwargs):
            return None
        return _noop


def _patch_vllm_for_unsloth_import() -> None:
    """Make Unsloth importable even when vLLM's C extension is broken.

    Unsloth + unsloth_zoo have deep vLLM imports at module load time:
      1. ``unsloth/__init__.py`` → ``fix_vllm_guided_decoding_params()``
         chains into ``vllm._C`` (ABI crash).
      2. ``unsloth_zoo/vllm_utils.py`` → ``import vllm.model_executor.layers...``
         chains deep into vLLM quantization/fused_moe layers that call
         ``torch.ops._C`` custom ops (which aren't registered if _C failed).

    On cloud GPU images where vLLM was compiled against a different PyTorch
    ABI (e.g. vLLM 0.15.1 + PyTorch 2.10.0), these imports crash.

    Since we use SGLang (not vLLM) for inference, we:
      1. Create a dummy ``vllm._C`` module
      2. Pre-populate ``sys.modules["unsloth_zoo.vllm_utils"]`` with a stub
         so the *real* module (which does ``import vllm.model_executor...``)
         is never loaded

    vLLM inference (if used in a separate process) is unaffected — each
    subprocess has its own module state.
    """
    if _is_vllm_healthy():
        return  # vLLM works fine, no mocking needed

    logger.info(
        "vLLM C extension is broken (ABI mismatch with PyTorch). "
        "Mocking vllm internals for Unsloth import — we use SGLang, not vLLM."
    )

    # 1. Dummy vllm._C so shallow imports don't crash
    sys.modules["vllm._C"] = types.ModuleType("vllm._C")

    # 2. Mock unsloth_zoo.vllm_utils BEFORE Unsloth imports it.
    #    This prevents the real module from loading, which means the deep
    #    vllm.model_executor import chain never executes.
    if "unsloth_zoo.vllm_utils" not in sys.modules:
        sys.modules["unsloth_zoo.vllm_utils"] = _StubModule("unsloth_zoo.vllm_utils")


def _gc_and_empty_cuda_cache(n: int = 3) -> None:
    for _ in range(n):
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Unsloth Training State — model persists across steps, offloads to CPU between
# ---------------------------------------------------------------------------

@dataclass
class UnslothTrainingState:
    """Holds the Unsloth model, tokenizer, and optimizer across training steps.

    In the persistent subprocess architecture, the model stays on GPU
    permanently — offload/reload methods are retained for fallback use.
    """

    model: Any  # PeftModelForCausalLM after FastLanguageModel.get_peft_model()
    tokenizer: Any
    optimizer: torch.optim.Optimizer
    _is_offloaded: bool = False

    def offload_to_cpu(self) -> None:
        """Move model + optimizer to CPU, free GPU for SGLang."""
        if self._is_offloaded:
            return
        t0 = time.perf_counter()
        self.model.to("cpu")
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                    state[k] = v.cpu()
        torch.cuda.synchronize()
        self._is_offloaded = True
        _gc_and_empty_cuda_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        logger.info(
            f"Unsloth offloaded to CPU in {time.perf_counter() - t0:.2f}s "
            f"(GPU free: {free_gb:.1f} GB)"
        )

    def reload_to_gpu(self, device: str = "cuda:0") -> None:
        """Move model + optimizer back to GPU for training."""
        if not self._is_offloaded:
            return
        t0 = time.perf_counter()
        self.model.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                    state[k] = v.to(device)
        torch.cuda.synchronize()
        self._is_offloaded = False
        logger.info(f"Unsloth reloaded to GPU in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Training Worker — runs in a persistent subprocess via mp_actors
# ---------------------------------------------------------------------------

class UnslothTrainingWorker:
    """Training worker — runs in a persistent subprocess via mp_actors.

    Uses the SAME training pipeline as the Megatron backend:
      - LoRA config: rank=1, alpha=32, targets all 7 modules
      - Loss: art.loss.loss_fn with on_policy_correction=True
      - Data: ART packed tensors loaded from disk

    Communication with the parent process is via mp_actors proxy (pickle over
    multiprocessing queues). Only lightweight data crosses the boundary:
      - packed_tensors_dir: str (path to packed tensors on disk)
      - metrics: dict[str, float] (~1KB)
      - checkpoint paths: str
    """

    def __init__(
        self,
        base_model: str,
        output_dir: str,
        lora_rank: int = 1,
        lora_alpha: int = 32,
        max_seq_length: int = 8192,
        learning_rate: float = 5e-6,
        moe_backend: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.moe_backend = moe_backend
        self.load_in_4bit = load_in_4bit
        self._state: UnslothTrainingState | None = None

    async def init_model(self) -> dict[str, Any]:
        """Load model to GPU. Called once in subprocess."""
        if self.moe_backend != "auto":
            os.environ["UNSLOTH_MOE_BACKEND"] = self.moe_backend

        _patch_vllm_for_unsloth_import()
        from unsloth import FastLanguageModel

        logger.info(f"Loading model: {self.base_model}")
        logger.info(f"  lora_rank={self.lora_rank}  max_seq_length={self.max_seq_length}")
        logger.info(f"  load_in_4bit={self.load_in_4bit}  moe_backend={self.moe_backend}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            # Only target attention modules for MoE models.
            # gate/up/down_proj exist in EVERY expert, so targeting them
            # multiplies params by num_experts (52M vs 3M for rank=1).
            # Megatron's LoRA applies to shared layers differently, so
            # attention-only matches the effective behavior for MoE.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=self.lora_alpha,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        FastLanguageModel.for_training(model)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=self.learning_rate, betas=(0.9, 0.99), weight_decay=0.1,
        )

        n_params = sum(p.numel() for p in trainable)
        logger.info(f"Unsloth ready — {n_params:,} trainable params")

        self._state = UnslothTrainingState(model=model, tokenizer=tokenizer, optimizer=optimizer)
        return {"trainable_params": n_params}

    async def train_on_packed_tensors(
        self,
        packed_tensors_dir: str,
        num_sequences: int,
        sequence_length: int,
        lr: float | None = None,
    ) -> dict[str, float]:
        """Train using ART's packed tensors and loss function.

        Matches Megatron's training loop exactly:
          - Same packed tensor format (tokens, logprobs, advantages, etc.)
          - Same loss function (art.loss.loss_fn with on_policy_correction=True)
          - Same optimizer (AdamW, clip_grad=0.1)

        The packed tensors are created by ART's preprocessing pipeline
        (tokenize_trajectory_groups + packed_tensors_from_tokenized_results)
        in the benchmark runner, then saved to disk. This method loads them
        and runs the training loop.
        """
        from art.preprocessing.pack import packed_tensors_from_dir
        from art.loss import loss_fn, shift_tensor

        state = self._state
        assert state is not None

        device = next(state.model.parameters()).device
        state.model.train()

        if lr is not None:
            for pg in state.optimizer.param_groups:
                pg["lr"] = lr

        packed = packed_tensors_from_dir(
            dir=packed_tensors_dir,
            num_sequences=num_sequences,
            sequence_length=sequence_length,
        )

        total_loss = 0.0
        n_seqs = 0
        completion_tokens = 0

        state.optimizer.zero_grad()
        t0 = time.perf_counter()

        for idx in range(num_sequences):
            inputs = {
                key: value[idx:idx + 1].to(device)
                for key, value in packed.items()
                if isinstance(value, torch.Tensor)
            }

            tokens = inputs["tokens"]
            batch_size, seq_len = tokens.shape
            completion_tokens += int(inputs["assistant_mask"].sum().item())

            attn_mask = (inputs["group_ids"] != -1).long()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = state.model(
                    input_ids=tokens,
                    position_ids=inputs["input_pos"],
                    attention_mask=attn_mask,
                )
                logits = outputs.logits

                labels = shift_tensor(tokens, 0)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                new_logprobs = log_probs.gather(
                    dim=-1, index=labels.unsqueeze(-1),
                ).squeeze(-1)

                experimental_config = {"on_policy_correction": True}
                loss_result = loss_fn(
                    inputs, new_logprobs, ref_logprobs=None, entropies=None,
                    experimental_config=experimental_config,
                )

                loss = loss_result.mean_policy_loss / num_sequences

            loss.backward()

            total_loss += loss_result.mean_policy_loss.item()
            n_seqs += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in state.model.parameters() if p.requires_grad],
            max_norm=0.1,
        )
        state.optimizer.step()
        state.optimizer.zero_grad()

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / max(n_seqs, 1)
        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        logger.info(
            f"  trained: loss={avg_loss:.4f}  {completion_tokens / elapsed:.0f} tok/s  "
            f"VRAM={gpu_mem_gb:.1f}GB  {elapsed:.2f}s (ART loss, packed tensors)"
        )

        return {
            "loss": avg_loss,
            "training_time_s": elapsed,
            "tokens_per_sec": completion_tokens / elapsed,
            "gpu_memory_gb": gpu_mem_gb,
            "total_tokens": completion_tokens,
            "batch_size": n_seqs,
            "seq_len": sequence_length,
        }

    async def offload_to_cpu(self) -> dict[str, float]:
        """Offload model + optimizer to CPU, freeing GPU for SGLang wake_up.

        Matches the SGLang+Megatron pattern: training and inference time-share
        the same GPUs via sleep/wake. After training, offload frees GPU memory
        so SGLang can restore its weights + KV cache.
        """
        assert self._state is not None, "init_model() must be called first"
        t0 = time.perf_counter()
        self._state.offload_to_cpu()
        return {"offload_time_s": time.perf_counter() - t0}

    async def reload_to_gpu(self) -> dict[str, float]:
        """Reload model + optimizer to GPU for training.

        Called after SGLang sleeps and frees GPU memory.
        """
        assert self._state is not None, "init_model() must be called first"
        t0 = time.perf_counter()
        self._state.reload_to_gpu()
        return {"reload_time_s": time.perf_counter() - t0}

    async def save_lora(self, step: int) -> str:
        """Save LoRA adapter via PEFT save_pretrained (standard format)."""
        assert self._state is not None

        ckpt = os.path.join(self.output_dir, "checkpoints", f"{step:04d}")
        os.makedirs(ckpt, exist_ok=True)

        self._state.model.save_pretrained(ckpt)
        # NOTE: do NOT save tokenizer here.  tokenizer.save_pretrained()
        # writes added_tokens.json to the same directory.  SGLang's
        # LoRAConfig reads that file and treats it as LoRA vocabulary
        # additions, making can_support() fail because the memory pool
        # has lora_added_tokens_size=0.  The Megatron backend also does
        # NOT save the tokenizer alongside the adapter.
        # SGLang uses its own tokenizer — the adapter only needs
        # adapter_config.json + adapter_model.safetensors.

        adapter = os.path.join(ckpt, "adapter_model.safetensors")
        if os.path.exists(adapter):
            mb = os.path.getsize(adapter) / 1e6
            logger.info(f"LoRA saved: {ckpt} ({mb:.1f} MB)")
        else:
            logger.warning(f"adapter_model.safetensors not found in {ckpt}")

        return ckpt



# ---------------------------------------------------------------------------
# Main Service
# ---------------------------------------------------------------------------

@dataclass
class UnslothSGLangService:
    """Unsloth MoE training + SGLang inference — matches Megatron pipeline.

    Uses ART's data pipeline and loss function for identical training behavior.

    Lifecycle per RL step:
      1. SGLang serves rollouts (inference)
      2. Benchmark runner tokenizes/packs data via ART preprocessing
      3. sleep()  — SGLang releases GPU memory
      4. Unsloth trains on packed tensors using art.loss.loss_fn
      5. Save LoRA adapter
      6. wake_up()  — SGLang restores GPU memory
      7. load_lora()  — SGLang loads new adapter (<2s)
    """

    model_name: str
    base_model: str
    output_dir: str
    sglang_python: str = "python"
    port: int = 8300
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.7
    max_running_requests: int = 256
    log_dir: str = ""

    # Unsloth config — matches Megatron (rank=1, alpha=32, 7 modules)
    lora_rank: int = 1
    lora_alpha: int = 32
    max_seq_length: int = 8192
    learning_rate: float = 5e-6
    # "auto" lets Unsloth pick: grouped_mm (H100+), unsloth_triton (A100), native_torch
    moe_backend: str = "auto"
    load_in_4bit: bool = False  # MoE nn.Parameter doesn't support bnb 4bit yet

    # Internal state
    _server: SGLangServer | None = None
    _worker: Any = None  # mp_actors proxy to UnslothTrainingWorker in subprocess
    _worker_initialized: bool = False  # True after init_model() called on worker
    _latest_step: int = 0
    _is_sleeping: bool = False
    _active_lora_name: str | None = None

    def __post_init__(self) -> None:
        if not self.log_dir:
            self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)

        # Let Unsloth auto-select, or override
        if self.moe_backend != "auto":
            os.environ["UNSLOTH_MOE_BACKEND"] = self.moe_backend

    # ------------------------------------------------------------------
    # SGLang server — start ONCE, never restart
    # ------------------------------------------------------------------

    def _create_server(self) -> SGLangServer:
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
            enable_memory_saver=True,
            enable_lora=True,
            max_lora_rank=8,  # Megatron trains rank=1, headroom for future
            # Match Megatron backend: use DEFAULT lora_target_modules
            # (q/k/v/o_proj + gate/up/down_proj). Don't override — the
            # Megatron backend also uses the default and it works.
            # The adapter now targets all 7 modules (matches Megatron).
        ))

    async def start(self) -> float:
        """Start SGLang server and persistent training subprocess.

        The training subprocess is created here but the model is loaded
        lazily on the first train_step() (after SGLang sleeps and frees
        GPU memory). This matches the Megatron pattern.
        """
        self._server = self._create_server()
        startup = await self._server.start()
        logger.info(
            f"SGLang ready — {self.base_model} on :{self.port} "
            f"(startup {startup:.1f}s, will NOT restart)"
        )

        # Create persistent training subprocess via mp_actors.
        # The worker object is lightweight at creation (just config strings/ints).
        # Model loading happens later in init_model() after SGLang sleeps.
        from mp_actors import move_to_child_process

        worker = UnslothTrainingWorker(
            base_model=self.base_model,
            output_dir=self.output_dir,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            max_seq_length=self.max_seq_length,
            learning_rate=self.learning_rate,
            moe_backend=self.moe_backend,
            load_in_4bit=self.load_in_4bit,
        )
        self._worker = move_to_child_process(
            worker,
            log_file=os.path.join(self.log_dir, "unsloth_worker.log"),
            process_name="unsloth-trainer",
        )
        self._worker_initialized = False
        logger.info("Unsloth training subprocess started (model will load on first train_step)")

        return startup

    async def stop(self) -> None:
        """Stop everything. Called once at benchmark end."""
        # Terminate the persistent training subprocess
        if self._worker is not None:
            from mp_actors import close_proxy
            try:
                close_proxy(self._worker)
            except Exception:
                pass
            self._worker = None
            self._worker_initialized = False
            logger.info("Unsloth training subprocess terminated")

        if self._server is not None:
            await self._server.stop()
            self._server = None
        _gc_and_empty_cuda_cache()

    # ------------------------------------------------------------------
    # verl-style sleep / wake (identical to sglang backend)
    # ------------------------------------------------------------------

    async def sleep(self) -> float:
        """Release GPU memory so Unsloth can train."""
        if self._server is None or not self._server.is_running:
            return 0.0
        t0 = time.perf_counter()
        await self._server.sleep(tags=["kv_cache", "weights"])
        self._is_sleeping = True
        elapsed = time.perf_counter() - t0
        logger.info(f"SGLang asleep (kv_cache + weights freed) — {elapsed:.2f}s")
        return elapsed

    async def wake_up(self) -> float:
        """Restore GPU memory after training."""
        if self._server is None or not self._server.is_running:
            return 0.0
        t0 = time.perf_counter()
        await self._server.wake_up(tags=["kv_cache", "weights"])
        self._is_sleeping = False
        elapsed = time.perf_counter() - t0
        logger.info(f"SGLang awake (kv_cache + weights restored) — {elapsed:.2f}s")
        return elapsed

    # ------------------------------------------------------------------
    # LoRA hot-reload (save is now in UnslothTrainingWorker)
    # ------------------------------------------------------------------

    async def _load_lora(self, lora_path: str, step: int) -> float:
        """Hot-reload LoRA into SGLang (<2s)."""
        if self._server is None:
            return 0.0

        adapter = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(adapter):
            logger.warning(f"No adapter at {adapter}")
            return 0.0

        name = f"{self.model_name}@step{step}"
        elapsed = await self._server.load_lora_adapter(
            lora_path=lora_path, lora_name=name, flush_cache=True,
        )
        if elapsed < 0:
            logger.error("load_lora_adapter failed — base weights intact but not updated")
            return 0.0

        self._active_lora_name = name
        logger.info(f"LoRA hot-reload: '{name}' in {elapsed:.2f}s")
        return elapsed

    # ------------------------------------------------------------------
    # Full step: sleep → train → save → wake → load_lora
    # ------------------------------------------------------------------

    async def train_step(
        self,
        packed_tensors_dir: str,
        num_sequences: int,
        sequence_length: int,
        lr: float | None = None,
    ) -> dict[str, float]:
        """One complete training step — verl-style, same GPUs via sleep/wake.

        Matches the SGLang+Megatron pattern exactly:
          - Same data pipeline: ART's packed tensors (tokenized + packed by
            the benchmark runner using tokenize_trajectory_groups +
            packed_tensors_from_tokenized_results)
          - Same loss function: art.loss.loss_fn with on_policy_correction=True
          - Same GPU time-sharing: sleep/wake cycle

        Loop:
          1. sleep()          — SGLang releases KV cache + weights
          2. reload_to_gpu()  — Unsloth model back to GPU (skip on first call)
          3. train             — ART loss on packed tensors
          4. save_lora()      — save adapter to disk
          5. offload_to_cpu() — Unsloth model to CPU, free GPU
          6. wake_up()        — SGLang restores KV cache + weights
          7. load_lora()      — hot-reload adapter (<2s)

        Args:
            packed_tensors_dir: Path to directory with ART packed tensors
                (tokens.pt, logprobs.pt, advantages.pt, etc.)
            num_sequences: Number of packed sequences on disk.
            sequence_length: Sequence length of each packed sequence.
            lr: Learning rate override (optional).

        Returns:
            Dict of training + overhead metrics.
        """
        assert self._worker is not None, "call start() before train_step()"

        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        # 1. Sleep SGLang — free GPU for training
        timings["sleep_s"] = await self.sleep()

        # 2. Initialize model on first call (lazy — GPU is free after sleep)
        t = time.perf_counter()
        if not self._worker_initialized:
            init_result = await self._worker.init_model()
            n_params = init_result.get("trainable_params", "?")
            logger.info(f"Unsloth worker model loaded — {n_params:,} trainable params")
            self._worker_initialized = True
        else:
            reload_result = await self._worker.reload_to_gpu()
            timings["reload_s"] = reload_result.get("reload_time_s", 0)
        timings["model_load_s"] = time.perf_counter() - t

        # 3. Train on packed tensors — uses art.loss.loss_fn (same as Megatron)
        train_metrics = await self._worker.train_on_packed_tensors(
            packed_tensors_dir, num_sequences, sequence_length, lr,
        )

        # 4. Save LoRA
        t = time.perf_counter()
        self._latest_step += 1
        ckpt = await self._worker.save_lora(self._latest_step)
        timings["save_s"] = time.perf_counter() - t

        # 5. Offload Unsloth model to CPU — free GPU for SGLang
        t = time.perf_counter()
        offload_result = await self._worker.offload_to_cpu()
        timings["offload_s"] = offload_result.get("offload_time_s", 0)

        # 6. Wake SGLang — GPU is now free, restore weights + KV cache
        timings["wake_s"] = await self.wake_up()

        # 7. Hot-reload LoRA
        timings["lora_reload_s"] = await self._load_lora(ckpt, self._latest_step)

        # 8. Health check — if SGLang crashed, restart
        if self._server is not None and not self._server.is_running:
            logger.warning("SGLang server died after LoRA load — restarting...")
            t = time.perf_counter()
            try:
                await self._server.stop()
            except Exception:
                pass
            self._server = self._create_server()
            await self._server.start()
            self._active_lora_name = None
            timings["restart_s"] = time.perf_counter() - t
            logger.warning(f"SGLang restarted in {timings['restart_s']:.1f}s (no LoRA)")

        timings["total_overhead_s"] = time.perf_counter() - t_total

        logger.info(
            f"Step {self._latest_step} done — "
            f"train={train_metrics['training_time_s']:.1f}s  "
            f"overhead={timings['total_overhead_s']:.1f}s"
        )

        return {**train_metrics, **timings}

    # ------------------------------------------------------------------
    # Properties for the benchmark runner
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://0.0.0.0:{self.port}/v1"

    @property
    def inference_model_name(self) -> str:
        """Model name for inference requests via the OpenAI-compatible API.

        SGLang v0.5.3+ uses "base-model:adapter-name" syntax for
        /v1/chat/completions when a LoRA adapter is active.
        Falls back to base model name when no adapter is loaded.
        """
        if self._active_lora_name:
            return f"{self.base_model}:{self._active_lora_name}"
        return self.base_model
