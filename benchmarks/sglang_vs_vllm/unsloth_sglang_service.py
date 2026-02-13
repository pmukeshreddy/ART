"""
Unsloth + SGLang service — self-contained MoE training with verl-style inference.

Unsloth is fully self-contained:
  - pip install --upgrade unsloth unsloth_zoo
  - transformers>=5.0.0 and trl>=0.27.1 (handled as Unsloth dependencies)
  - MoE Triton kernels, torch._grouped_mm, Split LoRA all baked in
  - Auto-selects best backend (grouped_mm / unsloth_triton / native_torch)
  - load_in_4bit=False required (MoE nn.Parameter doesn't support bnb 4bit yet)

Architecture:
  - SGLang server starts ONCE and NEVER restarts
  - Unsloth handles training in-process — no subprocess, no serialization
  - sleep/wake for memory management between inference and training
  - LoRA hot-reload for weight sync (<2s)
  - Full memory recovery every step

Training loop (per step):
  1. generate()     — SGLang active, KV cache + weights on GPU
  2. sleep()        — release KV cache AND weights
  3. Unsloth train  — FastLanguageModel + LoRA on MoE layers
  4. wake_up()      — restore base weights + KV cache
  5. load_lora()    — hot-reload ~2MB adapter

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
# Unsloth Training State — model persists across steps, offloads between
# ---------------------------------------------------------------------------

@dataclass
class UnslothTrainingState:
    """Holds the Unsloth model, tokenizer, and optimizer across training steps.

    Between steps the model is offloaded to CPU so SGLang can use the GPU.
    On the next training step it's reloaded — no re-initialization needed.
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
# Main Service
# ---------------------------------------------------------------------------

@dataclass
class UnslothSGLangService:
    """Self-contained Unsloth MoE training + SGLang inference.

    No ART dependency for training — just Unsloth + SGLang.

    Lifecycle per RL step:
      1. SGLang serves rollouts (inference)
      2. sleep()  — SGLang releases GPU memory
      3. Unsloth trains on rollout data (MoE Triton kernels)
      4. Save LoRA adapter
      5. wake_up()  — SGLang restores GPU memory
      6. load_lora()  — SGLang loads new adapter (<2s)
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

    # Unsloth config — all handled internally by Unsloth
    lora_rank: int = 16
    max_seq_length: int = 8192
    learning_rate: float = 5e-6
    # "auto" lets Unsloth pick: grouped_mm (H100+), unsloth_triton (A100), native_torch
    moe_backend: str = "auto"
    load_in_4bit: bool = False  # MoE nn.Parameter doesn't support bnb 4bit yet

    # Internal state
    _server: SGLangServer | None = None
    _training_state: UnslothTrainingState | None = None
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
            max_lora_rank=max(8, self.lora_rank),
            # MoE models in Transformers v5 use fused "gate_up_proj" instead of
            # separate "gate_proj"/"up_proj". Use "all" so SGLang accepts any
            # module the Unsloth adapter targets (including gate_up_proj).
            # SGLang auto-drops unsupported modules (e.g. embed_tokens for csgmv).
            lora_target_modules=["all"],
        ))

    async def start(self) -> float:
        """Start SGLang server. Called once at benchmark start."""
        self._server = self._create_server()
        startup = await self._server.start()
        logger.info(
            f"SGLang ready — {self.base_model} on :{self.port} "
            f"(startup {startup:.1f}s, will NOT restart)"
        )
        return startup

    async def stop(self) -> None:
        """Stop everything. Called once at benchmark end."""
        if self._server is not None:
            await self._server.stop()
            self._server = None
        if self._training_state is not None:
            try:
                self._training_state.offload_to_cpu()
                del self._training_state.model
                del self._training_state.optimizer
            except Exception:
                pass
            self._training_state = None
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
    # Unsloth model init — pure Unsloth, no ART
    # ------------------------------------------------------------------

    def _init_model(self) -> UnslothTrainingState:
        """Load model with FastLanguageModel + get_peft_model.

        Straight from the Unsloth MoE docs:
          https://unsloth.ai/docs/new/faster-moe

        Unsloth auto-selects the MoE backend based on hardware:
          - grouped_mm: H100, B200, T4+ (torch._grouped_mm)
          - unsloth_triton: A100, older PyTorch (custom Triton kernels)
          - native_torch: fallback (12x slower but VRAM savings still apply)
        """
        # Mock broken vLLM internals before Unsloth tries to import them
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

        # LoRA on MoE layers — gate_up_proj (fused in Transformers v5) and
        # down_proj are the expert layers. Attention modules also included.
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_up_proj", "down_proj",  # MoE expert layers (fused in TF v5)
            ],
            lora_alpha=self.lora_rank,  # alpha = rank per Unsloth MoE docs
            lora_dropout=0,  # Unsloth default: no dropout
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=self.learning_rate, betas=(0.9, 0.99), weight_decay=0.1,
        )

        n_params = sum(p.numel() for p in trainable)
        logger.info(f"Unsloth ready — {n_params:,} trainable params")

        return UnslothTrainingState(model=model, tokenizer=tokenizer, optimizer=optimizer)

    # ------------------------------------------------------------------
    # Training step — self-contained, no ART
    # ------------------------------------------------------------------

    def _train_on_texts(
        self,
        texts: list[str],
        lr: float | None = None,
    ) -> dict[str, float]:
        """Run one training step on a list of text completions.

        This is the actual forward+backward pass that exercises Unsloth's
        MoE Triton kernels (grouped-GEMM) and Split LoRA optimization.
        The texts come from the rollout completions.
        """
        state = self._training_state
        assert state is not None

        device = next(state.model.parameters()).device
        state.model.train()

        if lr is not None:
            for pg in state.optimizer.param_groups:
                pg["lr"] = lr

        # Tokenize
        encodings = state.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(device)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        batch_size, seq_len = input_ids.shape

        logger.info(f"  training: {batch_size} seqs, max_len={seq_len}")

        t0 = time.perf_counter()
        total_loss = 0.0
        n_micro = 0

        # Microbatch to fit in memory
        mb = max(1, min(batch_size, 4))
        accum_steps = max(1, batch_size // mb)

        state.optimizer.zero_grad()
        for i in range(0, batch_size, mb):
            mb_ids = input_ids[i:i + mb]
            mb_mask = attention_mask[i:i + mb]

            outputs = state.model(
                input_ids=mb_ids,
                attention_mask=mb_mask,
                labels=mb_ids,
            )
            loss = outputs.loss / accum_steps
            loss.backward()

            total_loss += outputs.loss.item()
            n_micro += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in state.model.parameters() if p.requires_grad],
            max_norm=0.1,
        )
        state.optimizer.step()
        state.optimizer.zero_grad()

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / max(n_micro, 1)
        total_tokens = int(attention_mask.sum().item())
        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        logger.info(
            f"  trained: loss={avg_loss:.4f}  {total_tokens / elapsed:.0f} tok/s  "
            f"VRAM={gpu_mem_gb:.1f}GB  {elapsed:.2f}s"
        )

        return {
            "loss": avg_loss,
            "training_time_s": elapsed,
            "tokens_per_sec": total_tokens / elapsed,
            "gpu_memory_gb": gpu_mem_gb,
            "total_tokens": total_tokens,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }

    # ------------------------------------------------------------------
    # LoRA save + hot-reload
    # ------------------------------------------------------------------

    def _save_lora(self, step: int) -> str:
        """Save LoRA adapter via PEFT save_pretrained (standard format)."""
        assert self._training_state is not None

        ckpt = os.path.join(self.output_dir, "checkpoints", f"{step:04d}")
        os.makedirs(ckpt, exist_ok=True)

        self._training_state.model.save_pretrained(ckpt)
        self._training_state.tokenizer.save_pretrained(ckpt)

        adapter = os.path.join(ckpt, "adapter_model.safetensors")
        if os.path.exists(adapter):
            mb = os.path.getsize(adapter) / 1e6
            logger.info(f"LoRA saved: {ckpt} ({mb:.1f} MB)")
        else:
            logger.warning(f"adapter_model.safetensors not found in {ckpt}")

        return ckpt

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
        texts: list[str],
        lr: float | None = None,
    ) -> dict[str, float]:
        """One complete training step with full memory recovery.

        Args:
            texts: Rollout completion texts to train on.
            lr: Learning rate override (optional).

        Returns:
            Dict of training + overhead metrics.
        """
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        # 1. Sleep SGLang — free GPU for training
        timings["sleep_s"] = await self.sleep()

        # 2. Get model on GPU
        t = time.perf_counter()
        if self._training_state is None:
            self._training_state = self._init_model()
        else:
            self._training_state.reload_to_gpu()
        timings["model_load_s"] = time.perf_counter() - t

        # 3. Train
        train_metrics = self._train_on_texts(texts, lr=lr)

        # 4. Save LoRA
        t = time.perf_counter()
        self._latest_step += 1
        ckpt = self._save_lora(self._latest_step)
        timings["save_s"] = time.perf_counter() - t

        # 5. Offload model to CPU
        self._training_state.offload_to_cpu()

        # 6. Wake SGLang — restore GPU
        timings["wake_s"] = await self.wake_up()

        # 7. Hot-reload LoRA
        timings["lora_reload_s"] = await self._load_lora(ckpt, self._latest_step)

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
