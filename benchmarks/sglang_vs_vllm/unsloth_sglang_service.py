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
  - Unsloth training runs in a PERSISTENT SUBPROCESS (model stays on GPU)
  - sleep/wake for memory management between inference and training
  - LoRA hot-reload for weight sync (<2s)
  - No CPU offload — model loaded once, stays on GPU permanently

Training loop (per step):
  1. generate()     — SGLang active, KV cache + weights on GPU
  2. sleep()        — release KV cache AND weights
  3. Unsloth train  — model already on GPU in subprocess (no reload!)
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
    """Persistent training worker — model stays on GPU permanently.

    Runs in a subprocess via mp_actors.move_to_child_process(). The model is
    loaded to GPU once at init_model() and NEVER offloaded to CPU, eliminating
    the ~50s per-step CPU<->GPU transfer overhead.

    Communication with the parent process is via mp_actors proxy (pickle over
    multiprocessing queues). Only lightweight data crosses the boundary:
      - train_data: list[dict] of prompts/completions/rewards (~100KB-1MB)
      - metrics: dict[str, float] (~1KB)
      - checkpoint paths: str
    """

    def __init__(
        self,
        base_model: str,
        output_dir: str,
        lora_rank: int = 16,
        max_seq_length: int = 8192,
        learning_rate: float = 5e-6,
        moe_backend: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.lora_rank = lora_rank
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.moe_backend = moe_backend
        self.load_in_4bit = load_in_4bit
        self._state: UnslothTrainingState | None = None

    def init_model(self) -> dict[str, Any]:
        """Load model to GPU. Called once in subprocess. Model stays permanently."""
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=self.lora_rank,
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

    def train_on_completions(
        self,
        train_data: list[dict],
        lr: float | None = None,
    ) -> dict[str, float]:
        """GRPO training on completions — mirrors src/art/loss.py exactly.

        Same loss formula as Megatron GRPO with on_policy_correction=True.
        Model is already on GPU (loaded once at init_model), so no reload needed.
        """
        state = self._state
        assert state is not None

        device = next(state.model.parameters()).device
        state.model.train()

        if lr is not None:
            for pg in state.optimizer.param_groups:
                pg["lr"] = lr

        # GRPO parameters (matching src/art/loss.py defaults)
        epsilon = 1.0
        epsilon_high = 4.0

        # Tokenize: prompt + completion, mask prompt tokens
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        advantages_list: list[float] = []

        for item in train_data:
            if item.get("error") or not item.get("completion"):
                continue

            msgs = item["prompt"]
            completion = item["completion"]
            advantage = item.get("advantage", 1.0)

            prompt_text = state.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = state.tokenizer.encode(
                prompt_text, add_special_tokens=False,
            )

            full_msgs = msgs + [{"role": "assistant", "content": completion}]
            full_text = state.tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False,
            )
            full_ids = state.tokenizer.encode(
                full_text, add_special_tokens=False,
            )

            full_ids = full_ids[: self.max_seq_length]

            n_prompt = min(len(prompt_ids), len(full_ids))
            labels = [-100] * n_prompt + full_ids[n_prompt:]
            labels = labels[: self.max_seq_length]

            if len(full_ids) <= n_prompt:
                continue

            input_ids_list.append(full_ids)
            labels_list.append(labels)
            advantages_list.append(advantage)

        if not input_ids_list:
            logger.warning("  no valid completions to train on")
            return {
                "loss": 0.0, "training_time_s": 0.0, "tokens_per_sec": 0.0,
                "gpu_memory_gb": 0.0, "total_tokens": 0, "batch_size": 0, "seq_len": 0,
            }

        # Pad to uniform length
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = state.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = state.tokenizer.eos_token_id or 0

        padded_ids, padded_labels, attn_masks = [], [], []
        for ids, labs in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            attn_masks.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(padded_ids, device=device)
        labels_t = torch.tensor(padded_labels, device=device)
        attention_mask = torch.tensor(attn_masks, device=device)
        advantages = torch.tensor(advantages_list, device=device, dtype=torch.float32)

        batch_size, seq_len = input_ids.shape
        completion_tokens = int((labels_t != -100).sum().item())
        logger.info(
            f"  training: {batch_size} seqs, max_len={seq_len}, "
            f"completion_tokens={completion_tokens} (GRPO, mirrors art/loss.py)"
        )

        # Forward/backward: Megatron GRPO loss formula
        t0 = time.perf_counter()
        total_policy_loss = 0.0
        n_micro = 0

        mb = max(1, min(batch_size, 4))
        accum_steps = max(1, batch_size // mb)

        state.optimizer.zero_grad()
        for i in range(0, batch_size, mb):
            mb_ids = input_ids[i:i + mb]
            mb_mask = attention_mask[i:i + mb]
            mb_labels = labels_t[i:i + mb]
            mb_adv = advantages[i:i + mb]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = state.model(
                    input_ids=mb_ids, attention_mask=mb_mask,
                ).logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = mb_labels[..., 1:].contiguous()

                log_probs = torch.nn.functional.log_softmax(
                    shift_logits, dim=-1,
                )
                token_ids = shift_labels.clamp(min=0)
                new_logprobs = log_probs.gather(
                    dim=-1, index=token_ids.unsqueeze(-1),
                ).squeeze(-1)

                old_logprobs = new_logprobs.detach()

                logprob_diff = new_logprobs - old_logprobs
                prob_ratio = torch.exp(logprob_diff)
                clipped_ratio = torch.clip(
                    prob_ratio.detach(),
                    1.0 - epsilon,
                    1.0 + epsilon_high,
                )

                per_token_advantages = mb_adv.unsqueeze(-1).expand_as(new_logprobs)
                policy_loss = -(clipped_ratio * per_token_advantages * new_logprobs)

                assistant_mask = (shift_labels != -100).float()

                masked_loss = (policy_loss * assistant_mask).sum(dim=-1)
                per_sample_loss = masked_loss / assistant_mask.sum(dim=-1).clamp(min=1)
                batch_loss = per_sample_loss.mean() / accum_steps

            batch_loss.backward()

            total_policy_loss += per_sample_loss.mean().item()
            n_micro += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in state.model.parameters() if p.requires_grad],
            max_norm=0.1,
        )
        state.optimizer.step()
        state.optimizer.zero_grad()

        elapsed = time.perf_counter() - t0
        avg_loss = total_policy_loss / max(n_micro, 1)
        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        logger.info(
            f"  trained: loss={avg_loss:.4f}  {completion_tokens / elapsed:.0f} tok/s  "
            f"VRAM={gpu_mem_gb:.1f}GB  {elapsed:.2f}s (GRPO, mirrors art/loss.py)"
        )

        return {
            "loss": avg_loss,
            "training_time_s": elapsed,
            "tokens_per_sec": completion_tokens / elapsed,
            "gpu_memory_gb": gpu_mem_gb,
            "total_tokens": completion_tokens,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }

    def save_lora(self, step: int) -> str:
        """Save LoRA adapter via PEFT save_pretrained (standard format)."""
        assert self._state is not None

        ckpt = os.path.join(self.output_dir, "checkpoints", f"{step:04d}")
        os.makedirs(ckpt, exist_ok=True)

        self._state.model.save_pretrained(ckpt)
        self._state.tokenizer.save_pretrained(ckpt)

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
            max_lora_rank=max(8, self.lora_rank),
            # MoE models in Transformers v5 use fused "gate_up_proj" instead of
            # separate "gate_proj"/"up_proj". Use "all" so SGLang accepts any
            # module the Unsloth adapter targets (including gate_up_proj).
            # SGLang auto-drops unsupported modules (e.g. embed_tokens for csgmv).
            lora_target_modules=["all"],
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
        train_data: list[dict],
        lr: float | None = None,
    ) -> dict[str, float]:
        """One complete training step — model stays on GPU in subprocess.

        Unlike the old approach (load from CPU → train → offload to CPU),
        the model is loaded ONCE on the first call and stays on GPU in
        the persistent subprocess. Subsequent steps skip the ~50s offload.

        GRPO-style: trains on completions with advantage-weighted loss,
        matching the Megatron GRPO training done by vLLM/SGLang backends.

        Args:
            train_data: List of dicts from _collect_completions, each with:
                - prompt: list of message dicts
                - completion: str (model-generated)
                - reward: float
                - advantage: float (group-relative)
                - error: bool
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
            logger.info(f"Unsloth worker model loaded — {n_params:,} trainable params (persistent)")
            self._worker_initialized = True
        timings["model_load_s"] = time.perf_counter() - t

        # 3. Train — model already on GPU in subprocess, no reload needed!
        train_metrics = await self._worker.train_on_completions(train_data, lr)

        # 4. Save LoRA
        t = time.perf_counter()
        self._latest_step += 1
        ckpt = await self._worker.save_lora(self._latest_step)
        timings["save_s"] = time.perf_counter() - t

        # 5. Wake SGLang — NO offload needed! Model stays on GPU in subprocess.
        timings["wake_s"] = await self.wake_up()

        # 6. Hot-reload LoRA
        timings["lora_reload_s"] = await self._load_lora(ckpt, self._latest_step)

        # 7. Health check — if SGLang crashed (e.g. LoRA incompatibility),
        #    restart it so the next rollout doesn't fail.
        if self._server is not None and not self._server.is_running:
            logger.warning("SGLang server died after LoRA load — restarting...")
            t = time.perf_counter()
            try:
                await self._server.stop()
            except Exception:
                pass
            self._server = self._create_server()
            await self._server.start()
            self._active_lora_name = None  # adapter lost on restart
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
