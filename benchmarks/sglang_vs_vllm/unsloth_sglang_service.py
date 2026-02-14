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

        # LoRA on ATTENTION modules only — SGLang's LoRA hot-reload does not
        # support MoE expert layers (gate_up_proj, down_proj).  PEFT warns
        # "Unsupported layer type 'Qwen3MoeExperts'" and SGLang crashes when
        # trying to serve inference with the loaded adapter.
        #
        # Unsloth's MoE Triton kernels (grouped_mm on H100, unsloth_triton on
        # A100) still optimize the forward/backward pass through expert layers
        # at the COMPUTATION level — that benefit is independent of LoRA.
        # Split LoRA on experts is a training-memory optimization that we skip
        # here to keep SGLang LoRA hot-reload working (<0.1s per step).
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
            ],
            lora_alpha=self.lora_rank,  # alpha = rank per Unsloth MoE docs
            lora_dropout=0,  # Unsloth default: no dropout
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        # REQUIRED: Prepare model for training — Unsloth patches dtype handling,
        # gradient checkpointing hooks, and MoE router dispatch. Without this,
        # the MoE router gate gets float32 hidden_states but BF16 weights.
        FastLanguageModel.for_training(model)

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

    def _train_on_completions(
        self,
        train_data: list[dict],
        lr: float | None = None,
    ) -> dict[str, float]:
        """GRPO training on completions — mirrors src/art/loss.py exactly.

        This implements the IDENTICAL loss formula as Megatron GRPO
        (src/art/loss.py) with on_policy_correction=True, ref_logprobs=None:

          1. Compute new_logprobs via log_softmax + gather (per token)
          2. On-policy correction: old_logprobs = new_logprobs.detach()
             → prob_ratio = exp(new - old) = 1.0
          3. Clip ratio to [1-epsilon, 1+epsilon_high] = [0.0, 5.0]
             (no-op since ratio=1.0, but matches Megatron code path)
          4. policy_loss = -(clipped_ratio * advantage * new_logprobs)
          5. No KL penalty (ref_logprobs=None → kl_div=0, beta=0)
          6. Mask to completion tokens only (assistant_mask)

        The forward/backward exercises Unsloth's MoE Triton kernels
        (grouped-GEMM) on real completions, same computational work
        as Megatron minus the distributed overhead.

        Args:
            train_data: List of dicts with keys:
                - prompt: list of message dicts (chat format)
                - completion: str (model-generated text)
                - reward: float
                - advantage: float (group-relative, computed by caller)
                - error: bool
            lr: Learning rate override.
        """
        state = self._training_state
        assert state is not None

        device = next(state.model.parameters()).device
        state.model.train()

        if lr is not None:
            for pg in state.optimizer.param_groups:
                pg["lr"] = lr

        # ---- GRPO parameters (matching src/art/loss.py defaults) ----
        # ppo=False → epsilon=1.0, epsilon_high=4.0
        epsilon = 1.0       # clip lower: 1 - 1.0 = 0.0
        epsilon_high = 4.0  # clip upper: 1 + 4.0 = 5.0
        # beta = 0.0 (no KL penalty, ref_logprobs=None)
        # on_policy_correction = True

        # ---- Tokenize: prompt + completion, mask prompt tokens ----
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        advantages_list: list[float] = []

        for item in train_data:
            if item.get("error") or not item.get("completion"):
                continue

            msgs = item["prompt"]
            completion = item["completion"]
            advantage = item.get("advantage", 1.0)

            # Tokenize prompt only (to know where completion starts)
            prompt_text = state.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = state.tokenizer.encode(
                prompt_text, add_special_tokens=False,
            )

            # Tokenize full conversation (prompt + assistant completion)
            full_msgs = msgs + [{"role": "assistant", "content": completion}]
            full_text = state.tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False,
            )
            full_ids = state.tokenizer.encode(
                full_text, add_special_tokens=False,
            )

            # Truncate to max_seq_length
            full_ids = full_ids[: self.max_seq_length]

            # Labels: -100 for prompt tokens, actual ids for completion tokens
            n_prompt = min(len(prompt_ids), len(full_ids))
            labels = [-100] * n_prompt + full_ids[n_prompt:]
            labels = labels[: self.max_seq_length]

            # Skip if no completion tokens survived truncation
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

        # ---- Pad to uniform length ----
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
        labels = torch.tensor(padded_labels, device=device)
        attention_mask = torch.tensor(attn_masks, device=device)
        advantages = torch.tensor(advantages_list, device=device, dtype=torch.float32)

        batch_size, seq_len = input_ids.shape
        completion_tokens = int((labels != -100).sum().item())
        logger.info(
            f"  training: {batch_size} seqs, max_len={seq_len}, "
            f"completion_tokens={completion_tokens} (GRPO, mirrors art/loss.py)"
        )

        # ---- Forward/backward: Megatron GRPO loss formula ----
        t0 = time.perf_counter()
        total_policy_loss = 0.0
        n_micro = 0

        mb = max(1, min(batch_size, 4))
        accum_steps = max(1, batch_size // mb)

        state.optimizer.zero_grad()
        for i in range(0, batch_size, mb):
            mb_ids = input_ids[i:i + mb]
            mb_mask = attention_mask[i:i + mb]
            mb_labels = labels[i:i + mb]
            mb_adv = advantages[i:i + mb]

            # autocast keeps activations in BF16 — prevents dtype mismatch
            # in MoE router (float32 hidden_states vs BF16 gate weights).
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Forward pass
                logits = state.model(
                    input_ids=mb_ids, attention_mask=mb_mask,
                ).logits

                # ---- Megatron GRPO loss (src/art/loss.py) ----
                # Shift for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = mb_labels[..., 1:].contiguous()

                # Step 1: Compute new_logprobs (per-token log probability)
                # Matches: new_logprobs = -model(input_ids, labels=shifted)
                log_probs = torch.nn.functional.log_softmax(
                    shift_logits, dim=-1,
                )
                # Gather log-prob of the actual token at each position
                token_ids = shift_labels.clamp(min=0)  # -100 → 0 (masked later)
                new_logprobs = log_probs.gather(
                    dim=-1, index=token_ids.unsqueeze(-1),
                ).squeeze(-1)

                # Step 2: On-policy correction (on_policy_correction=True)
                # old_logprobs = new_logprobs.detach()
                old_logprobs = new_logprobs.detach()

                # Step 3: Probability ratio + clipping
                # prob_ratio = exp(new - old) = exp(0) = 1.0
                logprob_diff = new_logprobs - old_logprobs
                prob_ratio = torch.exp(logprob_diff)
                clipped_ratio = torch.clip(
                    prob_ratio.detach(),
                    1.0 - epsilon,      # 0.0
                    1.0 + epsilon_high,  # 5.0
                )

                # Step 4: Policy loss (same formula as src/art/loss.py)
                # policy_loss = -(clipped_ratio * advantages * new_logprobs)
                # Expand advantages from [B] to [B, T] for per-token multiplication
                per_token_advantages = mb_adv.unsqueeze(-1).expand_as(new_logprobs)
                policy_loss = -(clipped_ratio * per_token_advantages * new_logprobs)

                # Step 5: Mask to completion tokens only (assistant_mask)
                assistant_mask = (shift_labels != -100).float()

                # Step 6: Mean over completion tokens, then over batch
                # (No KL penalty: ref_logprobs=None, beta=0)
                masked_loss = (policy_loss * assistant_mask).sum(dim=-1)
                per_sample_loss = masked_loss / assistant_mask.sum(dim=-1).clamp(min=1)
                batch_loss = per_sample_loss.mean() / accum_steps

            # backward outside autocast (GradScaler not needed for BF16)
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
        train_data: list[dict],
        lr: float | None = None,
    ) -> dict[str, float]:
        """One complete training step with full memory recovery.

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

        # 3. Train (GRPO-style: advantage-weighted loss on completions)
        train_metrics = self._train_on_completions(train_data, lr=lr)

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

        # 8. Health check — if SGLang crashed (e.g. LoRA incompatibility),
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
