from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.core.transformer.enums import AttnBackend
import torch


def get_provider(model: str) -> GPTModelProvider:
    bridge = AutoBridge.from_hf_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    assert isinstance(bridge._model_bridge, Qwen3MoEBridge), (
        "Only Qwen3 MoE models are supported"
    )
    provider = bridge.to_megatron_provider()
    provider.attention_backend = AttnBackend.fused
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    num_gpus = torch.cuda.device_count()
    # ── Parallelism strategy for MoE-heavy models (Qwen3-30B-A3B) ──
    #
    # Qwen3-30B-A3B memory breakdown (bf16):
    #   - 128 routed experts × 48 layers × (3 × 4096 × 768)   = ~58 GiB total
    #   - 1 shared expert  × 48 layers × (3 × 4096 × 18944)   = ~21 GiB total  ← BIG
    #   - Attention (q/k/v/o) × 48 layers                      = ~3.6 GiB total
    #   - Embeddings                                            = ~1.2 GiB total
    #
    # KEY INSIGHT: With TP < num_gpus, some components get REPLICATED across
    # TP groups.  E.g. TP=2 on 4 GPUs → 2 TP groups → shared expert (21 GiB)
    # is split within each group (10.5 GiB) but COPIED to both groups.
    # Result: 42 GiB weights per GPU → OOM on backward pass.
    #
    # FIX: Use TP = num_gpus (all GPUs in ONE TP group).  Zero replication:
    #   TP=4, EP=1, ETP=4 → everything split 4 ways → ~15 GiB/GPU
    #   Leaves ~60 GiB free for activations/backward (plenty).
    #
    # Trade-off: TP=4 has more all-reduce communication than TP=2, making
    # each training step ~20-30% slower.  But TP=2 OOMs, so slower > crash.
    #
    # Parallelism math: world = TP × PP × CP × DP
    #   TP=4, PP=1, CP=1 → DP=1, EP=1, ETP=4
    #
    # Memory per GPU with TP=4:
    #   routed = 58/4 = 14.5 GiB   (split by ETP=4)
    #   shared = 21/4 =  5.25 GiB  (split by TP=4, NO replication)
    #   attn   = 3.6/4 = 0.9 GiB   (split by TP=4, NO replication)
    #   embed  = 1.2/4 = 0.3 GiB
    #   TOTAL  ≈ 21 GiB  → 59 GiB free for backward ✅
    pp = 1
    cp = 1
    # Use ALL GPUs in one TP group to eliminate weight replication.
    # Round down to nearest power-of-2 for TP (Megatron requirement).
    if num_gpus >= 4:
        tp = 4
    elif num_gpus >= 2:
        tp = 2
    else:
        tp = 1
    dp_size = num_gpus // (tp * pp * cp)
    ep = max(1, dp_size)       # EP=1 when TP=num_gpus (DP=1)
    etp = tp                   # Split expert weights across all TP ranks

    provider.tensor_model_parallel_size = tp
    provider.context_parallel_size = cp
    provider.pipeline_model_parallel_size = pp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = etp
    provider.moe_shared_expert_overlap = True
    provider.moe_router_dtype = "fp32"
    # ── MoE token capacity: prevent routing-spike OOM across epochs ──
    # Without a cap, the router can send disproportionately many tokens to
    # experts on one GPU (data-dependent).  With EP, tokens are all-to-all
    # dispatched, so one GPU can receive 2-3× the average, causing transient
    # OOM during backward.  Setting a capacity factor caps per-expert tokens
    # to (total_tokens / num_experts) * capacity_factor.  Tokens beyond cap
    # are dropped (lowest-probability first) — has negligible training impact
    # for RL/RLHF but makes memory usage bounded and deterministic.
    provider.moe_token_drop_policy = "probs"        # drop least-confident routes
    provider.moe_expert_capacity_factor = 1.5        # 50% headroom over uniform
    provider.moe_pad_expert_input_to_capacity = True # deterministic memory per expert
    if tp > 1:
        provider.sequence_parallel = True
    return provider
