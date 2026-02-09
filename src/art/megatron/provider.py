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

    # ── Auto-detect parallelism for MoE models ──
    #
    # Constraints:
    #   world_size = TP * PP * CP * DP
    #   EP must divide DP (expert parallelism lives inside data-parallel group)
    #   num_experts (128 for Qwen3-30B-A3B) must be divisible by EP
    #   TP should be power-of-2 for Megatron all-reduce efficiency
    #   Higher TP = less weight replication = less memory, but more communication
    #
    # Strategy: maximize EP first (to distribute experts and reduce per-GPU
    # memory), then use TP for remaining parallelism on larger clusters.

    pp = 1
    cp = 1
    num_experts = 128  # Qwen3-30B-A3B

    # ── MoE parallelism strategy ──
    #
    # For MoE models, EP (expert parallelism) is MORE important than TP
    # because it distributes experts across GPUs, dramatically reducing
    # per-GPU memory for activations and dispatcher buffers.
    #
    # ms-swift best practice for Qwen3-30B-A3B on 2 GPUs (50 GiB/GPU):
    #   TP=1, EP=2 → each GPU holds 64 of 128 experts
    #
    # Previous (broken) strategy: TP=2, EP=1 → each GPU holds ALL 128
    # experts (TP-sharded), causing 134 GiB usage and OOM on backward.
    #
    # Strategy: maximize EP first (to distribute experts), then use TP
    # for the remaining parallelism if GPUs are available.

    if num_gpus <= 4:
        # Small GPU counts: prioritize EP over TP
        # EP gets all GPUs, TP=1 (no tensor-parallel overhead)
        tp = 1
        dp_size = num_gpus // (tp * pp * cp)
        ep = 1
        for candidate in range(dp_size, 0, -1):
            if dp_size % candidate == 0 and num_experts % candidate == 0:
                ep = candidate
                break
        etp = 1
    else:
        # Large GPU counts (8+): split between TP and EP
        # e.g. 8 GPUs → TP=4, DP=2, EP=2 (matches ms-swift 8-GPU recipe)
        tp = 1
        for candidate in [4, 2, 1]:
            if num_gpus % candidate == 0 and num_gpus >= candidate:
                tp = candidate
                break
        dp_size = num_gpus // (tp * pp * cp)
        ep = 1
        for candidate in range(dp_size, 0, -1):
            if dp_size % candidate == 0 and num_experts % candidate == 0:
                ep = candidate
                break
        etp = tp

    provider.tensor_model_parallel_size = tp
    provider.context_parallel_size = cp
    provider.pipeline_model_parallel_size = pp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = etp
    provider.moe_shared_expert_overlap = True
    provider.moe_router_dtype = "fp32"
    provider.moe_grouped_gemm = True  # ms-swift best practice: fused expert kernels

    # ── MoE token capacity: prevent routing-spike OOM ──
    provider.moe_token_drop_policy = "probs"
    provider.moe_expert_capacity_factor = 1.0  # 1.5 causes OOM on 2-GPU; 1.0 matches uniform distribution
    provider.moe_pad_expert_input_to_capacity = True

    if tp > 1:
        provider.sequence_parallel = True

    # Log chosen config for debugging
    dp_actual = num_gpus // (tp * pp * cp)
    print(
        f"[Megatron parallelism] GPUs={num_gpus} TP={tp} PP={pp} CP={cp} "
        f"DP={dp_actual} EP={ep} ETP={etp}"
    )

    return provider
