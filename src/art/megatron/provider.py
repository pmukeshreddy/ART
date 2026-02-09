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
    # Strategy: maximize TP (to avoid weight replication OOM on backward),
    # then set EP to largest valid divisor of the remaining DP dimension.

    pp = 1
    cp = 1
    num_experts = 128  # Qwen3-30B-A3B

    # Pick largest power-of-2 TP that divides num_gpus
    tp = 1
    for candidate in [8, 4, 2, 1]:
        if num_gpus % candidate == 0 and num_gpus >= candidate:
            tp = candidate
            break

    dp_size = num_gpus // (tp * pp * cp)

    # EP must divide both dp_size AND num_experts
    # Pick the largest valid EP for maximum expert distribution
    ep = 1
    for candidate in range(dp_size, 0, -1):
        if dp_size % candidate == 0 and num_experts % candidate == 0:
            ep = candidate
            break

    # ETP: split expert weights across TP ranks
    etp = tp

    provider.tensor_model_parallel_size = tp
    provider.context_parallel_size = cp
    provider.pipeline_model_parallel_size = pp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = etp
    provider.moe_shared_expert_overlap = True
    provider.moe_router_dtype = "fp32"

    # ── MoE token capacity: prevent routing-spike OOM ──
    provider.moe_token_drop_policy = "probs"
    provider.moe_expert_capacity_factor = 1.0
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
