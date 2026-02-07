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
    # Qwen3-30B-A3B: ~83% of params are experts (128 experts × 24 MoE layers).
    # Memory bottleneck is EXPERT weights, not attention.
    #
    # TP=2,EP=2 (BAD): each GPU holds 64 experts → ~29 GB expert weights → OOM
    # TP=1,EP=4 (GOOD): each GPU holds 32 experts → ~14.5 GB expert weights → fits
    #
    # TP=1 means attention layers are replicated (not split) across GPUs, but
    # Qwen3's attention is only ~5B params, so the +5 GB overhead is dwarfed
    # by the -14.5 GB savings from halving expert count per GPU.
    #
    # Parallelism math: world_size = TP × PP × CP × DP_total, EP ≤ DP_total
    #   TP=1, PP=1, CP=1 → DP_total = num_gpus → EP = num_gpus
    provider.tensor_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = num_gpus
    provider.expert_tensor_parallel_size = 1
    provider.moe_shared_expert_overlap = True
    provider.moe_router_dtype = "fp32"
    if provider.tensor_model_parallel_size > 1:
        provider.sequence_parallel = True
    return provider
