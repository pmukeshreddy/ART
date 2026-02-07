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
    # The SHARED EXPERT is the hidden memory hog (18944 vs 768 intermediate).
    # With TP=1, it's fully replicated → 21 GiB per GPU.
    # With TP=2, it's split → 10.5 GiB per GPU (saves 10.5 GiB!).
    #
    # TP=1,EP=4 (BAD):  routed=29 + shared=21 + attn=3.6 = ~54 GiB → OOM on backward
    # TP=2,EP=2,ETP=2:  routed=29 + shared=10.5 + attn=1.8 = ~42 GiB → 33 GiB free ✅
    #
    # expert_tensor_parallel_size (ETP) splits each expert's weights across TP
    # ranks, so routed expert memory stays ~29 GiB per GPU in both configs.
    # The win comes from TP=2 splitting the shared expert and attention.
    #
    # Parallelism math: world=TP×PP×CP×DP, EP≤DP, ETP≤TP
    #   TP=2, PP=1, CP=1 → DP=num_gpus/2 → EP=DP, ETP=TP
    pp = 1
    cp = 1
    if num_gpus >= 2 and num_gpus % 2 == 0:
        tp = 2  # Split shared expert + attention across 2 GPUs
    else:
        tp = 1
    dp_size = num_gpus // (tp * pp * cp)
    ep = max(1, dp_size)
    etp = tp  # Match expert-TP to TP so expert weights are also split

    provider.tensor_model_parallel_size = tp
    provider.context_parallel_size = cp
    provider.pipeline_model_parallel_size = pp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = etp
    provider.moe_shared_expert_overlap = True
    provider.moe_router_dtype = "fp32"
    if tp > 1:
        provider.sequence_parallel = True
    return provider
