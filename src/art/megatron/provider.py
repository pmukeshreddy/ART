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
    provider.tensor_model_parallel_size = min(2, num_gpus)
    provider.context_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    # EP must divide the data-parallel size: DP = world_size / (TP * PP * CP)
    # With TP=2, PP=1, CP=1 on 4 GPUs: DP=2, so max EP=2.
    # Setting EP > DP would cause a Megatron assertion or silent misconfiguration.
    tp = provider.tensor_model_parallel_size
    pp = provider.pipeline_model_parallel_size
    cp = provider.context_parallel_size
    dp_size = num_gpus // (tp * pp * cp)
    provider.expert_model_parallel_size = max(1, dp_size)
    provider.expert_tensor_parallel_size = 1
    provider.moe_shared_expert_overlap = True
    provider.moe_router_dtype = "fp32"
    if provider.tensor_model_parallel_size > 1:
        provider.sequence_parallel = True
    return provider
