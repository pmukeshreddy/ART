"""SGLang + Megatron backend for ART.

This module provides SGLangMegatronBackend, which combines:
- SGLang for inference (RadixAttention prefix caching, zero-overhead scheduler)
- Megatron-Core for distributed training (expert parallelism, tensor parallelism)

This is the SGLang equivalent of MegatronBackend (which uses vLLM for inference).

Usage:
    from art.megatron import SGLangMegatronBackend
    
    backend = SGLangMegatronBackend()
    await backend.register(model)
    result = await backend.train(model, trajectory_groups)
"""

from mp_actors import move_to_child_process

from ..local.backend import LocalBackend
from ..local.service import ModelService
from ..model import Model, TrainableModel
from ..utils.output_dirs import get_model_dir


class SGLangMegatronBackend(LocalBackend):
    """Backend using SGLang for inference and Megatron for distributed training.
    
    This is a drop-in replacement for MegatronBackend that uses SGLang instead
    of vLLM for the inference server.
    
    Benefits over MegatronBackend (vLLM):
        - RadixAttention: Better prefix caching for multi-turn agent trajectories
        - Zero-overhead scheduler: Lower latency for RL rollouts
        - Hot-reload: No server restart needed for weight sync
    
    Args:
        in_process: Run service in-process (default: False)
        path: Path for checkpoints/logs (default: ".art")
        sglang_config: SGLang server configuration (optional)
    
    Example:
        backend = SGLangMegatronBackend()
        await backend.register(model)
        result = await backend.train(model, trajectory_groups)
    """
    
    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
        sglang_config: dict | None = None,
    ) -> None:
        """Initialize SGLangMegatronBackend.
        
        Args:
            in_process: Run in-process (mainly for debugging)
            path: Checkpoint/log directory
            sglang_config: Optional dict with SGLang configuration:
                - mem_fraction_static: GPU memory fraction (default: 0.85)
                - server_timeout: Startup timeout in seconds (default: 300)
                - tensor_parallel_size: TP size for large models (default: auto)
                - sglang_python_path: Path to SGLang venv python (auto-detect)
                - preserve_cache_during_training: Keep server running during training
                  to preserve RadixAttention cache (default: True, requires spare GPUs)
                - sglang_gpu_ids: Explicit GPU IDs for SGLang (e.g., [0])
                  When set, remaining GPUs are used for Megatron training
        """
        super().__init__(in_process=in_process, path=path)
        self._sglang_config = sglang_config or {}

    def _model_inference_name(self, model: Model, step: int | None = None) -> str:
        """Return the inference name for a model checkpoint.
        
        For SGLang with RadixCache preservation, we use a FIXED adapter name
        (`model.name@latest`) instead of step-based names (`model.name@17`).
        
        This is critical for cache hits:
        - Same adapter name + same prefix = cache HIT
        - Different adapter name = new radix tree branch = cache MISS
        
        The step parameter is ignored because SGLang hot-reloads weights
        into the same `@latest` adapter slot.
        """
        # Always return fixed name for RadixCache consistency
        return f"{model.name}@latest"

    async def _get_service(self, model: TrainableModel) -> ModelService:
        """Get or create the SGLang + Megatron service.
        
        Overrides LocalBackend._get_service to use SGLangMegatronService
        instead of UnslothService.
        """
        # Import here to avoid circular imports and unnecessary dependencies
        from ..dev.get_model_config import get_model_config
        from .sglang_service import SGLangMegatronService, SGLangConfig

        if model.name not in self._services:
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
            )
            
            # Build SGLang config from dict
            sglang_cfg = SGLangConfig(
                mem_fraction_static=self._sglang_config.get("mem_fraction_static", 0.85),
                server_timeout=self._sglang_config.get("server_timeout", 300.0),
                tensor_parallel_size=self._sglang_config.get("tensor_parallel_size"),  # None = auto
                sglang_python_path=self._sglang_config.get("sglang_python_path"),
                log_level=self._sglang_config.get("log_level", "info"),
                # Cache preservation settings
                preserve_cache_during_training=self._sglang_config.get("preserve_cache_during_training", True),
                sglang_gpu_ids=self._sglang_config.get("sglang_gpu_ids"),  # None = auto
                training_gpu_ids=self._sglang_config.get("training_gpu_ids"),  # None = auto (use non-SGLang GPUs)
                # Cache method: "verl" (default) | "freeze" | "sleep_wake" | "hot_reload" | "restart"
                cache_method=self._sglang_config.get("cache_method", "verl"),
                # Performance optimization flags
                schedule_policy=self._sglang_config.get("schedule_policy", "lpm"),
                chunked_prefill_size=self._sglang_config.get("chunked_prefill_size"),
                enable_overlap_schedule=self._sglang_config.get("enable_overlap_schedule", False),
                enable_torch_compile=self._sglang_config.get("enable_torch_compile", False),
                cuda_graph_max_bs=self._sglang_config.get("cuda_graph_max_bs", 128),
                attention_backend=self._sglang_config.get("attention_backend", "flashinfer"),
                enable_cache_report=self._sglang_config.get("enable_cache_report", True),
            )
            
            self._services[model.name] = SGLangMegatronService(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
                sglang_config=sglang_cfg,
            )
            
            if not self._in_process:
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="sglang-megatron-service",
                )
        
        return self._services[model.name]
