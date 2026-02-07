from mp_actors import move_to_child_process

from ..local.backend import LocalBackend
from ..local.service import ModelService
from ..model import TrainableModel
from ..utils.output_dirs import get_model_dir


class MegatronBackend(LocalBackend):
    """Backend using vLLM for inference and Megatron for distributed training.
    
    Args:
        in_process: Run service in-process (default: False)
        path: Path for checkpoints/logs (default: ".art")
        tensor_parallel_size: TP size for vLLM (default: auto-detect)
        vllm_config: Optional dict with vLLM configuration:
            - preserve_cache_during_training: Keep engine running during training
              to preserve prefix cache (default: True, requires spare GPUs)
            - vllm_gpu_ids: Explicit GPU IDs for vLLM (e.g., [0])
              When set, remaining GPUs are used for Megatron training
            - cache_method: Method for cache preservation (default: "http"):
              "http": Use /v1/load_lora_adapter API (requires GPU isolation)
              "sleep_wake": Use do_sleep/do_wake_up (offloads to CPU)
              "none": Always restart server (no cache preservation)
    """
    
    def __init__(
        self,
        *,
        in_process: bool = False,
        path: str | None = None,
        tensor_parallel_size: int | None = None,
        vllm_config: dict | None = None,
    ) -> None:
        super().__init__(in_process=in_process, path=path)
        self._tensor_parallel_size = tensor_parallel_size
        self._vllm_config = vllm_config or {}

    def _get_tensor_parallel_size(self) -> int:
        """Get tensor parallel size, auto-detect if not specified."""
        if self._tensor_parallel_size:
            return self._tensor_parallel_size
        try:
            import torch
            return torch.cuda.device_count() or 1
        except Exception:
            return 1

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config
        from .service import MegatronService, VLLMConfig

        if model.name not in self._services:
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
            )
            # Add tensor parallelism to engine args
            tp_size = self._get_tensor_parallel_size()
            if "engine_args" not in config:
                config["engine_args"] = {}
            config["engine_args"]["tensor_parallel_size"] = tp_size
            config["engine_args"]["max_model_len"] = config["engine_args"].get("max_model_len", 8192)
            
            # Build vLLM config for cache preservation
            vllm_cfg = VLLMConfig(
                tensor_parallel_size=self._vllm_config.get("tensor_parallel_size") or tp_size,
                preserve_cache_during_training=self._vllm_config.get("preserve_cache_during_training", True),
                vllm_gpu_ids=self._vllm_config.get("vllm_gpu_ids"),
                training_gpu_ids=self._vllm_config.get("training_gpu_ids"),
                cache_method=self._vllm_config.get("cache_method", "http"),
            )
            
            self._services[model.name] = MegatronService(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
                vllm_config=vllm_cfg,
            )
            if not self._in_process:
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="megatron-service",
                )
        return self._services[model.name]
